from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoProcessor,
    AutoModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import random
import numpy as np
from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import os
import wandb

class EmotionClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=5, dropout=0.2):
        super().__init__()
        self.pool_norm = nn.LayerNorm(hidden_size)
        self.pre_dropout = nn.Dropout(dropout)

        mid1 = max(hidden_size // 2, num_labels * 4)
        mid2 = max(hidden_size // 4, num_labels * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mid1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(mid1),
            nn.Linear(mid1, mid2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(mid2),
            nn.Linear(mid2, num_labels),
        )

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1, keepdim=True)
            masked = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / lengths
        else:
            pooled = hidden_states.mean(dim=1)
        x = self.pool_norm(pooled)
        x = self.pre_dropout(x)
        logits = self.classifier(x)
        return logits
    
class ModelForEmotionClassification(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self, config, model_name, num_labels=5, dropout=0.2
    ):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True).model.encoder
        # print(config.encoder)
        hidden_size = config.encoder['d_model']
        self.classifier = EmotionClassifier(
            hidden_size, num_labels=num_labels, dropout=dropout
        )
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,       # [B, T, feat_in]
        input_lengths: torch.Tensor,        # [B]
        attention_mask: torch.Tensor = None, # [B, T′] после subsampling
        labels: torch.Tensor = None         # [B]
    ) -> SequenceClassifierOutput:
        encoded, out_lens = self.encoder(input_features, input_lengths)
        hidden_states = encoded.transpose(1, 2)

        if attention_mask is None:
            max_t = hidden_states.size(1)
            attention_mask = (
                torch.arange(max_t, device=out_lens.device)
                .unsqueeze(0)
                .lt(out_lens.unsqueeze(1))
                .long()
            )

        logits = self.classifier(hidden_states, attention_mask=attention_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

# reproducibility
def set_seed(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class DataCollatorForEncoderClassification:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        tensors = [
            f["input_features"]
            if isinstance(f["input_features"], torch.Tensor)
            else torch.tensor(f["input_features"], dtype=torch.float32)
            for f in features
        ]
        seq_lens = [t.shape[0] for t in tensors]
        assert len(set(seq_lens)) == 1, "Все sequences в батче должны быть одинаковой длины"
        batch_inputs = torch.stack(tensors, dim=0)  # shape: [B, T, feat_in]

        batch_labels = torch.tensor(
            [f["labels"] for f in features], dtype=torch.long
        )
        batch_lens = torch.tensor(
            [f["input_lengths"] for f in features], dtype=torch.long
        )

        return {
            "input_features": batch_inputs,
            "input_lengths": batch_lens,
            "labels": batch_labels,
        }


def prepare_dataset(batch):
    audio = batch["audio"]
    processed = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    )   
    batch["input_features"] = processed["input_features"][0]
    batch["input_lengths"] = processed["input_lengths"][0]
    batch["labels"] = batch["emotion"]
    return batch

# init
model_name = 'waveletdeboshir/gigaam-rnnt'
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = ModelForEmotionClassification(config, model_name, num_labels=5, dropout=0.05)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)
set_seed()

num_hidden_layers = 16
for name, param in model.named_parameters():
    if name.startswith("classifier"):
        param.requires_grad = True
    elif any(
        f"encoder.layers.{i}." in name
        for i in range(num_hidden_layers - 3, num_hidden_layers)
    ):
        param.requires_grad = True
    else:
        param.requires_grad = False

# for name, param in model.named_parameters():
#     print(f"{name:50} | requires_grad = {param.requires_grad}")

# dataset
ds = load_dataset('nixiieee/dusha_balanced')

train_ds = ds['train'].map(prepare_dataset, remove_columns=['audio','emotion'], num_proc=1)
val_ds   = ds['val'].map(prepare_dataset, remove_columns=['audio','emotion'], num_proc=1)
test_ds  = ds['test'].map(prepare_dataset, remove_columns=['audio','emotion'], num_proc=1)

data_collator = DataCollatorForEncoderClassification(processor)

# metrics
accuracy_metric = evaluate.load('accuracy')
precision_metric = evaluate.load('precision')
recall_metric = evaluate.load('recall')
f1_metric = evaluate.load('f1')

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {
        'accuracy': accuracy_metric.compute(predictions=preds, references=labels)['accuracy'],
        'balanced_accuracy': balanced_accuracy_score(labels, preds),
        'precision': precision_metric.compute(predictions=preds, references=labels, average='macro')['precision'],
        'recall': recall_metric.compute(predictions=preds, references=labels, average='macro')['recall'],
        'f1': f1_metric.compute(predictions=preds, references=labels, average='macro')['f1'],
    }

# W&B
project_name = 'gigaam-rnnt-emotion'
wandb.login()
wandb.init(project=project_name)

training_args = TrainingArguments(
    output_dir=f'./{project_name}',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=20,
    learning_rate=5e-5,
    num_train_epochs=3,
    fp16=True,
    report_to=['wandb'],
    load_best_model_at_end=True,
    metric_for_best_model='balanced_accuracy',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()

test_results = trainer.predict(test_ds)
wandb.log(test_results.metrics)
