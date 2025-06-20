from transformers import (
    WhisperProcessor,
    WhisperModel,
    PreTrainedModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
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
import torch.nn.functional as F
import os
import wandb


class WhisperClassifier(nn.Module):
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


class WhisperForEmotionClassification(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self, config, model_name="openai/whisper-small", num_labels=5, dropout=0.2
    ):
        super().__init__(config)
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        hidden_size = config.hidden_size
        self.classifier = WhisperClassifier(
            hidden_size, num_labels=num_labels, dropout=dropout
        )
        self.post_init()

    def forward(self, input_features, attention_mask=None, labels=None):
        encoder_output = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = encoder_output.last_hidden_state
        logits = self.classifier(hidden_states, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


model_name = "openai/whisper-large-v3-turbo"
processor = WhisperProcessor.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = WhisperForEmotionClassification(config, num_labels=5, dropout=0.05)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)


# Set the seed
set_seed()

# unfreeze last 3 layers and classifier
for name, param in model.named_parameters():
    if name.startswith("classifier"):
        param.requires_grad = True
    elif any(
        f"encoder.block.{i}." in name
        for i in range(config.num_hidden_layers - 3, config.num_hidden_layers)
    ):
        param.requires_grad = True
    else:
        param.requires_grad = False


@dataclass
class DataCollatorForEncoderClassification:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        batch["labels"] = torch.tensor(
            [feature["labels"] for feature in features], dtype=torch.long
        )

        return batch


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = batch["emotion"]
    return batch


# load & process dataset
ds = load_dataset("nixiieee/dusha_balanced")
train_ds = ds["train"].map(
    prepare_dataset, remove_columns=["audio", "emotion"], num_proc=17
)
val_ds = ds["val"].map(
    prepare_dataset, remove_columns=["audio", "emotion"], num_proc=17
)
test_ds = ds["test"].map(
    prepare_dataset, remove_columns=["audio", "emotion"], num_proc=17
)

data_collator = DataCollatorForEncoderClassification(processor)

# Metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "balanced_accuracy": balanced_accuracy,
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }


project_name = "whisper-small-emotion-classifier-dusha"

wandb.login()
wandb.init(project=project_name, entity="xenz5240-higher-school-of-economics")

training_args = TrainingArguments(
    output_dir=f"./{project_name}",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    learning_rate=5e-5,
    num_train_epochs=3,
    fp16=True,
    report_to=["wandb"],
    push_to_hub=True,
    hub_model_id=f"nixiieee/{project_name}",
    load_best_model_at_end=True,
    metric_for_best_model="balanced_accuracy",
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
trainer.push_to_hub(commit_message="Training completed!")

test_results = trainer.predict(test_ds)
wandb.log(test_results.metrics)
