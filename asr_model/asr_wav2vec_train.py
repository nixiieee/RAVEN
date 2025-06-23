import numpy as np
import torch
import re
import evaluate
from datasets import load_dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2ProcessorWithLM,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import wandb
from torch.optim import AdamW

model_name_or_path = "bond005/wav2vec2-large-ru-golos-with-lm"
train_split = "train"
eval_split = "validation"
batch_size = 8
num_train_epochs = 5
output_dir = "./wav2vec2-ru-finetuned-lengths-freeze-augs"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("nixiieee/seul-game-processed-cut", token=True)


wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name_or_path)
model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path,
                                    #    torch_dtype=torch.bfloat16,
                                       ctc_loss_reduction="mean", 
                                       pad_token_id=processor.tokenizer.pad_token_id,)

for param in model.wav2vec2.parameters():
    param.requires_grad = False

for layer in model.wav2vec2.encoder.layers[-6:]:
    for param in layer.parameters():
        param.requires_grad = True

for name, param in model.wav2vec2.named_parameters():
    if "layer_norm" in name or "pos_conv_embed" in name:
        param.requires_grad = False

model.freeze_feature_encoder()

for name, param in model.named_parameters():
    print(f"{name:50} | requires_grad = {param.requires_grad}")

model.to(device)


def normalize_text(txt):
    txt = txt.lower()
    txt = re.sub(r"[^а-яё0-9\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["lengths"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(normalize_text(batch["text"])).input_ids
    return batch

dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    batched=False
)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2ProcessorWithLM
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

def compute_metrics(pred):
    pred_logits = pred.predictions  
    pred_str = processor.batch_decode(logits=pred_logits).text

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer" : cer}
    
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_train_epochs,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    weight_decay=0.005,
    logging_steps=5,
    save_total_limit=2,
    # bf16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    gradient_checkpointing=False, 
    report_to=["wandb"],
    run_name="wav2vec2-ru-finetune-lenghts-freeze",
    group_by_length=True,
    length_column_name="lengths",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

lm_head_params = list(model.lm_head.parameters())
encoder_params = list(model.wav2vec2.encoder.layers[-2:].parameters())

optimizer_grouped_parameters = [
    {"params": lm_head_params, "lr": 5e-5},
    {"params": encoder_params, "lr": 1e-6},
]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    optimizers=(AdamW(optimizer_grouped_parameters, weight_decay=0.005), None),
)

if __name__ == "__main__":
    wandb.login()
    wandb.init(project="wav2vec", name="processed-data-clean")
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    test_results = trainer.predict(dataset["test"])
    wandb.log(test_results.metrics)
    wandb.finish()
