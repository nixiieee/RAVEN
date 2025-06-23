import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import wandb
import re
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

model_name_or_path = "waveletdeboshir/gigaam-ctc-with-lm"
train_split = "train"
eval_split = "validation"
batch_size = 4
learning_rate = 5e-5
num_train_epochs = 3
output_dir = "./gigaam-ctc-finetuned"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("nixiieee/seul-game-processed-cut", token=True)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name_or_path, config=config, trust_remote_code=True
)

n_layers_to_unfreeze = 4

for name, param in model.named_parameters():
    # print(name)
    if "encoder" in name:
        param.requires_grad = False

layers_to_unfreeze = [str(i) for i in range(15 - n_layers_to_unfreeze, 16)]

for name, param in model.named_parameters():
    for layer_id in layers_to_unfreeze:
        if f"encoder.layers.{layer_id}." in name:
            param.requires_grad = True

model.to(device)

max_input_length = processor.feature_extractor.n_samples


def normalize_text(txt):
    txt = txt.lower()
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt.strip()


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["input_lengths"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_lengths[0]
    batch["labels"] = processor(text=normalize_text(batch["text"])).input_ids
    return batch


dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)


@dataclass
class DataCollatorCTCWithPadding:
    def __init__(self, processor, max_len, padding=True):
        self.processor: AutoProcessor = processor
        self.max_len = max_len
        self.padding: Union[bool, str] = padding

    def __call__(self, features):
        batch = dict()
        batch["input_features"] = pad_sequence(
            [torch.tensor(f["input_features"]).permute(1, 0) for f in features],
            padding_value=0,
            batch_first=True,
        ).permute(0, 2, 1)
        batch["input_lengths"] = torch.stack(
            [torch.tensor(f["input_lengths"], dtype=torch.long) for f in features], 0
        )
        batch["labels"] = pad_sequence(
            [torch.tensor(f["labels"], dtype=torch.long) for f in features],
            padding_value=-100,
            batch_first=True,
        )

        return batch


datacollator = DataCollatorCTCWithPadding(
    processor=processor, max_len=max_input_length, padding=True
)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = processor.batch_decode(
        pred_logits, beam_width=64, alpha=0.5, beta=0.5
    ).text
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    refs = processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    wer = wer_metric.compute(predictions=pred_ids, references=refs)
    cer = cer_metric.compute(predictions=pred_ids, references=refs)

    return {"wer": wer, "cer": cer}


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    weight_decay=0.005,
    logging_steps=30,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    report_to=["wandb"],
    run_name="gigaam-ctc-finetune",
    group_by_length=True,
    length_column_name="input_lengths",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=datacollator,
    compute_metrics=compute_metrics,
)


if __name__ == "__main__":
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    test_results = trainer.predict(dataset["test"])
