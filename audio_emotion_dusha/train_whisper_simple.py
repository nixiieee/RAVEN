import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoModelForAudioClassification, WhisperProcessor
import torch
import random
import numpy as np
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from sklearn.metrics import balanced_accuracy_score

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed()

@dataclass
class DataCollatorForEncoderClassification:
    processor: Any 

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')
        
        batch['labels'] = torch.tensor(
            [feature['labels'] for feature in features],
            dtype=torch.long
        )
        
        return batch

processed_ds_train = load_from_disk("/home/llm_agent/video_audio_pipeline/data/correct_dusha_dataset_train")
processed_ds_test = load_from_disk("/home/llm_agent/video_audio_pipeline/data/correct_dusha_dataset_test")
processed_ds_train = processed_ds_train.remove_columns(["audio", "emotion"])
processed_ds_test = processed_ds_test.remove_columns(["audio", "emotion"])

# model 
model = AutoModelForAudioClassification.from_pretrained("openai/whisper-small", num_labels=5)

for name, param in model.named_parameters():
    if "classifier" not in name and "projector" not in name and "final_layer_norm" not in name:
        param.requires_grad = False

# check
# for name, param in model.named_parameters():
#     print(f"{name:50} | requires_grad = {param.requires_grad}")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
data_collator = DataCollatorForEncoderClassification(processor)

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    
    return {
        "accuracy": accuracy["accuracy"],
        "balanced_accuracy": balanced_accuracy,
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }

training_args = TrainingArguments(
    output_dir="./whisper-emotion-classfication-layernorm",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    auto_find_batch_size=True,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_dir="./logs",
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True, 
    report_to="tensorboard",
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,     
    train_dataset=processed_ds_train,
    eval_dataset=processed_ds_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()