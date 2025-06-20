
# `audio_emotion_dusha`: Training & Evaluation Module

This directory contains the experimental and training pipeline for the audio emotion recognition component used in the general pipeline. It includes scripts for model training, evaluation, and data preprocessing. This is a research-focused module, not intended for production deployment.

## Structure

```
audio_emotion_dusha/
├── .gitignore                  # Files/folders to ignore in version control
├── preprocess_data.ipynb       # Notebook for preparing and cleaning audio emotion datasets
├── requirements.txt            # Dependencies for training and evaluation
├── train_gigaam.py             # Training script for GigaAM model
├── train_whisper_mlp.py        # Training script combining Whisper embeddings + MLP
├── train_whisper_simple.py     # Baseline Whisper-based classifier training
├── ved_evaluate.ipynb          # VED: evaluation notebook for validation/testing
```

## Setup (Local)

```bash
uv venv --python 3.11
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

## Notebooks

* `preprocess_data.ipynb`: Used to format raw audio dataset DUSHA.
* `ved_evaluate.ipynb`: Used for analyzing model performance and comparing different approaches.

## Scripts

* `train_gigaam.py`: Trains an emotion classifier using the GigaAM architecture.
* `train_whisper_mlp.py`: Trains an emotion classifier using encoder from Whisper and trains a separate MLP classifier.
* `train_whisper_simple.py`: Minimal setup for training directly on Whisper encoder outputs.

## Notes

* Results from this module are intended to be exported and used in the `project/audio_emo_service/` folder for inference.
* You can manually copy model weights or export ONNX versions depending on deployment strategy.