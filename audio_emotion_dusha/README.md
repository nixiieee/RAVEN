
# Training & Evaluation of Emotion Classifiers

This directory contains the experimental and training pipeline for the audio emotion recognition component used in the general pipeline. It includes scripts for model training, evaluation, and data preprocessing. This is a research-focused module, not intended for production deployment.

Some information about Monte Carlo methods used to evaluate one of the models (whisper small classifier) can be found [here](https://github.com/nixiieee/whisper-emotion-classifier).

## 📁 Project Structure

```
audio_emotion_dusha/
├── .gitignore                  
├── preprocess_data.ipynb       # Notebook for preparing and cleaning audio emotion dataset DUSHA
├── requirements.txt            # Dependencies for training and evaluation
├── train_gigaam.py             # Training script for GigaAM encoder + MLP classifier
├── train_whisper_mlp.py        # Training script for Whisper encoder + MLP classifier
├── train_whisper_simple.py     # Baseline Whisper encoder-based classifier training
├── ved_evaluate.ipynb          # Evaluation notebook for testing models on different datasets
```

## 🤗 Hugging Face Resources

- Dataset: [`nixiieee/dusha_balanced`](https://huggingface.co/datasets/nixiieee/dusha_balanced)  
- Whisper small classifier: [`nixiieee/whisper-small-emotion-classifier-dusha`](https://huggingface.co/nixiieee/whisper-small-emotion-classifier-dusha)
- Whisper large-v3-turbo classifier: [`nixiieee/whisper-large-v3-emotion-classifier-dusha`](https://huggingface.co/nixiieee/whisper-large-v3-emotion-classifier-dusha)
- GigaAM classifier: [`nixiieee/gigaam-rnnt-emotion-classifier-dusha`](https://huggingface.co/nixiieee/gigaam-rnnt-emotion-classifier-dusha)

## 📊 Results 

| Model                            | Weighted Accuracy | Unweighted Accuracy | F1 Score |
|----------------------------------|-------------------|----------------------|----------|
| Whisper small base classifier    | 0.71              | 0.71                 | 0.74     |
| Whisper small MLP classifier     | 0.79              | 0.77                 | 0.80     |
| Whisper large-v3-turbo MLP       | 0.82              | 0.79                 | 0.81     |
| GigaAM MLP classifier            | 0.84              | 0.82                 | 0.84     |
| GigaAM-Emo (Sber)                | 0.72              | 0.87                 | 0.71     |

## ⚙️ Setup (Local)

```bash
uv venv --python 3.11
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

## 📚 Notebooks

* `preprocess_data.ipynb`: Used to format raw audio dataset DUSHA.
* `ved_evaluate.ipynb`: Used for analyzing model performance and comparing different approaches.

## </> Scripts

* `train_gigaam.py`: Trains an emotion classifier using the GigaAM architecture.
* `train_whisper_mlp.py`: Trains an emotion classifier using encoder from Whisper and MLP classifier.
* `train_whisper_simple.py`: Trains an emotion classifier using encoder from Whisper and AutoClassifier head from transformers library.