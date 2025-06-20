# 🐦‍⬛ RAVEN: Recognition of Audio-Visual Emotional Nuances

RAVEN is a multimodal system for recognizing emotional nuances in both audio and video streams. The project leverages pretrained open-source models for text transcription and audio and video emotion recognition for further recording analysis, all wrapped in an interactive mircoservice app with Gradio web interface.

---

## Features

- **STT**: Get text transcription from audio or video.
- **Audio Emotion Recognition**: Extract emotional cues from audio using state-of-the-art pretrained open-source models.  
- **Video Emotion Recognition**: Analyze facial expressions in videos using computer vision techniques.  
- **Interactive UI**: Gradio-powered interface for real-time emotion detection.  
- **Dockerized Environment**: Fully containerized setup for reproducible deployments.  
- **Scalable Microservice Architecture**: Easily extendable if neede to use other models via FastAPI and Docker Compose with support for GPU acceleration.

---

## Prerequisites

- Git (to clone the repository)  
- Python 3.8 or higher  
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) for containerized setup  
- A Hugging Face API token (stored in `.env` file)  

---

## Usage 

More about project deployment can be read [here](project/README.md).