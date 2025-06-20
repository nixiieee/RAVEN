# 🐦‍⬛ RAVEN: Recognition of Audio-Visual Emotional Nuances

RAVEN is a multimodal system for recognizing emotional nuances in both audio and video streams. The project leverages pretrained Hugging Face models for audio emotion recognition and computer vision techniques for video analysis, all wrapped in an interactive web interface built with Gradio.

---

## Features

- **Audio Emotion Recognition**: Extract emotional cues from audio using state-of-the-art pretrained models from Hugging Face.  
- **Video Emotion Recognition**: Analyze facial expressions and body language in videos using computer vision techniques.  
- **Interactive UI**: Gradio-powered interface for real-time emotion detection and parameter tuning.  
- **Dockerized Environment**: Fully containerized setup for reproducible deployments.  
- **Scalable Architecture**: Easily extendable via Docker Compose with support for GPU acceleration.

---

## Prerequisites

- Git (to clone the repository)  
- Python 3.8 or higher  
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) for containerized setup  
- A Hugging Face API token (stored in `.env` file)  

---

## Usage 

More about project deployment can be read [here](project/README.md).