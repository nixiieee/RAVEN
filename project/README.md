
# Inference Module

This directory contains the main parts used for pipeline deployments. 

## 📁 Project Structure

```bash
project
├── api_gateway
│   ├── app.py
│   ├── Dockerfile
│   ├── gradio_interface.py
│   └── utils.py
├── asr_service
│   ├── app.py
│   ├── asr_inference.py
│   ├── Dockerfile
│   └──__init__.py
├── audio_emo_service
│   ├── app.py
│   ├── audioemo_inference.py
│   ├── Dockerfile
│   ├── __init__.py
│   └── models.py
├── video_emo_service
│   ├── app.py
│   ├── Dockerfile
│   ├── __init__.py
│   ├── videoemo_inference.py 
│   └── yolov11n-face.pt
├── docker-compose.yml
├── Dockerfile.base
├── gradio_pipeline.py
├── Makefile
└── requirements.txt
```

---

## ⚙️ Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/raven-audio-video-emotion.git
   cd raven-audio-video-emotion
   ```
2. **Create a .env file in the project root with your Hugging Face token**:
    ```bash
    HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```

## 📐 Structure

The main inference classes are stored in `*_inference.py` files. If you want to use different models then those defined in `app.py` files, you can change the script accordingly and just import a different class. The same goes when needeed to use model for inference outside of the main pipeline.

## 🚀 Quickstart

| Target              | Description                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------- |
| `build-base`        | Builds the base Docker image with Python and dependencies, injecting the Hugging Face token. |
| `build`             | Builds all Docker Compose services (can toggle BuildKit).                                    |
| `up`                | Starts services in detached mode (`docker compose up -d`).                                   |
| `down`              | Stops and removes containers and networks (`docker compose down`).                           |
| `clean`             | Removes unused Docker objects (images, containers, volumes).                                 |
| `run`               | Shortcut: runs `make build` followed by `make up`.                                           |
| `stop`              | Shortcut: runs `make down` and then `make clean`.                                            |
| `run-script-docker` | Builds the base image and runs `gradio_pipeline.py` inside a Docker container.               |

Example usage:

```bash
make build-base
make build
make up

# To stop and clean up
make stop
```

The pipeline for startup defined above is recommeneded. However the script can be run without docker using local environment (see below).

---

## ⚙️ Usage

Once the services are running, open the Gradio app in your browser at:
``` bash
http://localhost:7860
```
- Upload an audio or video file (`.mp3`, `.mp4` and `.wav` formats are supported).
- Adjust parameters (num of speakers, need for json-file output).
- View emotion recognition and audio transcription results.

## 🚀 Running locally (without docker)

### 1. Install dependencies

#### Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv --python 3.11
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

#### Alternatively, using pip:

```bash
pip install -r requirements.txt
```

### 2. Launch Gradio App

```bash
python3 project/gradio_pipeline.py
```

This will open a local Gradio interface at `http://localhost:7860` for real-time emotion recognition from audio or video (`.mp3`, `.mp4` and `.wav` formats are supported). 