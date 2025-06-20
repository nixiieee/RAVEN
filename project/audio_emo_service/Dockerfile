FROM audio-pipeline-base:latest
WORKDIR /app
COPY audio_emo_service/ .

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]