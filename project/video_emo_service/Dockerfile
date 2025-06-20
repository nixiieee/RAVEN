FROM audio-pipeline-base:latest
WORKDIR /app
COPY video_emo_service/ .

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]