FROM audio-pipeline-base:latest
WORKDIR /app
COPY asr_service/ .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]