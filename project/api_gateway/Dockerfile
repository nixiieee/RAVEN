FROM audio-pipeline-base:latest
WORKDIR /app
COPY api_gateway/ .

EXPOSE 7860
CMD ["python", "gradio_interface.py"]