from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict
import json
import tempfile
import os
from videoemo_inference import VideoEmotionDetectionYOLOInferencer

class Segments(BaseModel):
    segments: list

app = FastAPI()

model: VideoEmotionDetectionYOLOInferencer | None = None

@app.on_event("startup")
async def load_model_on_startup():
    global model
    model = VideoEmotionDetectionYOLOInferencer(device='cuda')
    print("Video Emotion model loaded and ready")

@app.post('/predict_video_emotion')
async def predict_video(
    file: UploadFile = File(...),
    segments: str = Form(...)
):
    try:
        segments_list: List[Dict[str, float]] = json.loads(segments)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for 'segments'"}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        out = model.inference(video_path=tmp_path, segments=segments_list)
    finally:
        os.remove(tmp_path)
    return {'segments': out}