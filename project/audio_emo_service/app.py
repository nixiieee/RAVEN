from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import json 
from typing import List, Dict
from audioemo_inference import GigaEmotionInferencer, prepare_audio, AudioConversionError
import tempfile
import os

device = "cuda"
HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

class Segment(BaseModel):
    segment: dict

app = FastAPI()

model: GigaEmotionInferencer | None = None
@app.on_event("startup")
async def load_model_on_startup():
    global model
    model = GigaEmotionInferencer(device=device, HF_TOKEN=HF_TOKEN)
    print("Audio Emotion model loaded and ready")

@app.post('/predict_audio_emotion')
async def predict_audio(
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
        audio_path = prepare_audio(file_path=tmp_path)
        out = model.inference(audio_path=audio_path, segments=segments_list)
    except AudioConversionError as e:
        raise e
    finally:
        os.remove(tmp_path)
    return {'segments': out}