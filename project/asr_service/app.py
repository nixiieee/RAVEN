from fastapi import FastAPI, File, UploadFile, Form
from asr_inference import WhisperASRInferencer, GigaamCtcInferencer, prepare_audio, AudioConversionError
import os
import tempfile

HF_TOKEN = ""
device = "cuda"

app = FastAPI()

transcriber: GigaamCtcInferencer | None = None

@app.on_event("startup")
async def load_model_on_startup():
    global transcriber
        # model = WhisperASRInferencer(
    #     model_name='openai/whisper-large-v3-turbo',
    #     device='cuda', language='Russian', task='transcribe', hf_token=HF_TOKEN
    # )
    transcriber = GigaamCtcInferencer(model_name="waveletdeboshir/gigaam-ctc-with-lm",
                                      device=device,
                                      hf_token=HF_TOKEN)
    print("ASR model loaded and ready")

@app.post('/transcribe')
async def transcribe(
    file: UploadFile = File(...),
    min_spk: int = Form(0),
    max_spk: int = Form(1)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        audio_path = prepare_audio(file_path=tmp_path)
        segments = transcriber.transcribe_file(audio_path, min_speakers=min_spk, max_speakers=max_spk)
    except AudioConversionError as e:
        raise e
    finally:
        os.remove(tmp_path)
    return {'segments': segments}