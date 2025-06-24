import torch
import gradio as gr
import json
import os

from asr_service import prepare_audio, AudioConversionError, GigaamCtcInferencer
from audio_emo_service import GigaEmotionInferencer
from video_emo_service import VideoEmotionDetectionYOLOInferencer

device = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# import models

transcriber = GigaamCtcInferencer(model_name="waveletdeboshir/gigaam-ctc-with-lm",
                                      device=device)

audio_emotion_model = GigaEmotionInferencer(device=device)

video_emotion_model = VideoEmotionDetectionYOLOInferencer(device=device, path_to_yolo='project/video_emo_service/yolov11n-face.pt')

def analyze(file_path: str, min_spk: int, max_spk: int, return_as_file: bool):
    try:
        audio_path = prepare_audio(file_path=file_path)
    except AudioConversionError as e:
        raise gr.Error(str(e))
    
    records = transcriber.transcribe_file(audio_path, min_speakers=max(min_spk, 1), max_speakers=max(max_spk, 1))

    records = audio_emotion_model.analyze_file(audio_path=audio_path, segments=records)

    if file_path.endswith(".mp4"):
        records = video_emotion_model.inference(video_path=file_path, segments=records)

    os.remove(audio_path)
    os.remove(file_path)

    if return_as_file:
        json_path = "/tmp/segments.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return records, json_path

    return records, None

def main():
    description = (
        "Загрузите аудио или видеофайл.\n"
    )

    iface = gr.Interface(
        fn=analyze,
        inputs=[
            gr.File(type="filepath", label="Аудио (wav/mp3) или видео (mp4)"),
            gr.Number(value=0, label="Минимальное число спикеров", precision=0),
            gr.Number(value=1, label="Максимальное число спикеров", precision=0),
            gr.Checkbox(label="Вернуть результат как файл", value=False),
        ],
        outputs=[
            gr.JSON(label="Транскрипция"),
            gr.File(label="Скачать JSON")
        ],
        title="Транскрипция и детекция эмоций аудио- и видеофайлов",
        description=description
    )

    iface.launch()

if __name__ == "__main__":
    main()
