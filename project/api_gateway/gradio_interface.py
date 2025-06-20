import gradio as gr
import asyncio
import os
import json

from app import run_analysis

def analyze(file_path: str, min_spk: int, max_spk: int, return_as_file: bool):
    records = asyncio.run(run_analysis(file_path.name, min_spk, max_spk, file_path.name.endswith(".mp4")))
    os.remove(file_path)
    if return_as_file:
        json_path = "/tmp/segments.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return records, json_path
    return records, None

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
    description=(
        "Загрузите аудио или видеофайл.\n"
    )
)
iface.launch(server_name="0.0.0.0", server_port=7860)