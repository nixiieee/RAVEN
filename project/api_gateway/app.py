from fastapi import FastAPI, HTTPException
import httpx
import json
import os

app = FastAPI()
ASR_URL = "http://asr:8000/transcribe"
AUDIO_EMO_URL = "http://audio_emo:8001/predict_audio_emotion"
VIDEO_EMO_URL = "http://video_emo:8002/predict_video_emotion"

async def run_analysis(tmp_path: str, min_spk: int, max_spk: int, is_video: bool):
    async with httpx.AsyncClient(timeout=3600) as client:
        filename = os.path.basename(tmp_path)
        with open(tmp_path, "rb") as f:
            files = {"file": (tmp_path, f, "application/octet-stream")}
            data = {"min_spk": min_spk, "max_spk": max_spk}
            asr_resp = await client.post(ASR_URL, files=files, data=data)
        if asr_resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ASR service error: {asr_resp.text}")
        segments = asr_resp.json().get("segments", [])

        # for seg in segments:
        #     payload = {"segment": json.dumps(seg)}
        #     with open(tmp_path, "rb") as f:
        #         files = {"file": (tmp_path, f, "application/octet-stream")}
        #         resp = await client.post(AUDIO_EMO_URL, files=files, data=payload)
        #     if resp.status_code != 200:
        #         seg["audio_error"] = resp.text
        #     else:
        #         seg.update(resp.json())

        audio_payload = {'segments': json.dumps(segments)}
        with open(tmp_path, "rb") as f:
            files = {"file": (tmp_path, f, "application/octet-stream")}
            audemo_resp = await client.post(AUDIO_EMO_URL, files=files, data=audio_payload)
        if audemo_resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Audio-emo error: {audemo_resp.text}")
        segments = audemo_resp.json()['segments']

        if is_video:
            video_payload = {"segments": json.dumps(segments)}
            with open(tmp_path, "rb") as f:
                files = {"file": (tmp_path, f, "application/octet-stream")}
                vid_resp = await client.post(VIDEO_EMO_URL, files=files, data=video_payload)
            if vid_resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Video-emo error: {vid_resp.text}")
            segments = vid_resp.json().get("segments", segments)

    return segments