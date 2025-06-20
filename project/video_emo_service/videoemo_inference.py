import logging
from typing import Union, List, Dict

import torch
from tqdm import tqdm

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

# Configure module-level logger
logger = logging.getLogger(__name__)

class VideoEmotionDetectionYOLOInferencer:
    def __init__(
        self,
        path_to_yolo: str = "yolov11n-face.pt",
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_yolo = YOLO(path_to_yolo)
        self.tracker = DeepSort(max_age=30) 
        self.fer = EmotiEffLibRecognizer(engine="onnx", model_name=get_model_list()[0], device=device)

    def inference(
        self,
        video_path: str,
        segments: List[Dict[str, float]],
        frame_interval: int = 10  # frequency
    ) -> List[Dict[str, float]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_number = start_frame

            segment_face_data = {}

            while frame_number < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_number - start_frame) % frame_interval == 0:
                    timestamp = frame_number / fps

                    # Face detection
                    results = self.model_yolo.predict(source=frame, device=0, verbose=False)
                    detections = []
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().item()
                            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'face'))

                    tracks = self.tracker.update_tracks(detections, frame=frame)

                    face_images = []
                    track_ids = []
                    boxes = []

                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        track_id = track.track_id
                        ltrb = track.to_ltrb()
                        x1, y1, x2, y2 = map(int, ltrb)
                        face_image = frame[y1:y2, x1:x2]

                        if face_image.size == 0:
                            continue

                        face_images.append(face_image)
                        track_ids.append(track_id)
                        boxes.append((x1, y1, x2, y2))

                    if face_images:
                        _, scores_list = self.fer.predict_emotions(face_images, logits=True)

                        for i, (emotion_scores, track_id) in enumerate(zip(scores_list, track_ids)):
                            emotion_label = self.fer.idx_to_emotion_class[np.argmax(emotion_scores)]
                            emotion_confidence = float(np.max(emotion_scores))

                            if track_id not in segment_face_data:
                                segment_face_data[track_id] = {}

                            if emotion_label not in segment_face_data[track_id]:
                                segment_face_data[track_id][emotion_label] = {
                                    "count": 0,
                                    "confidence_sum": 0.0
                                }

                            segment_face_data[track_id][emotion_label]["count"] += 1
                            segment_face_data[track_id][emotion_label]["confidence_sum"] += emotion_confidence

                frame_number += 1

            segment["video_emotions"] = segment_face_data

        cap.release()
        return segments
