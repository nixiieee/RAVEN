import os
import logging
from typing import Union, List, Tuple, Dict, Generator, Optional

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    WhisperProcessor,
    AutoModelForAudioClassification,
    AutoProcessor,
    AutoConfig
)
from pyannote.audio import Pipeline

import tempfile
import soundfile as sf
import ffmpeg
import librosa

from models import ModelForEmotionClassification

import gigaam

# Configure module-level logger
logger = logging.getLogger(__name__)

class AudioConversionError(Exception):
    """If conversion to WAV failed."""
    pass

def prepare_audio(file_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        (
            ffmpeg
            .input(file_path)
            .output(wav_path,
                    format='wav',
                    acodec='pcm_s16le',  # 16-bit PCM
                    ar='16000',          # 16 kHz
                    ac=1)                # моно
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        try: os.remove(wav_path)
        except OSError: pass

        stderr = e.stderr.decode('utf-8', errors='ignore')
        raise AudioConversionError(f"Failed to convert file:\n{stderr}") from e

    return wav_path


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file, resample if needed, and return a mono waveform tensor.
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def chunk_audio(
    audio: torch.Tensor,
    chunk_length_s: float,
    sr: int = 16000
) -> Generator[torch.Tensor, None, None]:
    """Yield fixed-length chunks (seconds) of an audio tensor."""
    chunk_size = int(chunk_length_s * sr)
    total_len = audio.size(1)
    for start in range(0, total_len, chunk_size):
        yield audio[:, start: start + chunk_size]


class BaseDiarizer:
    """Mixin providing speaker diarization via pyannote pipeline."""
    def __init__(
        self,
        vad_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: str = "",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device if isinstance(device, str) else device)
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.vad_pipeline = Pipeline.from_pretrained(vad_name, use_auth_token=hf_token).to(self.device)

    def _run_diarization(self, audio: torch.Tensor, sr: int) -> List[Tuple[float, float, str]]:
        params = {'waveform': audio, 'sample_rate': sr}
        kwargs = {}
        if self.min_speakers is not None:
            kwargs['min_speakers'] = self.min_speakers
        if self.max_speakers is not None:
            kwargs['max_speakers'] = self.max_speakers
        diarization = self.vad_pipeline(params, **kwargs)
        return [(seg.start, seg.end, label)
                for seg, _, label in diarization.itertracks(yield_label=True)]

class WhisperEmotionInferencer(BaseDiarizer):
    """Emotion classification of audio segments using a Whisper-based model."""
    def __init__(
        self,
        model_path: str,
        device: Union[str, torch.device] = "cpu",
        **diarizer_kwargs
    ):
        super().__init__(device=device, **diarizer_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else device)
        self.model = AutoModelForAudioClassification.from_pretrained(model_path).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(
            model_path, return_token_type_ids=False
        )
        self.id2label = ['neutral', 'angry', 'positive', 'sad', 'other']
        self.model.eval()

    def predict_emotion(self, audio: torch.Tensor) -> Tuple[str, float]:
        sr = self.processor.feature_extractor.sampling_rate
        features = self.processor.feature_extractor(
            audio.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt"
        ).input_features.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_features=features)
            probs = F.softmax(outputs.logits, dim=-1)
        idx = self.id2label[int(probs.argmax(dim=-1))]
        label = self.model.config.id2label[idx]
        return label, float(probs[0, idx].item())

    def analyze_file(self, audio_path: str, segments: Optional[List[Dict[str, float]]]) -> List[Dict]:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio = load_audio(audio_path, target_sr=self.processor.feature_extractor.sampling_rate)
        return self.inference(audio, segments)

    def inference(self, audio: torch.Tensor, segments: Optional[List[Dict[str, float]]]) -> List[Dict]:
        sr = self.processor.feature_extractor.sampling_rate
        if segments is None:
            segments = self._run_diarization(audio, sr)
        results = []
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            seg_audio = audio[:, int(start_time*sr):int(end_time*sr)]
            label, score = self.predict_emotion(seg_audio)
            segment['emotion'] = label
            segment['emotion_confidence_score'] = score
        return results


class GigaEmotionInferencer(BaseDiarizer):
    """
    Wrapper for GigaAM emotion classification model.
    Methods:
      - predict_probas: return full probability dict
      - predict_emotion: return top emotion index and its probability
    """
    def __init__(
        self,
        model_name: str = "emo",
        device: Union[str, torch.device] = "cpu",
        sample_rate: Optional[int] = None,
    ):
        self.device = torch.device(device if isinstance(device, str) else device)
        try:
            self.model = gigaam.load_model(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load GigaAM model '{model_name}': {e}")
        self.sample_rate = sample_rate
        

    def predict_probas(
        self,
        input: str
    ) -> Dict[str, float]:
        try:
            probs = self.model.get_probs(input)
        except Exception as e:
            raise RuntimeError(f"Emotion inference failed: {e}")

        return {emotion: float(prob) for emotion, prob in probs.items()}

    def predict_emotion(
        self,
        input: Union[str, torch.Tensor]
    ) -> Tuple[int, float]:
        probs = self.predict_probas(input)
        top_emotion = max(probs, key=probs.get)
        return top_emotion, probs[top_emotion]
    
    def analyze_file(self, audio_path: str, segments: Optional[List[Dict[str, float]]]) -> List[Dict]:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        waveform, _ = librosa.load(audio_path, sr=16000)
        return self.inference(waveform, segments)
    
    def inference(self, audio: torch.Tensor, segments: Optional[List[Dict[str, float]]]) -> List[Dict]:
        sr = 16000
        if segments is None:
            segments = self._run_diarization(audio, sr)
        
        results = []
        for segment in tqdm(segments):
            start, end= segment['start'], segment['end']
            s_sample = int(start * sr)
            e_sample = int(end * sr)
            seg_audio = audio[s_sample:e_sample]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                wav_tmp_path = tmp.name
                sf.write(wav_tmp_path, seg_audio, samplerate=sr)
                emo, score = self.predict_emotion(wav_tmp_path)
            segment['emotion'] = emo
            segment['emotion_confidence_score'] = score
            results.append(segment)

        return results
    
class GigaamEmotionInferencer(BaseDiarizer):
    """Emotion classification of audio segments using Gigaam RNN-T-based model."""
    def __init__(
        self,
        model_path: str,
        label_mapping: Dict[str, int],
        device: Union[str, torch.device] = "cpu",
        **diarizer_kwargs
    ):
        super().__init__(device=device, **diarizer_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else device)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = ModelForEmotionClassification.from_pretrained(
            model_path, config=config, model_name=model_path
        ).to(self.device)
        self.label_mapping = label_mapping
        self.id2label = {v: k for k, v in label_mapping.items()}
        self.model.eval()

    def predict_emotion(self, audio: torch.Tensor, sampling_rate: int) -> Tuple[str, float]:
        inputs = self.processor(
            audio.squeeze().numpy(),
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        input_lengths = inputs["input_lengths"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_features=input_features, input_lengths=input_lengths)
            probs = F.softmax(outputs.logits, dim=-1)
            idx = int(torch.argmax(probs, dim=-1))
            label = self.id2label[idx]
            confidence = float(probs[0, idx].item())
        return label, confidence

    def analyze_file(self, audio_path: str, segments: Optional[List[Dict[str, float]]]) -> List[Dict]:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio = load_audio(audio_path, target_sr=self.processor.sampling_rate)
        return self.inference(audio, segments)

    def inference(self, audio: torch.Tensor, segments: Optional[List[Dict[str, float]]]) -> List[Dict]:
        sr = self.processor.sampling_rate
        if segments is None:
            segments = self._run_diarization(audio, sr)

        results = []
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            seg_audio = audio[:, int(start_time * sr):int(end_time * sr)]
            label, score = self.predict_emotion(seg_audio, sampling_rate=sr)
            segment['emotion'] = label
            segment['emotion_confidence_score'] = score
            results.append(segment)
        return results
