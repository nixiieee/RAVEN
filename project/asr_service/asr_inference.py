import os
import re
import logging
from typing import Union, List, Tuple, Dict, Generator, Optional

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModel,
    AutoProcessor,
)
from pyannote.audio import Pipeline
import tempfile
import os
import ffmpeg

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

# Pre-compile stop-phrase regex patterns
def _build_stop_patterns(stop_list: List[str]) -> List[re.Pattern]:
    return [re.compile(re.escape(p), flags=re.IGNORECASE) for p in stop_list]

stop_phrases = [
    'Редактор субтитров А.Семкин Корректор А.Егорова',
    'Субтитры добавил DimaTorzok',
    'Аллах Акбар, Иблис Аллах',
    'Редактор субтитров А.Синецкая Корректор А.Егорова',
    'Субтитры делал DimaTorzok',
    'ПОКА!',
    'Продолжение следует...',
    'Спасибо за просмотр!',
    'Удачи!'
]
_stop_patterns = _build_stop_patterns(stop_phrases)

def remove_stop_phrases(text: str) -> str:
    """Remove configured stop phrases from a given text."""
    for pat in _stop_patterns:
        text = pat.sub('', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


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


class WhisperASRInferencer(BaseDiarizer):
    """ASR inference using OpenAI Whisper with speaker diarization."""
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: Union[str, torch.device] = "cpu",
        language: str = "Russian",
        task: str = "transcribe",
        chunk_length_s: float = 30.0,
        **diarizer_kwargs
    ):
        super().__init__(device=device, **diarizer_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else device)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, attn_implementation="sdpa"
        ).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(
            model_name, language=language, task=task
        )
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )
        self.chunk_length_s = chunk_length_s
        self.model.eval()

    def _decode_segment(self, segment: torch.Tensor) -> str:
        sr = self.processor.feature_extractor.sampling_rate
        features = self.processor.feature_extractor(
            segment.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt"
        ).input_features.half().to(self.device)
        with torch.amp.autocast(device_type=('cuda' if self.device.type=='cuda' else 'cpu')):
            gen_ids = self.model.generate(
                features,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                forced_decoder_ids=self.forced_decoder_ids,
            )
        raw = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return remove_stop_phrases(raw)

    def transcribe_file(self, path: str, min_speakers: int, max_speakers: int) -> List[Dict]:
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio = load_audio(path, target_sr=self.processor.feature_extractor.sampling_rate)
        return self.transcribe(audio)

    def transcribe(self, audio: torch.Tensor) -> List[Dict]:
        sr = self.processor.feature_extractor.sampling_rate
        segments = self._run_diarization(audio, sr)
        results, prev = [], {'start': None, 'end': None, 'speaker': None, 'text': ""}
        for start, end, speaker in tqdm(segments, desc="Diarization segments"):
            seg_audio = audio[:, int(start*sr):int(end*sr)]
            text = self._decode_segment(seg_audio)
            if not text: continue
            can_merge = (prev['speaker']==speaker and prev['start'] and end-prev['start']<=self.chunk_length_s) or \
                        (prev['end'] and start-prev['end']<0.001)
            if prev['start'] is None:
                prev.update(start=start, end=end, speaker=speaker, text=text)
            elif can_merge:
                prev['text'] += ' ' + text
                prev['end'] = max(prev['end'], end)
            else:
                results.append(prev.copy())
                prev = {'start': start, 'end': end, 'speaker': speaker, 'text': text}
        if prev['start'] is not None and prev['text']:
            results.append(prev)
        if self.device.type=='cuda': torch.cuda.empty_cache()
        return results

class GigaamCtcInferencer(BaseDiarizer):
    """CTC-based ASR inference with GigaAM model and speaker diarization."""
    def __init__(
        self,
        model_name: str = "waveletdeboshir/gigaam-ctc",
        device: Union[str, torch.device] = "cpu",
        chunk_length_s: float = 30.0,
        **diarizer_kwargs
    ):
        super().__init__(**diarizer_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model_name = model_name
        self.chunk_length_s = chunk_length_s
        self.model.eval()

    def _decode_segment(self, segment: torch.Tensor) -> str:
        sr = 16000
        inputs = self.processor(segment.squeeze(0), sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        if self.model_name == 'waveletdeboshir/gigaam-ctc-with-lm':
            return self.processor.batch_decode(logits=logits.cpu().numpy(),
                                            beam_width=64,
                                            alpha=0.5, 
                                            beta=0.5,
                                            ).text[0]
        greedy_ids = logits.argmax(dim=-1)
        return self.processor.batch_decode(greedy_ids)[0].strip()

    def transcribe_file(self, path: str, min_speakers: int, max_speakers: int) -> List[Dict]:
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio = load_audio(path, target_sr=16000)
        return self.transcribe(audio)

    def transcribe(self, audio: torch.Tensor) -> List[Dict]:
        sr = 16000
        segments = self._run_diarization(audio, sr)
        results, prev = [], {'start': None, 'end': None, 'speaker': None, 'text': ""}
        for start, end, speaker in tqdm(segments, desc="Diarization segments"):
            seg_audio = audio[:, int(start*sr):int(end*sr)]
            text = self._decode_segment(seg_audio)
            if not text: continue
            can_merge = (prev['speaker']==speaker and prev['start'] and end-prev['start']<=self.chunk_length_s) or \
                        (prev['end'] and start-prev['end']<0.001)
            if prev['start'] is None:
                prev.update(start=start, end=end, speaker=speaker, text=text)
            elif can_merge:
                prev['text'] += ' ' + text
                prev['end'] = max(prev['end'], end)
            else:
                results.append(prev.copy())
                prev = {'start': start, 'end': end, 'speaker': speaker, 'text': text}
        if prev['start'] is not None and prev['text']:
            results.append(prev)
        if self.device.type=='cuda': torch.cuda.empty_cache()
        return results
