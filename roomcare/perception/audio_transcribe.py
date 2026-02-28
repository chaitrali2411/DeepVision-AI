"""On-device audio transcription (faster-whisper). No raw audio stored."""
from __future__ import annotations

import io
from typing import Optional

import numpy as np

from roomcare.config import SAMPLE_RATE, WHISPER_MODEL


class AudioTranscriber:
    """On-device speech-to-text; no audio is persisted."""

    def __init__(self, model_size: str = WHISPER_MODEL, device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        self._model = WhisperModel(self.model_size, device=self.device, compute_type="int8")

    def transcribe(self, audio: np.ndarray | bytes, sample_rate: int = SAMPLE_RATE) -> str:
        """Return transcribed text. audio: float32 mono [-1,1] or bytes."""
        self._ensure_loaded()
        if isinstance(audio, bytes):
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(audio))
            if sr != sample_rate and len(audio) > 0:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
            audio = audio.astype(np.float32)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / (np.iinfo(audio.dtype).max or 1)
        segments, _ = self._model.transcribe(audio, sample_rate=sample_rate, language="en")
        return " ".join(s.text.strip() for s in segments if s.text).strip()
