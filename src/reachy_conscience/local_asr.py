"""Optional offline faster-whisper adapter for complete owned PCM segments."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .reachy_audio import SAMPLE_RATE, AudioChunk


class FasterWhisperTranscriber:
    """Transcribe a complete bounded segment with a caller-supplied local model.

    Model loading is explicit and local-only. This adapter neither downloads
    weights nor sends audio to a service. Model assets are not redistributed.
    """

    def __init__(self, model: Any, *, language: str = "en") -> None:
        if not language or len(language) > 16:
            raise ValueError("invalid language")
        self.model = model
        self.language = language

    @classmethod
    def from_local_model(cls, model_path: str | Path, *, language: str = "en") -> FasterWhisperTranscriber:
        path = Path(model_path)
        if not path.is_dir():
            raise ValueError("local model directory does not exist")
        # faster-whisper may otherwise fetch a tokenizer by model name. Require
        # a self-contained converted model directory instead of implicit I/O.
        if any(not (path / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")):
            raise ValueError("local model must contain model.bin, config.json, and tokenizer.json")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("install reachy-conscience[asr] to use local ASR") from exc
        return cls(
            WhisperModel(str(path), device="cpu", compute_type="int8", local_files_only=True),
            language=language,
        )

    def _transcribe(self, pcm: AudioChunk) -> str:
        import numpy as np

        if pcm.sample_rate != SAMPLE_RATE or pcm.channels not in (1, 2):
            raise ValueError("unsupported PCM format")
        if (
            not pcm.data
            or len(pcm.data) % (4 * pcm.channels)
            or len(pcm.data) > SAMPLE_RATE * pcm.channels * 4 * 12
        ):
            raise ValueError("ASR segment outside PCM bounds")
        values = np.frombuffer(pcm.data, dtype="<f4")
        if not np.isfinite(values).all() or (np.abs(values) > 1).any():
            raise ValueError("invalid PCM samples")
        mono = values if pcm.channels == 1 else values.reshape(-1, 2).mean(axis=1)
        segments, _info = self.model.transcribe(
            mono.astype(np.float32, copy=False),
            language=self.language,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if len(text) > 2_000:
            raise ValueError("ASR transcript exceeds input limit")
        return text

    async def transcribe(self, pcm: AudioChunk) -> str:
        return await asyncio.to_thread(self._transcribe, pcm)
