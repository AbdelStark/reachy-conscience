"""Bounded, owned PCM ingress before the guarded conversation pipeline.

This is an energy-based utterance delimiter, not a speech recognizer or a
reliable voice-activity detector. It never sends audio to an output sink.
"""

from __future__ import annotations

import asyncio
import math
import struct
import time
from collections import deque
from typing import Protocol

from .reachy_audio import SAMPLE_RATE, AudioChunk

FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


class CapturePort(Protocol):
    async def start_capture(self) -> None: ...
    async def read_capture(self) -> AudioChunk | None: ...
    async def stop_capture(self) -> None: ...


class Transcriber(Protocol):
    async def transcribe(self, pcm: AudioChunk) -> str: ...


class EnergySegmenter:
    """Delimit mono/stereo 16 kHz PCM into at most 12-second utterances.

    Every sample is validated before buffering. At most 200 ms of pre-roll is
    retained, and long speech is cut at the configured duration cap. The
    thresholds are application heuristics; they must be tuned on real devices.
    """

    def __init__(
        self,
        *,
        channels: int,
        threshold_rms: float = 0.015,
        min_speech_ms: int = 200,
        end_silence_ms: int = 600,
        max_segment_s: int = 12,
    ) -> None:
        if channels not in (1, 2):
            raise ValueError("channels must be mono or stereo")
        if not 0 < threshold_rms <= 1:
            raise ValueError("threshold_rms must be within (0,1]")
        if not 20 <= min_speech_ms <= 2_000 or min_speech_ms % FRAME_MS:
            raise ValueError("min_speech_ms must be a 20 ms multiple within 20..2000")
        if not 20 <= end_silence_ms <= 2_000 or end_silence_ms % FRAME_MS:
            raise ValueError("end_silence_ms must be a 20 ms multiple within 20..2000")
        if not 1 <= max_segment_s <= 12:
            raise ValueError("max_segment_s must be within 1..12")
        self.channels = channels
        self.threshold_rms = threshold_rms
        self.min_speech_frames = min_speech_ms // FRAME_MS
        self.end_silence_frames = end_silence_ms // FRAME_MS
        self.max_frames = max_segment_s * 1_000 // FRAME_MS
        self._frame_bytes = FRAME_SAMPLES * channels * 4
        self.reset()

    def reset(self) -> None:
        self._pending = bytearray()
        self._preroll: deque[bytes] = deque(maxlen=10)
        self._frames: list[bytes] = []
        self._speech_frames = 0
        self._silence_frames = 0

    def _finish(self) -> bytes | None:
        result = b"".join(self._frames) if self._speech_frames >= self.min_speech_frames else None
        self._frames = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._preroll.clear()
        return result

    def feed(self, chunk: AudioChunk) -> list[AudioChunk]:
        if chunk.sample_rate != SAMPLE_RATE or chunk.channels != self.channels:
            raise ValueError("unexpected microphone format")
        if not chunk.data or len(chunk.data) % (4 * self.channels):
            raise ValueError("incomplete microphone frames")
        if len(chunk.data) > SAMPLE_RATE * self.channels * 4:
            raise ValueError("microphone chunk exceeds one second")
        # Validate the complete chunk before it can enter any buffer.
        for (sample,) in struct.iter_unpack("<f", chunk.data):
            if not math.isfinite(sample) or abs(sample) > 1:
                raise ValueError("invalid microphone sample")
        self._pending.extend(chunk.data)
        complete: list[AudioChunk] = []
        while len(self._pending) >= self._frame_bytes:
            frame = bytes(self._pending[: self._frame_bytes])
            del self._pending[: self._frame_bytes]
            values = (sample for (sample,) in struct.iter_unpack("<f", frame))
            rms = math.sqrt(sum(sample * sample for sample in values) / (FRAME_SAMPLES * self.channels))
            voiced = rms >= self.threshold_rms
            if not self._frames:
                if not voiced:
                    self._preroll.append(frame)
                    continue
                self._frames = [*self._preroll, frame]
                self._preroll.clear()
                self._speech_frames = 1
            else:
                self._frames.append(frame)
                if voiced:
                    self._speech_frames += 1
                    self._silence_frames = 0
                else:
                    self._silence_frames += 1
            if self._silence_frames >= self.end_silence_frames or len(self._frames) >= self.max_frames:
                data = self._finish()
                if data is not None:
                    complete.append(AudioChunk(data, SAMPLE_RATE, self.channels))
        return complete


class OwnedAudioIngress:
    """Capture one final transcript without giving ASR an output handle.

    Capture always stops before transcription or the caller's guarded turn.
    Empty or timed-out input returns ``None``; a malformed sample or ASR error
    raises to the caller and never produces a transcript.
    """

    def __init__(
        self,
        capture: CapturePort,
        transcriber: Transcriber,
        *,
        channels: int,
        poll_interval_s: float = 0.01,
        read_timeout_s: float = 1.0,
        asr_timeout_s: float = 15.0,
    ) -> None:
        if not 0 < poll_interval_s <= 0.1 or not 0 < read_timeout_s <= 5 or not 0 < asr_timeout_s <= 120:
            raise ValueError("invalid ingress timing")
        self.capture = capture
        self.transcriber = transcriber
        self.segmenter = EnergySegmenter(channels=channels)
        self.poll_interval_s = poll_interval_s
        self.read_timeout_s = read_timeout_s
        self.asr_timeout_s = asr_timeout_s
        self._lock = asyncio.Lock()

    async def listen_once(self, *, timeout_s: float = 30.0) -> str | None:
        if not 1 <= timeout_s <= 120:
            raise ValueError("timeout_s must be within 1..120")
        async with self._lock:
            self.segmenter.reset()
            segment: AudioChunk | None = None
            deadline = time.monotonic() + timeout_s
            try:
                await self.capture.start_capture()
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    chunk = await asyncio.wait_for(
                        self.capture.read_capture(), min(self.read_timeout_s, max(remaining, 0.001))
                    )
                    if chunk is None:
                        await asyncio.sleep(self.poll_interval_s)
                        continue
                    segments = self.segmenter.feed(chunk)
                    if segments:
                        segment = segments[0]
                        break
            finally:
                await self.capture.stop_capture()
                self.segmenter.reset()
            if segment is None:
                return None
            transcript = await asyncio.wait_for(self.transcriber.transcribe(segment), self.asr_timeout_s)
            if not isinstance(transcript, str) or len(transcript) > 2_000:
                raise ValueError("invalid ASR transcript")
            return transcript.strip() or None
