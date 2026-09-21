"""Bounded Reachy Mini 1.10 media adapter, with no model or planner access.

The adapter deliberately does not implement the pipeline's EmergencyStop port:
an emergency stop must also cancel motion and any other owned output sinks.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Protocol

SAMPLE_RATE = 16_000
MAX_OUTPUT_SECONDS = 30
MAX_INPUT_SECONDS = 1


class ReachyMediaPort(Protocol):
    def get_output_audio_samplerate(self) -> int: ...
    def get_output_channels(self) -> int: ...
    def start_playing(self) -> None: ...
    def push_audio_sample(self, samples: Any) -> None: ...
    def stop_playing(self) -> None: ...
    def get_input_audio_samplerate(self) -> int: ...
    def get_input_channels(self) -> int: ...
    def start_recording(self) -> None: ...
    def get_audio_sample(self) -> Any | None: ...
    def stop_recording(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Immutable interleaved little-endian float32 microphone data."""

    data: bytes
    sample_rate: int
    channels: int


class ReachyMediaAudio:
    """Owned microphone/speaker boundary for Reachy Mini SDK 1.10.0.

    Speaker input is complete, guarded PCM in interleaved float32 little-endian
    at 16 kHz. ``push_audio_sample`` is non-blocking; ``halt_audio`` disables
    subsequent pushes and stops the SDK playback pipeline. This has only been
    source-inspected and fake-port tested, not measured on a physical robot.
    """

    def __init__(self, media: ReachyMediaPort, *, output_channels: int = 1) -> None:
        if output_channels not in (1, 2):
            raise ValueError("output channels must be mono or stereo")
        self.media = media
        self.output_channels = output_channels
        self._lock = threading.Lock()
        self._playing = False
        self._recording = False

    def _available(self) -> None:
        # MediaManager methods can silently return when its backend is absent.
        if getattr(self.media, "audio", object()) is None:
            raise RuntimeError("Reachy audio backend unavailable")

    def _arm_playback(self) -> None:
        with self._lock:
            self._available()
            if self.media.get_output_audio_samplerate() != SAMPLE_RATE:
                raise RuntimeError("unsupported Reachy output sample rate")
            if self.media.get_output_channels() not in (1, 2):
                raise RuntimeError("unsupported Reachy output channels")
            if not self._playing:
                self.media.start_playing()
                self._playing = True

    async def arm_playback(self) -> None:
        """Explicitly allow guarded output; call again after an audio halt."""
        await asyncio.to_thread(self._arm_playback)

    def _enqueue(self, audio: bytes) -> None:
        import numpy as np

        if not isinstance(audio, bytes) or not audio or len(audio) % (4 * self.output_channels):
            raise ValueError("audio must contain complete float32 frames")
        if len(audio) > SAMPLE_RATE * self.output_channels * 4 * MAX_OUTPUT_SECONDS:
            raise ValueError("audio exceeds 30-second output cap")
        samples = np.frombuffer(audio, dtype="<f4")
        if not np.isfinite(samples).all() or (np.abs(samples) > 1).any():
            raise ValueError("audio samples must be finite and within [-1,1]")
        if self.output_channels == 2:
            samples = samples.reshape(-1, 2)
        with self._lock:
            self._available()
            if not self._playing:
                raise RuntimeError("Reachy playback is not armed")
            self.media.push_audio_sample(samples.copy())

    async def enqueue(self, audio: bytes) -> None:
        await asyncio.to_thread(self._enqueue, audio)

    def _halt_audio(self) -> None:
        with self._lock:
            self._playing = False
            # SDK 1.10's local backend shares a GStreamer pipeline for input
            # and output; stopping playback can invalidate capture as well.
            self._recording = False
            self.media.stop_playing()

    async def halt_audio(self) -> None:
        """Stop the SDK playback pipeline and require explicit re-arming."""
        await asyncio.to_thread(self._halt_audio)

    def _start_capture(self) -> None:
        with self._lock:
            self._available()
            if self.media.get_input_audio_samplerate() != SAMPLE_RATE:
                raise RuntimeError("unsupported Reachy input sample rate")
            if self.media.get_input_channels() not in (1, 2):
                raise RuntimeError("unsupported Reachy input channels")
            if not self._recording:
                self.media.start_recording()
                self._recording = True

    async def start_capture(self) -> None:
        await asyncio.to_thread(self._start_capture)

    def _read_capture(self) -> AudioChunk | None:
        import numpy as np

        with self._lock:
            self._available()
            if not self._recording:
                raise RuntimeError("Reachy recording is not started")
            raw = self.media.get_audio_sample()
            channels = self.media.get_input_channels()
            samples = None if raw is None else np.array(raw, copy=True)
        if samples is None:
            return None
        if samples.dtype != np.float32 or samples.ndim != 2 or samples.shape[1] != channels:
            raise ValueError("unexpected Reachy microphone sample shape or dtype")
        if samples.shape[0] == 0 or samples.shape[0] > SAMPLE_RATE * MAX_INPUT_SECONDS:
            raise ValueError("Reachy microphone chunk outside duration bounds")
        if not np.isfinite(samples).all() or (np.abs(samples) > 1).any():
            raise ValueError("microphone samples must be finite and within [-1,1]")
        return AudioChunk(samples.astype("<f4", copy=False).tobytes(), SAMPLE_RATE, channels)

    async def read_capture(self) -> AudioChunk | None:
        """Poll one chunk. The ASR/VAD owner decides when a segment is final."""
        return await asyncio.to_thread(self._read_capture)

    def _stop_capture(self) -> None:
        with self._lock:
            self._recording = False
            self._playing = False
            self.media.stop_recording()

    async def stop_capture(self) -> None:
        await asyncio.to_thread(self._stop_capture)
