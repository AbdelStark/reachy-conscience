"""Opt-in offline TTS fallback: complete text to bounded Reachy PCM bytes.

eSpeak NG writes WAV to stdout; FFmpeg converts it to 16 kHz float32 PCM.
Neither program is allowed to play audio. GuardedConversation calls this only
after an utterance has been approved, before its owned audio enqueue.
"""

from __future__ import annotations

import asyncio
import math
import re
import struct

from .reachy_audio import MAX_OUTPUT_SECONDS, SAMPLE_RATE

_VOICE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}(\+[A-Za-z0-9_-]{1,10})?$")
_MAX_WAV_BYTES = 8_000_000


class EspeakFfmpegSynthesizer:
    """Offline, subprocess-isolated speech synthesis with no playback side effect.

    This is a reference voice, not a quality or latency claim for a robot.
    eSpeak NG and FFmpeg must be installed by the host; neither is bundled.
    """

    def __init__(
        self,
        *,
        voice: str = "en-us",
        speed_wpm: int = 175,
        channels: int = 1,
        timeout_s: float = 10.0,
        espeak_binary: str = "espeak-ng",
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        if not _VOICE.fullmatch(voice):
            raise ValueError("invalid eSpeak voice name")
        if not 100 <= speed_wpm <= 250 or channels not in (1, 2) or not 0 < timeout_s <= 30:
            raise ValueError("invalid TTS settings")
        self.voice = voice
        self.speed_wpm = speed_wpm
        self.channels = channels
        self.timeout_s = timeout_s
        self.espeak_binary = espeak_binary
        self.ffmpeg_binary = ffmpeg_binary

    async def _call(self, *args: str, data: bytes, max_bytes: int) -> bytes:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, _stderr = await asyncio.wait_for(process.communicate(data), self.timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.wait()
            raise
        if process.returncode != 0 or not output or len(output) > max_bytes:
            raise RuntimeError("offline TTS process failed or exceeded audio cap")
        return output

    async def synthesize(self, text: str) -> bytes:
        if not isinstance(text, str) or not text.strip() or len(text) > 240 or len(text.split()) > 55:
            raise ValueError("offline TTS text outside 1..240 characters / 55 words")
        wav = await self._call(
            self.espeak_binary,
            "--stdout",
            "--stdin",
            "-v",
            self.voice,
            "-s",
            str(self.speed_wpm),
            data=text.encode("utf-8"),
            max_bytes=_MAX_WAV_BYTES,
        )
        pcm = await self._call(
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "pipe",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(self.channels),
            "-f",
            "f32le",
            "pipe:1",
            data=wav,
            max_bytes=SAMPLE_RATE * self.channels * 4 * MAX_OUTPUT_SECONDS,
        )
        if len(pcm) % (4 * self.channels):
            raise ValueError("TTS returned incomplete PCM frames")
        for (sample,) in struct.iter_unpack("<f", pcm):
            if not math.isfinite(sample) or abs(sample) > 1:
                raise ValueError("TTS returned invalid PCM samples")
        return pcm
