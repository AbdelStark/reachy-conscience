"""Opt-in, one-frame camera-sign OCR with no planner or output handle.

The pinned Reachy SDK supplies JPEG bytes. Pillow checks format and dimensions;
an installed local Tesseract binary extracts bounded text entirely on the host.
Neither image nor OCR text is saved by this adapter.
"""

from __future__ import annotations

import asyncio
import io
from contextlib import suppress
from typing import Any, Protocol

MAX_JPEG_BYTES = 2_000_000
MAX_PIXELS = 12_000_000
MAX_OCR_BYTES = 8_192
MAX_TEXT_CHARS = 2_000


class CameraFramePort(Protocol):
    async def capture_jpeg(self) -> bytes | None: ...


class OcrPort(Protocol):
    async def extract_text(self, jpeg: bytes) -> str | None: ...


class ReachyCameraFrame:
    """One current JPEG from SDK 1.10 MediaManager, with no camera lifecycle control."""

    def __init__(self, media: Any) -> None:
        if not callable(getattr(media, "get_frame_jpeg", None)):
            raise ValueError("Reachy media has no JPEG camera method")
        self.media = media

    async def capture_jpeg(self) -> bytes | None:
        frame = await asyncio.to_thread(self.media.get_frame_jpeg)
        if frame is None:
            return None
        if not isinstance(frame, bytes) or not 0 < len(frame) <= MAX_JPEG_BYTES:
            raise ValueError("camera JPEG outside byte limit")
        return frame


def _validate_jpeg(jpeg: bytes) -> None:
    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= MAX_JPEG_BYTES:
        raise ValueError("camera JPEG outside byte limit")
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError("install the ocr extra for JPEG validation") from exc
    try:
        with Image.open(io.BytesIO(jpeg)) as frame:
            width, height = frame.size
            if frame.format != "JPEG" or width < 1 or height < 1 or width * height > MAX_PIXELS:
                raise ValueError("camera JPEG has an unsupported format or size")
            frame.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("invalid camera JPEG") from exc


class LocalTesseractOcr:
    """Run offline English OCR on a complete validated frame, never on a stream."""

    def __init__(self, *, binary: str = "tesseract", timeout_s: float = 5.0) -> None:
        if not binary or "/" in binary or "\\" in binary or any(ord(c) < 33 for c in binary):
            raise ValueError("OCR binary must be a simple command name")
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not 0 < timeout_s <= 10:
            raise ValueError("OCR timeout must be within (0,10] seconds")
        self.binary = binary
        self.timeout_s = timeout_s

    async def extract_text(self, jpeg: bytes) -> str | None:
        _validate_jpeg(jpeg)
        process = await asyncio.create_subprocess_exec(
            self.binary,
            "stdin",
            "stdout",
            "-l",
            "eng",
            "--psm",
            "6",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def collect() -> bytes:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(jpeg)
            await process.stdin.drain()
            process.stdin.close()
            chunks: list[bytes] = []
            size = 0
            while chunk := await process.stdout.read(1024):
                size += len(chunk)
                if size > MAX_OCR_BYTES:
                    raise ValueError("OCR text exceeds byte limit")
                chunks.append(chunk)
            if await process.wait() != 0:
                raise RuntimeError("local OCR failed")
            return b"".join(chunks)

        try:
            raw = await asyncio.wait_for(collect(), self.timeout_s)
        except BaseException:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await process.wait()
            raise
        try:
            text = raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("OCR returned invalid UTF-8") from exc
        if len(text) > MAX_TEXT_CHARS or any(ord(c) < 32 and c not in "\r\n\t" for c in text):
            raise ValueError("OCR text outside transcript bounds")
        return text or None


class CameraSignIngress:
    """Capture a single still, then OCR it; no audio, planner, or robot output."""

    def __init__(self, camera: CameraFramePort, ocr: OcrPort, *, capture_timeout_s: float = 2.0) -> None:
        if (
            not isinstance(capture_timeout_s, (int, float))
            or isinstance(capture_timeout_s, bool)
            or not 0 < capture_timeout_s <= 10
        ):
            raise ValueError("invalid camera capture timeout")
        self.camera = camera
        self.ocr = ocr
        self.capture_timeout_s = capture_timeout_s

    async def read_once(self) -> str | None:
        jpeg = await asyncio.wait_for(self.camera.capture_jpeg(), self.capture_timeout_s)
        if jpeg is None:
            return None
        if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= MAX_JPEG_BYTES:
            raise ValueError("camera JPEG outside byte limit")
        text = await self.ocr.extract_text(jpeg)
        if text is None:
            return None
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > MAX_TEXT_CHARS
            or any(ord(c) < 32 and c not in "\r\n\t" for c in text)
        ):
            raise ValueError("invalid OCR text")
        return text.strip()
