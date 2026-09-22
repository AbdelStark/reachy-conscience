"""One owned voice turn, from microphone capture to guarded output.

This host never opens the Conversation App's output queue. A separate physical
stop remains necessary: a spoken stop cannot be heard while playback is active
in this half-duplex reference path.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from .guard import is_hard_stop
from .pipeline import GuardedConversation, TurnResult


class Ingress(Protocol):
    async def listen_once(self, *, timeout_s: float) -> str | None: ...


class Playback(Protocol):
    async def arm_playback(self) -> None: ...
    async def enqueue(self, audio: bytes) -> None: ...
    async def halt_audio(self) -> None: ...


class OwnedPlaybackGate:
    """Arm the owned speaker only when guarded speech reaches its audio sink.

    A terminal stop revokes future enqueues synchronously. Stop requests the
    backend halt without waiting for a stalled arm, then waits for any arm or
    enqueue already in flight and halts again if it raced with the first halt.
    Queue clearance remains an SDK request, not a physical-silence receipt.
    """

    def __init__(self, backend: Playback) -> None:
        self.backend = backend
        self._output_lock = asyncio.Lock()
        self._armed = False
        self._needs_flush = False
        self._revoked = False

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def needs_flush(self) -> bool:
        """A queue-clearance request must succeed before another capture."""
        return self._needs_flush

    def revoke(self) -> None:
        self._revoked = True

    async def enqueue(self, audio: bytes) -> None:
        async with self._output_lock:
            if self._revoked:
                raise RuntimeError("owned playback was stopped")
            self._needs_flush = True
            try:
                if not self._armed:
                    await self.backend.arm_playback()
                    self._armed = True
                if self._revoked:
                    raise RuntimeError("owned playback was stopped")
                await self.backend.enqueue(audio)
                if self._revoked:
                    raise RuntimeError("owned playback was stopped")
            except BaseException:
                # Arm or push may have partially changed the SDK queue.
                self._armed = False
                self._needs_flush = True
                try:
                    await self.backend.halt_audio()
                    self._needs_flush = False
                except Exception:
                    pass  # Keep the original failure; retry flush before capture.
                raise

    async def halt_audio(self) -> None:
        self._armed = False
        self._needs_flush = True
        try:
            await self.backend.halt_audio()
            self._needs_flush = False
        finally:
            async with self._output_lock:
                if self._armed or self._needs_flush:
                    self._needs_flush = True
                    try:
                        await self.backend.halt_audio()
                        self._needs_flush = False
                    finally:
                        self._armed = False


class OwnedVoiceSession:
    """Serialize capture and turns; allow an external stop to interrupt either.

    Capture stops before transcription returns. Playback is armed only when
    guarded speech reaches the owned audio sink. A later run clears the
    previous owned audio queue before capture;
    the caller must pace turns because queue clearance is not an acknowledged
    playback-complete signal. ``stop`` is terminal for this session; construct
    a new instance to resume after an operator has checked the robot.
    """

    def __init__(
        self, ingress: Ingress, conversation: GuardedConversation, playback: OwnedPlaybackGate
    ) -> None:
        if conversation.audio is not playback:
            raise ValueError("conversation must use the owned playback gate")
        if conversation.quiet_output is not None and conversation.quiet_output is not playback:
            raise ValueError("quiet output must use the owned playback gate")
        self.ingress = ingress
        self.conversation = conversation
        self.playback = playback
        self._turn_lock = asyncio.Lock()
        self._capture_task: asyncio.Task[str | None] | None = None
        self._stopped = False

    async def run_once(self, *, listen_timeout_s: float = 30.0) -> TurnResult:
        async with self._turn_lock:
            if self._stopped:
                return TurnResult("stopped")
            if self.playback.armed or self.playback.needs_flush:
                # A completed enqueue is not a playback-complete receipt. Clear
                # the previous owned queue before reopening the microphone.
                try:
                    await self.playback.halt_audio()
                except Exception:
                    return TurnResult("held", reason="playback_unavailable")
                if self._stopped:
                    return TurnResult("interrupted")
            capture = asyncio.create_task(self.ingress.listen_once(timeout_s=listen_timeout_s))
            self._capture_task = capture
            try:
                transcript = await capture
            except asyncio.CancelledError:
                if self._stopped:
                    return TurnResult("interrupted")
                raise
            except Exception:
                return TurnResult("held", reason="ingress_unavailable")
            finally:
                self._capture_task = None
            if self._stopped:
                return TurnResult("interrupted")
            if transcript is None:
                return TurnResult("no_input")
            if is_hard_stop(transcript):
                self._stopped = True
                return await self.conversation.run_turn(transcript)
            result = await self.conversation.run_turn(transcript)
            if result.status == "stopped":
                self._stopped = True
            return result

    async def stop(self) -> TurnResult:
        """Request output stop promptly, then join any cancelled capture cleanup."""
        self._stopped = True
        capture = self._capture_task
        if capture is not None and not capture.done():
            capture.cancel()
        try:
            return await self.conversation.run_turn("stop")
        finally:
            if capture is not None:
                try:
                    await capture
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass  # Capture failure cannot authorize another action.
