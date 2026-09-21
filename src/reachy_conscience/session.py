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


class OwnedVoiceSession:
    """Serialize capture and turns; allow an external stop to interrupt either.

    Capture stops before transcription returns. Playback is armed only after a
    non-stop final transcript, and GuardedConversation still owns every output
    decision. ``stop`` is terminal for this session; construct a new instance
    to resume after an operator has checked the robot.
    """

    def __init__(self, ingress: Ingress, conversation: GuardedConversation, playback: Playback) -> None:
        self.ingress = ingress
        self.conversation = conversation
        self.playback = playback
        self._turn_lock = asyncio.Lock()
        self._output_lock = asyncio.Lock()
        self._capture_task: asyncio.Task[str | None] | None = None
        self._stopped = False
        self._arming = False

    async def run_once(self, *, listen_timeout_s: float = 30.0) -> TurnResult:
        async with self._turn_lock:
            if self._stopped:
                return TurnResult("stopped")
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
            async with self._output_lock:
                if self._stopped:
                    return TurnResult("interrupted")
                try:
                    self._arming = True
                    await self.playback.arm_playback()
                except Exception:
                    return TurnResult("held", reason="playback_unavailable")
                finally:
                    self._arming = False
                if self._stopped:
                    return TurnResult("interrupted")
            return await self.conversation.run_turn(transcript)

    async def stop(self) -> TurnResult:
        """Interrupt capture and the guarded turn without waiting for either."""
        self._stopped = True
        capture = self._capture_task
        if capture is not None and not capture.done():
            capture.cancel()
        arming = self._arming
        # The first stop must not wait for a stalled playback-arm call.
        result = await self.conversation.run_turn("stop")
        if arming:
            # If arming raced with the first stop, disarm again once it ends.
            async with self._output_lock:
                await self.conversation.run_turn("stop")
        if capture is not None:
            try:
                await capture
            except asyncio.CancelledError:
                pass
            except Exception:
                pass  # Capture failure cannot authorize another action.
        return result
