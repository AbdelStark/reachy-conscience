"""Owned, non-streaming conversation path with guards before every output sink.

The planner is a proposal-only port: it must not own TTS, tool, or robot SDK handles.
No upstream Conversation App output queue is used here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .guard import Action, Verdict, is_hard_stop
from .ledger import Ledger


@dataclass(frozen=True, slots=True)
class Speech:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    summary: str


@dataclass(frozen=True, slots=True)
class Motion:
    motion_class: str
    target: Mapping[str, Any]


Proposal = Speech | ToolCall | Motion
Guard = Callable[[Action], Awaitable[Verdict]]


class Planner(Protocol):
    async def plan(self, transcript: str) -> Sequence[Proposal]: ...


class Synthesizer(Protocol):
    async def synthesize(self, text: str) -> bytes: ...


class AudioOutput(Protocol):
    async def enqueue(self, audio: bytes) -> None: ...


class ToolOutput(Protocol):
    async def execute(self, name: str, arguments: Mapping[str, Any]) -> None: ...


class MotionOutput(Protocol):
    async def execute(self, motion_class: str, target: Mapping[str, Any]) -> None: ...


class EmergencyStop(Protocol):
    async def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TurnResult:
    status: str
    delivered: int = 0
    reason: str | None = None


class GuardedConversation:
    """Own the transcript -> proposal -> guarded sink sequence.

    The caller supplies a proposal-only planner and independently owned output
    adapters. A held action stops the turn; this MVP has no automatic resume.
    Tool execution is opt-in for names the owner has registered as read-only.
    Motion is disabled unless explicitly enabled with an output adapter.
    """

    def __init__(
        self,
        *,
        planner: Planner,
        guard: Guard,
        synthesizer: Synthesizer,
        audio: AudioOutput,
        emergency_stop: EmergencyStop,
        tools: ToolOutput | None = None,
        read_only_tools: frozenset[str] = frozenset(),
        motion: MotionOutput | None = None,
        enable_motion: bool = False,
        ledger: Ledger | None = None,
        guard_timeout_s: float = 2.0,
    ) -> None:
        if guard_timeout_s <= 0:
            raise ValueError("guard timeout must be positive")
        if enable_motion and motion is None:
            raise ValueError("motion enabled without an output adapter")
        self.planner = planner
        self.guard = guard
        self.synthesizer = synthesizer
        self.audio = audio
        self.emergency_stop = emergency_stop
        self.tools = tools
        self.read_only_tools = read_only_tools
        self.motion = motion
        self.enable_motion = enable_motion
        self.ledger = ledger
        self.guard_timeout_s = guard_timeout_s
        self._turn_lock = asyncio.Lock()
        self._generation = 0

    async def _judge(self, action: Action) -> Verdict:
        start = time.monotonic()
        try:
            verdict = await asyncio.wait_for(self.guard(action), self.guard_timeout_s)
            if verdict.kind not in ("approve", "hold", "block"):
                verdict = Verdict("hold", "invalid_verdict")
        except Exception:
            verdict = Verdict("hold", "judgment_unavailable")
        if self.ledger is not None:
            self.ledger.append(action, verdict, {}, (time.monotonic() - start) * 1000)
        return verdict

    async def run_turn(self, transcript: str) -> TurnResult:
        if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 2000:
            return TurnResult("rejected", reason="invalid_transcript")
        if is_hard_stop(transcript):
            # This path must not wait for a model, planner, TTS, or turn lock.
            self._generation += 1
            await self.emergency_stop.stop()
            return TurnResult("stopped")
        async with self._turn_lock:
            generation = self._generation
            inbound = Action(kind="inbound", summary="inbound utterance", untrusted_text=transcript)
            verdict = await self._judge(inbound)
            if generation != self._generation:
                return TurnResult("interrupted")
            if verdict.kind != "approve":
                return TurnResult(verdict.kind, reason=verdict.reason)
            try:
                proposals = await self.planner.plan(transcript)
            except Exception:
                return TurnResult("held", reason="planner_unavailable")
            if generation != self._generation:
                return TurnResult("interrupted")
            delivered = 0
            for proposal in proposals:
                result = await self._emit(proposal, generation)
                if result.status != "delivered":
                    return TurnResult(result.status, delivered, result.reason)
                delivered += 1
            return TurnResult("complete", delivered)

    async def _emit(self, proposal: Proposal, generation: int) -> TurnResult:
        if generation != self._generation:
            return TurnResult("interrupted")
        try:
            if isinstance(proposal, Speech):
                if not proposal.text.strip() or len(proposal.text) > 2000:
                    return TurnResult("held", reason="invalid_speech")
                action = Action(kind="utterance", summary="speech proposal", text=proposal.text)
            elif isinstance(proposal, ToolCall):
                if not proposal.name or len(proposal.name) > 80 or not proposal.summary.strip():
                    return TurnResult("held", reason="invalid_tool")
                action = Action(kind="tool_call", summary=proposal.summary[:140], tool=proposal.name)
            elif isinstance(proposal, Motion):
                action = Action(kind="motion", summary="motion proposal", motion_class=proposal.motion_class)
            else:
                return TurnResult("held", reason="unknown_proposal")
        except (TypeError, ValueError):
            return TurnResult("held", reason="invalid_proposal")
        verdict = await self._judge(action)
        if generation != self._generation:
            return TurnResult("interrupted")
        if verdict.kind != "approve":
            return TurnResult(verdict.kind, reason=verdict.reason)
        if isinstance(proposal, Speech):
            try:
                audio = await self.synthesizer.synthesize(proposal.text)
            except Exception:
                return TurnResult("held", reason="synthesis_unavailable")
            if generation != self._generation:
                return TurnResult("interrupted")
            if not audio:
                return TurnResult("held", reason="empty_audio")
            try:
                await self.audio.enqueue(audio)
            except Exception:
                return TurnResult("output_error", reason="audio_sink_failed")
        elif isinstance(proposal, ToolCall):
            if self.tools is None or proposal.name not in self.read_only_tools:
                return TurnResult("held", reason="tool_not_enabled")
            try:
                await self.tools.execute(proposal.name, proposal.arguments)
            except Exception:
                return TurnResult("output_error", reason="tool_sink_failed")
        elif isinstance(proposal, Motion):
            if not self.enable_motion or self.motion is None:
                return TurnResult("held", reason="motion_not_enabled")
            try:
                await self.motion.execute(proposal.motion_class, proposal.target)
            except Exception:
                return TurnResult("output_error", reason="motion_sink_failed")
        return TurnResult("delivered", 1)
