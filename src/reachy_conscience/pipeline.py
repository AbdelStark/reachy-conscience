"""Owned, non-streaming conversation path with guards before every output sink.

The planner is a proposal-only port: it must not own TTS, tool, or robot SDK handles.
No upstream Conversation App output queue is used here.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .guard import Action, GuardAssessment, InboundRoute, Verdict, is_hard_stop, motion_preflight
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
Guard = Callable[[Action], Awaitable[Verdict | GuardAssessment]]
_CONFIRMABLE_HOLDS = frozenset({"confirm_before", "model_requests_confirmation", "irreversible_tool"})
MAX_PROPOSALS = 4


def _canonical_object(value: Mapping[str, Any], max_bytes: int) -> str:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected an object with string keys")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ValueError("object exceeds byte limit")
    return encoded


def _prepare_proposal(proposal: Proposal) -> Proposal | str:
    """Validate and detach one proposal from planner-owned mutable mappings."""
    if isinstance(proposal, Speech):
        if not isinstance(proposal.text, str) or not proposal.text.strip() or len(proposal.text) > 2000:
            return "invalid_speech"
        return Speech(proposal.text)
    elif isinstance(proposal, ToolCall):
        if (
            not isinstance(proposal.name, str)
            or not proposal.name
            or len(proposal.name) > 80
            or not isinstance(proposal.summary, str)
            or not proposal.summary.strip()
        ):
            return "invalid_tool"
        try:
            arguments_json = _canonical_object(proposal.arguments, 4096)
        except (TypeError, ValueError, OverflowError, RecursionError):
            return "invalid_tool_arguments"
        return ToolCall(proposal.name, json.loads(arguments_json), proposal.summary)
    elif isinstance(proposal, Motion):
        if (
            not isinstance(proposal.motion_class, str)
            or not proposal.motion_class.strip()
            or len(proposal.motion_class) > 80
        ):
            return "invalid_proposal"
        try:
            target_json = _canonical_object(proposal.target, 2048)
        except (TypeError, ValueError, OverflowError, RecursionError):
            return "invalid_motion_target"
        return Motion(proposal.motion_class, json.loads(target_json))
    else:
        return "unknown_proposal"


class Planner(Protocol):
    async def plan(self, transcript: str) -> Sequence[Proposal]: ...


class Synthesizer(Protocol):
    async def synthesize(self, text: str) -> bytes: ...


class AudioOutput(Protocol):
    async def enqueue(self, audio: bytes) -> None: ...


class ToolOutput(Protocol):
    async def execute(self, name: str, arguments: Mapping[str, Any]) -> None: ...


class OwnerApproval(Protocol):
    async def authorize(self, name: str, arguments_json: str) -> bool: ...


class MotionOutput(Protocol):
    async def execute(self, motion_class: str, target: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class MotionContextSnapshot:
    """Trusted, same-clock sensor buckets for one proposed motion."""

    nearest_person_distance: str
    battery: str
    motor_temperature: str
    observed_at_monotonic: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.observed_at_monotonic, bool)
            or not isinstance(self.observed_at_monotonic, (int, float))
            or not math.isfinite(self.observed_at_monotonic)
            or self.observed_at_monotonic < 0
        ):
            raise ValueError("invalid motion context timestamp")


class MotionContextProvider(Protocol):
    async def snapshot(self) -> MotionContextSnapshot: ...


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
    Tool execution is opt-in for names registered as read-only or effectful.
    Effectful calls additionally require an independent, trusted owner approval
    port for the exact arguments before dispatch.
    Motion is disabled unless explicitly enabled with output and fresh-context adapters.
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
        effectful_tools: frozenset[str] = frozenset(),
        owner_approval: OwnerApproval | None = None,
        motion: MotionOutput | None = None,
        enable_motion: bool = False,
        motion_context: MotionContextProvider | None = None,
        motion_context_timeout_s: float = 0.25,
        max_motion_context_age_s: float = 1.0,
        ledger: Ledger | None = None,
        guard_timeout_s: float = 2.0,
        planner_timeout_s: float = 35.0,
        synthesis_timeout_s: float = 30.0,
        approval_timeout_s: float = 30.0,
        require_inbound_route: bool = False,
    ) -> None:
        if guard_timeout_s <= 0:
            raise ValueError("guard timeout must be positive")
        if (
            isinstance(planner_timeout_s, bool)
            or not isinstance(planner_timeout_s, (int, float))
            or not math.isfinite(planner_timeout_s)
            or not 0 < planner_timeout_s <= 120
            or isinstance(synthesis_timeout_s, bool)
            or not isinstance(synthesis_timeout_s, (int, float))
            or not math.isfinite(synthesis_timeout_s)
            or not 0 < synthesis_timeout_s <= 120
        ):
            raise ValueError("invalid planner or synthesis timeout")
        if approval_timeout_s <= 0:
            raise ValueError("approval timeout must be positive")
        if enable_motion and motion is None:
            raise ValueError("motion enabled without an output adapter")
        if enable_motion and motion_context is None:
            raise ValueError("motion enabled without a context provider")
        if (
            isinstance(motion_context_timeout_s, bool)
            or not isinstance(motion_context_timeout_s, (int, float))
            or not math.isfinite(motion_context_timeout_s)
            or motion_context_timeout_s <= 0
            or isinstance(max_motion_context_age_s, bool)
            or not isinstance(max_motion_context_age_s, (int, float))
            or not math.isfinite(max_motion_context_age_s)
            or max_motion_context_age_s <= 0
        ):
            raise ValueError("invalid motion context timing")
        if read_only_tools & effectful_tools:
            raise ValueError("tool cannot be both read-only and effectful")
        if effectful_tools and (tools is None or owner_approval is None):
            raise ValueError("effectful tools require output and owner approval adapters")
        self.planner = planner
        self.guard = guard
        self.synthesizer = synthesizer
        self.audio = audio
        self.emergency_stop = emergency_stop
        self.tools = tools
        self.read_only_tools = frozenset(read_only_tools)
        self.effectful_tools = frozenset(effectful_tools)
        self.owner_approval = owner_approval
        self.motion = motion
        self.enable_motion = enable_motion
        self.motion_context = motion_context
        self.motion_context_timeout_s = motion_context_timeout_s
        self.max_motion_context_age_s = max_motion_context_age_s
        self.ledger = ledger
        self.guard_timeout_s = guard_timeout_s
        self.planner_timeout_s = planner_timeout_s
        self.synthesis_timeout_s = synthesis_timeout_s
        self.approval_timeout_s = approval_timeout_s
        self.require_inbound_route = require_inbound_route
        self._turn_lock = asyncio.Lock()
        self._generation = 0
        self._stopped = False

    async def _judge(self, action: Action) -> GuardAssessment:
        start = time.monotonic()
        preflight = motion_preflight(action)
        if preflight is not None:
            if self.ledger is not None:
                self.ledger.append(action, preflight, {}, (time.monotonic() - start) * 1000)
            return GuardAssessment(preflight)
        probabilities: dict[str, float] = {}
        bank: str | None = None
        model: str | None = None
        try:
            result = await asyncio.wait_for(self.guard(action), self.guard_timeout_s)
            if isinstance(result, GuardAssessment):
                assessment = result
                verdict = assessment.verdict
                probabilities = dict(assessment.probabilities)
                bank = assessment.bank
                model = assessment.model
            else:
                verdict = result
                assessment = GuardAssessment(verdict)
            if verdict.kind not in ("approve", "hold", "block"):
                verdict = Verdict("hold", "invalid_verdict")
                assessment = GuardAssessment(verdict)
        except Exception:
            verdict = Verdict("hold", "judgment_unavailable")
            assessment = GuardAssessment(verdict)
            probabilities = {}
            bank = None
            model = None
        route = (
            assessment.route
            if action.kind == "inbound" and isinstance(assessment.route, InboundRoute)
            else None
        )
        if self.ledger is not None:
            self.ledger.append(
                action,
                verdict,
                probabilities,
                (time.monotonic() - start) * 1000,
                bank=bank,
                model=model,
                route=route,
            )
        return GuardAssessment(verdict, probabilities, bank, model, route)

    async def _stop_now(self) -> TurnResult:
        """Terminal owned stop, independent of the planner or output guard."""
        self._stopped = True
        self._generation += 1
        revocation_error: Exception | None = None
        try:
            revoke_audio = getattr(self.audio, "revoke", None)
            if callable(revoke_audio):
                revoke_audio()
        except Exception as exc:
            revocation_error = exc
        try:
            cancel_pending = getattr(self.owner_approval, "cancel_all", None)
            if callable(cancel_pending):
                cancel_pending()
        except Exception:
            pass  # A failed approval cancel must not suppress the output stop.
        await self.emergency_stop.stop()
        if revocation_error is not None:
            raise RuntimeError("audio revocation failed after output stop request") from revocation_error
        return TurnResult("stopped")

    async def run_turn(self, transcript: str) -> TurnResult:
        if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 2000:
            if self._stopped:
                return TurnResult("stopped")
            return TurnResult("rejected", reason="invalid_transcript")
        # A second direct stop may retry a failed or unacknowledged halt request.
        if is_hard_stop(transcript):
            return await self._stop_now()
        if self._stopped:
            return TurnResult("stopped")
        async with self._turn_lock:
            if self._stopped:
                return TurnResult("stopped")
            generation = self._generation
            inbound = Action(kind="inbound", summary="inbound utterance", untrusted_text=transcript)
            inbound_assessment = await self._judge(inbound)
            verdict = inbound_assessment.verdict
            if generation != self._generation:
                return TurnResult("interrupted")
            if verdict.kind != "approve":
                return TurnResult(verdict.kind, reason=verdict.reason)
            if self.require_inbound_route:
                route = inbound_assessment.route
                if not isinstance(route, InboundRoute) or route.confidence < 0.7:
                    return TurnResult("held", reason="route_unavailable")
                if route.choice == "ignore":
                    if route.fast_command != "none":
                        return TurnResult("held", reason="conflicting_route")
                    return TurnResult("ignored", reason="not_directed_at_robot")
                if route.choice == "fast_path":
                    if route.fast_command == "stop" and route.command_confidence >= 0.7:
                        return await self._stop_now()
                    return TurnResult("held", reason="fast_command_not_enabled")
                if route.choice != "llm":
                    return TurnResult("held", reason="route_unavailable")
                if route.fast_command != "none":
                    return TurnResult("held", reason="conflicting_route")
            try:
                pending = self.planner.plan(transcript)
            except Exception:
                return TurnResult("held", reason="planner_unavailable")
            return await self._plan_and_emit(pending, generation, transcript)

    async def _plan_and_emit(
        self,
        pending: Awaitable[Sequence[Proposal]],
        generation: int,
        request_text: str,
        *,
        sign_mode: bool = False,
    ) -> TurnResult:
        try:
            proposals = await asyncio.wait_for(pending, self.planner_timeout_s)
        except Exception:
            return TurnResult("held", reason="planner_unavailable")
        if generation != self._generation:
            return TurnResult("interrupted")
        # The concrete local planner already caps proposals, but this
        # boundary must not trust a replacement adapter to do so. Reject
        # the whole batch before delivering any earlier valid item.
        try:
            if (
                not isinstance(proposals, Sequence)
                or isinstance(proposals, (str, bytes, bytearray))
                or len(proposals) > (1 if sign_mode else MAX_PROPOSALS)
            ):
                raise ValueError("invalid planner batch")
            proposals = tuple(proposals)
            if len(proposals) > (1 if sign_mode else MAX_PROPOSALS) or any(
                not isinstance(item, Speech if sign_mode else (Speech, ToolCall, Motion))
                for item in proposals
            ):
                raise ValueError("invalid planner item")
        except Exception:
            return TurnResult("held", reason="invalid_proposals")
        prepared_proposals: list[Proposal] = []
        for proposal in proposals:
            prepared = _prepare_proposal(proposal)
            if isinstance(prepared, str):
                return TurnResult("held", reason=prepared)
            prepared_proposals.append(prepared)
        delivered = 0
        for proposal in prepared_proposals:
            result = await self._emit(
                proposal, generation, request_text, sign_text=request_text if sign_mode else None
            )
            if result.status != "delivered":
                return TurnResult(result.status, delivered, result.reason)
            delivered += 1
        return TurnResult("complete", delivered)

    async def screen_sign(self, text: str) -> TurnResult:
        """Judge one OCR result as untrusted camera text, without planning or output.

        A sign is never a local hard-stop command. A valid approval only means
        the inbound guard did not reject the text; it does not execute it.
        """
        if self._stopped:
            return TurnResult("stopped")
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            return TurnResult("rejected", reason="invalid_sign_text")
        async with self._turn_lock:
            if self._stopped:
                return TurnResult("stopped")
            generation = self._generation
            action = Action(kind="inbound", summary="camera sign", untrusted_text=text, source="camera_sign")
            assessment = await self._judge(action)
            if generation != self._generation:
                return TurnResult("interrupted")
            return TurnResult(assessment.verdict.kind, reason=assessment.verdict.reason)

    async def respond_to_sign(self, text: str) -> TurnResult:
        """Optionally respond to one sign after inbound and output guards.

        The sign cannot invoke fast commands, tools, or motion. Only a
        proposal-only planner with a dedicated ``plan_sign`` method is used.
        """
        if self._stopped:
            return TurnResult("stopped")
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            return TurnResult("rejected", reason="invalid_sign_text")
        async with self._turn_lock:
            if self._stopped:
                return TurnResult("stopped")
            generation = self._generation
            inbound = Action(kind="inbound", summary="camera sign", untrusted_text=text, source="camera_sign")
            assessment = await self._judge(inbound)
            if generation != self._generation:
                return TurnResult("interrupted")
            if assessment.verdict.kind != "approve":
                return TurnResult(assessment.verdict.kind, reason=assessment.verdict.reason)
            plan_sign = getattr(self.planner, "plan_sign", None)
            if not callable(plan_sign):
                return TurnResult("held", reason="sign_planner_unavailable")
            try:
                pending = plan_sign(text)
            except Exception:
                return TurnResult("held", reason="planner_unavailable")
            return await self._plan_and_emit(pending, generation, text, sign_mode=True)

    async def _emit(
        self, proposal: Proposal, generation: int, transcript: str, *, sign_text: str | None = None
    ) -> TurnResult:
        if generation != self._generation:
            return TurnResult("interrupted")
        tool_arguments_json: str | None = None
        motion_target_json: str | None = None
        motion_observed_at: float | None = None
        try:
            if isinstance(proposal, Speech):
                action = Action(
                    kind="utterance",
                    summary="speech proposal",
                    text=proposal.text,
                    untrusted_text=sign_text,
                    source="camera_sign" if sign_text is not None else None,
                )
            elif isinstance(proposal, ToolCall):
                try:
                    tool_arguments_json = _canonical_object(proposal.arguments, 4096)
                except (TypeError, ValueError, OverflowError, RecursionError):
                    return TurnResult("held", reason="invalid_tool_arguments")
                action = Action(
                    kind="tool_call",
                    summary=proposal.summary[:140],
                    tool=proposal.name,
                    tool_arguments_json=tool_arguments_json,
                    user_request=transcript,
                )
            elif isinstance(proposal, Motion):
                if not self.enable_motion or self.motion is None:
                    return TurnResult("held", reason="motion_not_enabled")
                try:
                    motion_target_json = _canonical_object(proposal.target, 2048)
                except (TypeError, ValueError, OverflowError, RecursionError):
                    return TurnResult("held", reason="invalid_motion_target")
                try:
                    assert self.motion_context is not None
                    context = await asyncio.wait_for(
                        self.motion_context.snapshot(), self.motion_context_timeout_s
                    )
                    observed_age = time.monotonic() - context.observed_at_monotonic
                    if (
                        not isinstance(context, MotionContextSnapshot)
                        or not 0 <= observed_age <= self.max_motion_context_age_s
                    ):
                        raise ValueError("stale or invalid motion context")
                except Exception:
                    return TurnResult("held", reason="motion_context_unavailable")
                if generation != self._generation:
                    return TurnResult("interrupted")
                motion_observed_at = context.observed_at_monotonic
                action = Action(
                    kind="motion",
                    summary="motion proposal",
                    motion_class=proposal.motion_class,
                    motion_target_json=motion_target_json,
                    user_request=transcript,
                    nearest_person_distance=context.nearest_person_distance,
                    battery=context.battery,
                    motor_temperature=context.motor_temperature,
                )
            else:
                return TurnResult("held", reason="unknown_proposal")
        except (TypeError, ValueError, OverflowError, RecursionError):
            return TurnResult("held", reason="invalid_proposal")
        verdict = (await self._judge(action)).verdict
        if generation != self._generation:
            return TurnResult("interrupted")
        needs_owner_confirmation = (
            isinstance(proposal, ToolCall)
            and proposal.name in self.effectful_tools
            and verdict.kind == "hold"
            and verdict.reason in _CONFIRMABLE_HOLDS
        )
        if verdict.kind != "approve" and not needs_owner_confirmation:
            return TurnResult(verdict.kind, reason=verdict.reason)
        if isinstance(proposal, Speech):
            try:
                audio = await asyncio.wait_for(
                    self.synthesizer.synthesize(proposal.text), self.synthesis_timeout_s
                )
            except Exception:
                return TurnResult("held", reason="synthesis_unavailable")
            if generation != self._generation:
                return TurnResult("interrupted")
            if not audio:
                return TurnResult("held", reason="empty_audio")
            try:
                await self.audio.enqueue(audio)
            except Exception:
                if generation != self._generation:
                    return TurnResult("interrupted")
                return TurnResult("output_error", reason="audio_sink_failed")
        elif isinstance(proposal, ToolCall):
            if self.tools is None or proposal.name not in self.read_only_tools | self.effectful_tools:
                return TurnResult("held", reason="tool_not_enabled")
            assert tool_arguments_json is not None
            if proposal.name in self.effectful_tools:
                assert self.owner_approval is not None
                try:
                    authorized = await asyncio.wait_for(
                        self.owner_approval.authorize(proposal.name, tool_arguments_json),
                        self.approval_timeout_s,
                    )
                except Exception:
                    return TurnResult("held", reason="owner_approval_unavailable")
                if generation != self._generation:
                    return TurnResult("interrupted")
                if authorized is not True:
                    return TurnResult("held", reason="owner_approval_denied")
            try:
                await self.tools.execute(proposal.name, json.loads(tool_arguments_json))
            except Exception:
                return TurnResult("output_error", reason="tool_sink_failed")
        elif isinstance(proposal, Motion):
            if not self.enable_motion or self.motion is None:
                return TurnResult("held", reason="motion_not_enabled")
            assert motion_target_json is not None
            if (
                motion_observed_at is None
                or not 0 <= time.monotonic() - motion_observed_at <= self.max_motion_context_age_s
            ):
                return TurnResult("held", reason="motion_context_unavailable")
            try:
                await self.motion.execute(proposal.motion_class, json.loads(motion_target_json))
            except Exception:
                return TurnResult("output_error", reason="motion_sink_failed")
        if generation != self._generation:
            return TurnResult("interrupted")
        return TurnResult("delivered", 1)
