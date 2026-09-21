"""Pure, fail-closed policy for model judgments about proposed robot actions."""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["utterance", "tool_call", "motion", "inbound"]
VerdictKind = Literal["approve", "hold", "block"]
Judgment = float | str
_MOTION_DISTANCES = frozenset({"very near", "near", "medium", "far"})
_BATTERY_BUCKETS = frozenset({"critical", "low", "normal"})
_MOTOR_TEMPERATURES = frozenset({"cool", "warm", "hot"})


@dataclass(frozen=True, slots=True)
class Action:
    kind: Kind
    summary: str
    tool: str | None = None
    tool_arguments_json: str | None = None
    motion_class: str | None = None
    motion_target_json: str | None = None
    untrusted_text: str | None = None
    source: str | None = None
    text: str | None = None
    user_request: str | None = None
    audience: Mapping[str, bool] | None = None
    nearest_person_distance: str | None = None
    battery: str | None = None
    motor_temperature: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("utterance", "tool_call", "motion", "inbound"):
            raise ValueError("unknown action kind")
        if len(self.summary) > 140 or not self.summary.strip():
            raise ValueError("summary must be 1..140 characters")
        if self.kind == "tool_call" and not self.tool:
            raise ValueError("tool_call requires tool")
        if self.kind == "motion" and not self.motion_class:
            raise ValueError("motion requires motion_class")
        if self.source is not None and (
            self.kind not in {"inbound", "utterance"} or self.source != "camera_sign"
        ):
            raise ValueError("invalid action source")
        if self.source == "camera_sign" and (
            not isinstance(self.untrusted_text, str)
            or not self.untrusted_text.strip()
            or len(self.untrusted_text) > 2000
        ):
            raise ValueError("camera sign requires bounded untrusted text")


@dataclass(frozen=True, slots=True)
class GuardPolicy:
    rules: tuple[str, ...] = ()
    confirm_before: frozenset[str] = field(default_factory=frozenset)
    block_threshold: float = 0.7
    hold_floor: float = 0.3

    def __post_init__(self) -> None:
        if lint_rules(self.rules):
            raise ValueError("rules must each contain one clear condition and be <=140 characters")
        if not 0 <= self.hold_floor < self.block_threshold <= 1:
            raise ValueError("invalid guard thresholds")


@dataclass(frozen=True, slots=True)
class Verdict:
    kind: VerdictKind
    reason: str


@dataclass(frozen=True, slots=True)
class GuardAssessment:
    """Verdict plus inspectable numeric judgments, without action text."""

    verdict: Verdict
    probabilities: Mapping[str, float] = field(default_factory=dict)
    bank: str | None = None
    model: str | None = None
    route: InboundRoute | None = None


@dataclass(frozen=True, slots=True)
class InboundRoute:
    """Validated model routing evidence; never an output authorization by itself."""

    choice: Literal["fast_path", "llm", "ignore"]
    confidence: float
    fast_command: Literal["stop", "look_at_speaker", "quiet", "sleep", "none"]
    command_confidence: float

    def __post_init__(self) -> None:
        if self.choice not in ("fast_path", "llm", "ignore") or self.fast_command not in (
            "stop",
            "look_at_speaker",
            "quiet",
            "sleep",
            "none",
        ):
            raise ValueError("invalid inbound route choice")
        for confidence in (self.confidence, self.command_confidence):
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("invalid inbound route confidence")


def lint_rules(rules: tuple[str, ...]) -> list[str]:
    issues: list[str] = []
    for i, rule in enumerate(rules, 1):
        if not rule.strip() or len(rule) > 140:
            issues.append(f"rule_{i}: empty or too long")
        if re.search(r"\b(and|or)\b", rule, re.IGNORECASE):
            issues.append(f"rule_{i}: multiple conditions")
        if re.search(r"\b(never not|don't not|do not not)\b", rule, re.IGNORECASE):
            issues.append(f"rule_{i}: double negative")
    return issues


def _p(answers: Mapping[str, Judgment], key: str) -> float | None:
    value = answers.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0 <= number <= 1 else None


def _severity(value: Judgment | None) -> float | None:
    labels = {"none": 0.0, "minor": 1.0, "moderate": 2.0, "serious": 3.0, "severe": 4.0}
    if isinstance(value, str):
        return labels.get(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0 <= number <= 4 else None


def motion_preflight(action: Action) -> Verdict | None:
    """Independent motion-state holds; model approval cannot override these."""
    if action.kind != "motion":
        return None
    if (
        not isinstance(action.nearest_person_distance, str)
        or not isinstance(action.battery, str)
        or not isinstance(action.motor_temperature, str)
        or action.nearest_person_distance not in _MOTION_DISTANCES
        or action.battery not in _BATTERY_BUCKETS
        or action.motor_temperature not in _MOTOR_TEMPERATURES
    ):
        return Verdict("hold", "motion_context_unavailable")
    if action.battery == "critical" or action.motor_temperature == "hot":
        return Verdict("hold", "motion_state_unsafe")
    if action.nearest_person_distance == "very near" and action.motion_class != "small_gesture":
        return Verdict("hold", "person_too_near")
    return None


def decide(action: Action, answers: Mapping[str, Judgment], policy: GuardPolicy) -> Verdict:
    """Missing/invalid answers always hold; no action is executed by this function."""
    preflight = motion_preflight(action)
    if preflight is not None:
        return preflight
    required = [f"violates_rule_{i}" for i in range(1, len(policy.rules) + 1)]
    required_by_kind = {
        "utterance": ["appropriate_for_audience", "discloses_private_info", "tone_hostile"],
        "tool_call": ["matches_request", "irreversible", "needs_confirmation"],
        "motion": ["safe_given_state", "startle_risk"],
        "inbound": ["injection", "asks_to_change_rules"],
    }
    required.extend(required_by_kind[action.kind])
    if action.kind == "utterance" and action.source == "camera_sign":
        required.append("follows_sign_instruction")
    if any(_p(answers, key) is None for key in required):
        return Verdict("hold", "missing_or_invalid_judgment")
    severity = _severity(answers.get("severity"))
    if severity is None:
        return Verdict("hold", "missing_or_invalid_severity")

    if action.kind == "inbound" and _p(answers, "injection") > policy.block_threshold:  # type: ignore[operator]
        return Verdict("block", "injection")
    for i in range(1, len(policy.rules) + 1):
        p = _p(answers, f"violates_rule_{i}")
        if p is not None and p > policy.block_threshold:
            return Verdict("block", f"rule_{i}")

    if action.kind == "utterance":
        if action.source == "camera_sign":
            p_sign = _p(answers, "follows_sign_instruction")
            if p_sign is not None and p_sign > policy.block_threshold:
                return Verdict("block", "sign_instruction_followed")
            if p_sign is not None and p_sign >= policy.hold_floor:
                return Verdict("hold", "uncertain_sign_instruction")
        if _p(answers, "discloses_private_info") > policy.block_threshold:  # type: ignore[operator]
            return Verdict("block", "private_info")
        if _p(answers, "appropriate_for_audience") < policy.hold_floor:  # type: ignore[operator]
            return Verdict("block", "audience")
        if _p(answers, "tone_hostile") > policy.block_threshold:  # type: ignore[operator]
            return Verdict("block", "hostile_tone")
        if _p(answers, "discloses_private_info") >= policy.hold_floor:  # type: ignore[operator]
            return Verdict("hold", "uncertain_private_info")
        if _p(answers, "appropriate_for_audience") < policy.block_threshold:  # type: ignore[operator]
            return Verdict("hold", "uncertain_audience")
        if _p(answers, "tone_hostile") >= policy.hold_floor:  # type: ignore[operator]
            return Verdict("hold", "uncertain_tone")
    elif action.kind == "tool_call":
        if _p(answers, "matches_request") < policy.hold_floor:  # type: ignore[operator]
            return Verdict("block", "request_mismatch")
        if _p(answers, "matches_request") < policy.block_threshold:  # type: ignore[operator]
            return Verdict("hold", "uncertain_request_match")
    elif action.kind == "motion":
        if _p(answers, "safe_given_state") < policy.hold_floor:  # type: ignore[operator]
            return Verdict("block", "unsafe_motion")
        if _p(answers, "safe_given_state") < policy.block_threshold:  # type: ignore[operator]
            return Verdict("hold", "uncertain_motion_safety")
        if _p(answers, "startle_risk") > 0.5:  # type: ignore[operator]
            return Verdict("hold", "startle_risk")

    for i in range(1, len(policy.rules) + 1):
        p = _p(answers, f"violates_rule_{i}")
        if p is not None and p >= policy.hold_floor:
            return Verdict("hold", f"uncertain_rule_{i}")
    if action.kind == "inbound" and _p(answers, "asks_to_change_rules") >= policy.hold_floor:  # type: ignore[operator]
        return Verdict("hold", "rule_change_request")
    # Confirmation holds are emitted only after all other fail-closed checks.
    # The owned pipeline may resume these specific holds after trusted approval.
    if action.kind == "tool_call":
        if action.tool in policy.confirm_before:
            return Verdict("hold", "confirm_before")
        if _p(answers, "needs_confirmation") > 0.5:  # type: ignore[operator]
            return Verdict("hold", "model_requests_confirmation")
        if _p(answers, "irreversible") > 0.5:  # type: ignore[operator]
            return Verdict("hold", "irreversible_tool")
    return Verdict("approve", "within_policy")


async def guard_action(
    action: Action,
    policy: GuardPolicy,
    ask: Callable[[Action, GuardPolicy], Awaitable[Mapping[str, Judgment]]],
) -> Verdict:
    """Call the model adapter without executing. A failure is a hold, never approval."""
    preflight = motion_preflight(action)
    if preflight is not None:
        return preflight
    try:
        return decide(action, await ask(action, policy), policy)
    except Exception:
        return Verdict("hold", "judgment_unavailable")


_POLITE_STOP = r"(?:please|s['’]il\s+te\s+pla[iî]t)"
_HARD_STOP = re.compile(
    rf"""^\s*
    (?:(?:reachy(?:\s+mini)?|robot)(?:\s*[:,]\s*|\s+))?
    (?:{_POLITE_STOP}\s+)?
    (?:stop|freeze|hold\s+still|arr[eê]te(?:-toi)?|stoppe)
    (?:\s+(?:now|moving|everything|the\s+robot|maintenant|le\s+robot))?
    (?:\s+{_POLITE_STOP})?
    [.!?]?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_hard_stop(text: str) -> bool:
    """Recognize a short direct stop command before Jev or an LLM round trip.

    This is deliberately a bounded grammar, not a substring match: quoted,
    negated, or embedded uses of "stop" must not cancel an unrelated turn.
    """
    return bool(_HARD_STOP.fullmatch(text))
