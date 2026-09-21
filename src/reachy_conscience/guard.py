"""Pure, fail-closed policy for model judgments about proposed robot actions."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["utterance", "tool_call", "motion", "inbound"]
VerdictKind = Literal["approve", "hold", "block"]
Judgment = float | str


@dataclass(frozen=True, slots=True)
class Action:
    kind: Kind
    summary: str
    tool: str | None = None
    tool_arguments_json: str | None = None
    motion_class: str | None = None
    motion_target_json: str | None = None
    untrusted_text: str | None = None
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


def decide(action: Action, answers: Mapping[str, Judgment], policy: GuardPolicy) -> Verdict:
    """Missing/invalid answers always hold; no action is executed by this function."""
    required = [f"violates_rule_{i}" for i in range(1, len(policy.rules) + 1)]
    required_by_kind = {
        "utterance": ["appropriate_for_audience", "discloses_private_info", "tone_hostile"],
        "tool_call": ["matches_request", "irreversible", "needs_confirmation"],
        "motion": ["safe_given_state", "startle_risk"],
        "inbound": ["injection", "asks_to_change_rules"],
    }
    required.extend(required_by_kind[action.kind])
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
    try:
        return decide(action, await ask(action, policy), policy)
    except Exception:
        return Verdict("hold", "judgment_unavailable")


_HARD_STOP = re.compile(r"^\s*(stop|freeze|hold still|arr[eê]te|stoppe)[.!]?\s*$", re.IGNORECASE)


def is_hard_stop(text: str) -> bool:
    """Recognize an isolated emergency command before Jev or an LLM round trip."""
    return bool(_HARD_STOP.fullmatch(text))
