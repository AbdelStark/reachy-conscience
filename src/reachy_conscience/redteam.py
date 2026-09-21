"""Dispatch-free synthetic challenge run; results are not model validation.

The corpus contains self-authored prompts and benign look-alikes, not consented
Conversation App sessions. Intent labels describe why a case was written, not
ground-truth guard verdicts or measured safety.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.resources import files

from .guard import Action, GuardAssessment, Verdict


@dataclass(frozen=True, slots=True)
class RedTeamCase:
    case_id: str
    channel: str
    intent: str
    action: Action


@dataclass(frozen=True, slots=True)
class RedTeamResult:
    case_id: str
    channel: str
    intent: str
    verdict: Verdict
    latency_ms: float


@dataclass(frozen=True, slots=True)
class RedTeamReport:
    results: tuple[RedTeamResult, ...]
    counts: dict[str, int]
    p95_latency_ms: float


def redteam_cases() -> tuple[RedTeamCase, ...]:
    """Load the fixed packaged corpus; callers cannot supply arbitrary prompts."""
    raw = files("reachy_conscience").joinpath("assets", "redteam.json").read_bytes()
    if len(raw) > 16_384:
        raise ValueError("red-team corpus exceeds limit")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"schema", "cases"}:
        raise ValueError("invalid red-team corpus")
    if value["schema"] != "conscience.synthetic-redteam@1":
        raise ValueError("unsupported red-team corpus")
    rows = value["cases"]
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("red-team corpus must have 20 cases")
    cases: list[RedTeamCase] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "channel", "intent", "text"}:
            raise ValueError("invalid red-team case")
        case_id, channel, intent, prompt = (row[key] for key in ("id", "channel", "intent", "text"))
        if (
            not isinstance(case_id, str)
            or not case_id.isidentifier()
            or case_id in seen
            or channel not in ("spoken", "sign")
            or intent not in ("challenge", "benign")
            or not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > 2_000
        ):
            raise ValueError("invalid red-team case content")
        seen.add(case_id)
        cases.append(
            RedTeamCase(
                case_id,
                channel,
                intent,
                Action(kind="inbound", summary=f"synthetic {channel} red-team case", untrusted_text=prompt),
            )
        )
    return tuple(cases)


async def run_redteam(
    guard: Callable[[Action], Awaitable[Verdict | GuardAssessment]],
    *,
    timeout_s: float = 2.0,
) -> RedTeamReport:
    """Run guard judgments only, sequentially, with no planner or output port."""
    if not 0 < timeout_s <= 30:
        raise ValueError("invalid red-team timeout")
    results: list[RedTeamResult] = []
    for case in redteam_cases():
        start = time.monotonic()
        try:
            assessment = await asyncio.wait_for(guard(case.action), timeout_s)
            verdict = assessment.verdict if isinstance(assessment, GuardAssessment) else assessment
            if not isinstance(verdict, Verdict) or verdict.kind not in ("approve", "hold", "block"):
                raise ValueError("invalid red-team verdict")
        except Exception:
            verdict = Verdict("hold", "judgment_unavailable")
        results.append(
            RedTeamResult(case.case_id, case.channel, case.intent, verdict, (time.monotonic() - start) * 1000)
        )
    counts = {
        kind: sum(item.verdict.kind == kind for item in results) for kind in ("approve", "hold", "block")
    }
    ordered = sorted(item.latency_ms for item in results)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return RedTeamReport(tuple(results), counts, p95)
