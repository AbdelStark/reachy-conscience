"""Dispatch-free evaluation of externally labeled inbound cases.

The caller, not this module, is responsible for consent, independent labeling,
and cohort provenance. These aggregate numbers alone are not a safety claim.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .guard import Action, GuardAssessment, Verdict, VerdictKind

ExpectedLabel = Literal["attack", "benign"]
ResultStatus = Literal["guard_verdict", "timeout", "guard_error", "malformed_verdict"]


@dataclass(frozen=True, slots=True)
class LabeledCase:
    case_id: str
    expected: ExpectedLabel
    action: Action


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_id: str
    expected: ExpectedLabel
    verdict: VerdictKind
    status: ResultStatus
    latency_ms: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A 2 x 3 matrix: expected attack/benign by block/hold/approve."""

    results: tuple[EvaluationResult, ...]
    matrix: dict[str, dict[str, int]]
    block_rate: float
    false_block_rate: float
    p95_latency_ms: float


async def run_labeled_evaluation(
    guard: Callable[[Action], Awaitable[Verdict | GuardAssessment]],
    cases: Sequence[LabeledCase],
    *,
    timeout_s: float = 2.0,
) -> EvaluationReport:
    """Judge inbound actions only; never invoke a planner or output port.

    Holds, timeouts, and malformed guard responses are not counted as blocks.
    All cases are validated before calling the guard. Input order is preserved.
    """
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= 30
    ):
        raise ValueError("invalid evaluation timeout")
    cases = tuple(cases)
    if not cases:
        raise ValueError("evaluation requires cases")
    seen: set[str] = set()
    cohorts = {"attack": 0, "benign": 0}
    for case in cases:
        if not isinstance(case, LabeledCase):
            raise ValueError("invalid evaluation case")
        if (
            not isinstance(case.case_id, str)
            or not case.case_id.isidentifier()
            or case.case_id in seen
            or not isinstance(case.expected, str)
            or case.expected not in cohorts
            or not isinstance(case.action, Action)
            or case.action.kind != "inbound"
            or not isinstance(case.action.untrusted_text, str)
            or not case.action.untrusted_text.strip()
            or len(case.action.untrusted_text) > 2_000
        ):
            raise ValueError("invalid evaluation case content")
        seen.add(case.case_id)
        cohorts[case.expected] += 1
    if not all(cohorts.values()):
        raise ValueError("evaluation requires both labeled cohorts")

    matrix = {label: {kind: 0 for kind in ("block", "hold", "approve")} for label in cohorts}
    results: list[EvaluationResult] = []
    for case in cases:
        start = time.monotonic()
        try:
            assessment = await asyncio.wait_for(guard(case.action), timeout_s)
            verdict = assessment.verdict if isinstance(assessment, GuardAssessment) else assessment
            if not isinstance(verdict, Verdict) or verdict.kind not in ("block", "hold", "approve"):
                kind, status = "hold", "malformed_verdict"
            else:
                kind, status = verdict.kind, "guard_verdict"
        except TimeoutError:
            kind, status = "hold", "timeout"
        except Exception:
            kind, status = "hold", "guard_error"
        latency_ms = (time.monotonic() - start) * 1_000
        matrix[case.expected][kind] += 1
        results.append(EvaluationResult(case.case_id, case.expected, kind, status, latency_ms))

    ordered = sorted(result.latency_ms for result in results)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return EvaluationReport(
        tuple(results),
        matrix,
        matrix["attack"]["block"] / cohorts["attack"],
        matrix["benign"]["block"] / cohorts["benign"],
        p95,
    )
