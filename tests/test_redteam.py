"""Synthetic corpus and dispatch-free run; no model, robot, or output port."""

import asyncio

import pytest

from reachy_conscience import GuardAssessment, Verdict
from reachy_conscience.redteam import redteam_cases, run_redteam


def test_packaged_corpus_has_20_distinct_synthetic_prompts():
    cases = redteam_cases()
    assert len(cases) == len({case.case_id for case in cases}) == 20
    assert sum(case.intent == "challenge" for case in cases) == 10
    assert sum(case.intent == "benign" for case in cases) == 10
    assert {case.channel for case in cases} == {"spoken", "sign"}
    assert all(case.action.kind == "inbound" and case.action.untrusted_text for case in cases)
    assert all("synthetic" in case.action.summary for case in cases)


@pytest.mark.asyncio
async def test_redteam_counts_verdicts_and_never_has_an_output_port():
    seen = []

    async def guard(action):
        seen.append(action)
        if len(seen) == 3:
            raise OSError("fixture guard unavailable")
        return GuardAssessment(Verdict("block" if len(seen) % 2 else "approve", "fixture"))

    report = await run_redteam(guard)
    assert len(seen) == len(report.results) == 20
    assert report.counts == {"approve": 10, "hold": 1, "block": 9}
    assert report.results[2].verdict == Verdict("hold", "judgment_unavailable")
    assert report.p95_latency_ms >= 0
    assert all(result.latency_ms >= 0 for result in report.results)


@pytest.mark.asyncio
async def test_redteam_timeout_and_malformed_verdict_hold_without_short_circuit():
    calls = 0

    async def guard(_action):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.02)
        if calls == 2:
            return "approve"  # A string is not a typed verdict.
        return Verdict("approve", "fixture")

    report = await run_redteam(guard, timeout_s=0.001)
    assert calls == 20
    assert [item.verdict.kind for item in report.results[:3]] == ["hold", "hold", "approve"]
    assert report.counts == {"approve": 18, "hold": 2, "block": 0}


@pytest.mark.asyncio
async def test_redteam_rejects_invalid_timeout():
    async def guard(_action):
        return Verdict("approve", "fixture")

    with pytest.raises(ValueError, match="timeout"):
        await run_redteam(guard, timeout_s=0)
