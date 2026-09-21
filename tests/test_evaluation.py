"""Only synthetic fixture inputs; no model, robot, or consented data."""

import asyncio

import pytest

from reachy_conscience.evaluation import LabeledCase, run_labeled_evaluation
from reachy_conscience.guard import Action, GuardAssessment, Verdict


def labeled_case(case_id, expected, text="fixture"):
    return LabeledCase(case_id, expected, Action("inbound", "synthetic fixture", untrusted_text=text))


@pytest.mark.asyncio
async def test_matrix_keeps_holds_out_of_block_and_false_block_numerators():
    cases = (
        labeled_case("attack_one", "attack"),
        labeled_case("attack_two", "attack"),
        labeled_case("attack_three", "attack"),
        labeled_case("benign_one", "benign"),
        labeled_case("benign_two", "benign"),
        labeled_case("benign_three", "benign"),
    )
    outcomes = iter(("block", "hold", "approve", "block", "hold", "approve"))

    async def guard(_action):
        return GuardAssessment(Verdict(next(outcomes), "fixture"))

    report = await run_labeled_evaluation(guard, cases)
    assert report.matrix == {
        "attack": {"block": 1, "hold": 1, "approve": 1},
        "benign": {"block": 1, "hold": 1, "approve": 1},
    }
    assert report.block_rate == report.false_block_rate == 1 / 3
    assert [result.case_id for result in report.results] == [case.case_id for case in cases]
    assert report.p95_latency_ms >= 0


@pytest.mark.asyncio
async def test_timeout_exception_and_bad_verdict_fail_to_hold_and_continue():
    cases = tuple(labeled_case(f"attack_{n}", "attack") for n in range(3)) + (
        labeled_case("benign_one", "benign"),
    )
    calls = 0

    async def guard(_action):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.02)
        if calls == 2:
            raise RuntimeError("fixture unavailable")
        if calls == 3:
            return "block"
        return Verdict("approve", "fixture")

    report = await run_labeled_evaluation(guard, cases, timeout_s=0.001)
    assert calls == 4
    assert report.matrix == {
        "attack": {"block": 0, "hold": 3, "approve": 0},
        "benign": {"block": 0, "hold": 0, "approve": 1},
    }
    assert report.block_rate == report.false_block_rate == 0
    assert [result.status for result in report.results] == [
        "timeout",
        "guard_error",
        "malformed_verdict",
        "guard_verdict",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cases",
    [
        (),
        (labeled_case("attack_one", "attack"),),
        (labeled_case("benign_one", "benign"),),
        (labeled_case("same", "attack"), labeled_case("same", "benign")),
        (labeled_case("attack_one", "attack", " "), labeled_case("benign_one", "benign")),
        (labeled_case("attack_one", "unknown"), labeled_case("benign_one", "benign")),
        (labeled_case("attack_one", [], "fixture"), labeled_case("benign_one", "benign")),
        (labeled_case("attack_one", "attack"), object()),
    ],
)
async def test_invalid_cohorts_rejected_before_guard_is_called(cases):
    calls = 0

    async def guard(_action):
        nonlocal calls
        calls += 1
        return Verdict("approve", "fixture")

    with pytest.raises(ValueError):
        await run_labeled_evaluation(guard, cases)
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, 31, float("nan"), float("inf"), True, "2"])
async def test_invalid_timeout_rejected(timeout):
    async def guard(_action):
        return Verdict("approve", "fixture")

    with pytest.raises(ValueError, match="timeout"):
        await run_labeled_evaluation(
            guard,
            (labeled_case("attack_one", "attack"), labeled_case("benign_one", "benign")),
            timeout_s=timeout,
        )
