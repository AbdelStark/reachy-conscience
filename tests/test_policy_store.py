"""Policy file and dry-run tests use only synthetic actions and fake judgments."""

import json
import stat

import pytest

from reachy_conscience import Action, GuardAssessment, GuardPolicy, Verdict
from reachy_conscience.policy_store import (
    DEFAULT_RULES,
    POLICY_SCHEMA,
    PolicyStore,
    dry_run_policy,
    policy_from_lines,
    preview_cases,
)


def test_policy_text_lint_defaults_and_confirm_before():
    assert policy_from_lines("\n").rules == DEFAULT_RULES
    policy = policy_from_lines(
        "Never reveal a password\nDo not startle a nearby person\n", confirm_before=["send_message"]
    )
    assert policy.rules == DEFAULT_RULES
    assert policy.confirm_before == frozenset({"send_message"})
    with pytest.raises(ValueError, match="one clear condition"):
        policy_from_lines("Never reveal a password and never leave the room")
    with pytest.raises(ValueError, match="too many"):
        policy_from_lines("\n".join(f"Rule {index}" for index in range(21)))
    with pytest.raises(ValueError, match="not text"):
        policy_from_lines("Never reveal a password", confirm_before="send_message")


def test_policy_store_atomic_private_round_trip_and_registered_tool_gate(tmp_path):
    path = tmp_path / "policy.json"
    store = PolicyStore(path, registered_tools={"send_message"})
    assert store.load().rules == DEFAULT_RULES
    policy = policy_from_lines("Never reveal a password", confirm_before={"send_message"})
    store.save(policy)
    assert store.load() == policy
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["schema"] == POLICY_SCHEMA
    with pytest.raises(ValueError, match="not registered"):
        store.save(GuardPolicy(rules=DEFAULT_RULES, confirm_before=frozenset({"purchase"})))
    assert store.load() == policy


def test_policy_store_rejects_corrupt_or_oversized_files(tmp_path):
    path = tmp_path / "policy.json"
    store = PolicyStore(path)
    path.write_text("not json")
    with pytest.raises(ValueError, match="JSON"):
        store.load()
    path.write_text("x" * 8_193)
    with pytest.raises(ValueError, match="byte limit"):
        store.load()
    path.write_text(json.dumps({"schema": "unknown"}))
    with pytest.raises(ValueError, match="schema"):
        store.load()
    path.write_text('{"schema":"conscience.policy@1","schema":"conscience.policy@1"}')
    with pytest.raises(ValueError, match="duplicate"):
        store.load()


@pytest.mark.asyncio
async def test_five_action_preview_cannot_dispatch_and_failures_hold():
    seen: list[Action] = []

    async def fake_guard(action):
        seen.append(action)
        if action.kind == "motion":
            raise OSError("fixture model unavailable")
        return GuardAssessment(
            Verdict("block" if action.kind == "inbound" else "approve", "fixture"), {"p": 0.8}
        )

    results = await dry_run_policy(fake_guard)
    assert len(results) == 5
    assert [result.case_id for result in results] == [case.case_id for case in preview_cases()]
    assert len(seen) == 5
    assert results[2].verdict == Verdict("hold", "judgment_unavailable")
    assert results[2].probabilities == {}
    assert results[4].verdict.kind == "block"
    assert all(result.latency_ms >= 0 for result in results)
