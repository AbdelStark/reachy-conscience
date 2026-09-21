from __future__ import annotations

from reachy_conscience import Action, HoldRegistry, Ledger, Verdict


def test_hold_is_one_shot_and_expires() -> None:
    now = 0.0
    registry = HoldRegistry(timeout_s=10, clock=lambda: now)
    action = Action(kind="tool_call", tool="send_message", summary="send reminder")
    token = registry.hold(action)
    assert registry.confirm(token) == action
    assert registry.confirm(token) is None
    token = registry.hold(action)
    now = 10.0
    assert registry.confirm(token) is None


def test_ledger_persists_minimal_verdicts(tmp_path) -> None:
    path = tmp_path / "ledger.db"
    ledger = Ledger(path)
    row_id = ledger.append(
        Action(kind="motion", motion_class="fast_turn", summary="fast turn"),
        Verdict("hold", "startle_risk"),
        {"startle_risk": 0.8},
        12.5,
    )
    rows = ledger.recent()
    assert rows[0]["id"] == row_id
    assert rows[0]["reason"] == "startle_risk"
    assert "0.8" in rows[0]["probabilities"]
    ledger.close()
