from __future__ import annotations

import json
import sqlite3

import pytest

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
    assert rows[0]["summary"] == "motion action"
    ledger.close()


def test_ledger_minimizes_sensitive_text_and_export_by_default(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    secret = "private address 12 Oak Street"
    ledger.append(
        Action(kind="tool_call", tool="send_message", summary=secret),
        Verdict("hold", "confirm_before"),
        {
            "irreversible": 0.8,
            "needs_confirmation": 0.7,
            "matches_request": 0.9,
            "bad": float("nan"),
            "bool": True,
        },
        12.5,
        bank="conscience.guard@0.1.0",
        model="fixture",
        outcome="cancelled",
        redteam=True,
    )
    row = ledger.recent()[0]
    assert row["summary"] == "tool_call action"
    assert row["redteam"] is True
    assert secret not in str(row)
    assert len(json.loads(row["probabilities"])) == 3
    exported = ledger.export_jsonl()
    assert secret not in exported
    value = json.loads(exported)
    assert value["schema"] == "conscience.verdict@1"
    assert value["probabilities"] == {"irreversible": 0.8, "matches_request": 0.9, "needs_confirmation": 0.7}
    assert value["redteam"] is True
    assert "summary" not in value
    assert json.loads(ledger.export_jsonl(include_summary=True))["summary"] == "tool_call action"
    ledger.close()


def test_ledger_opt_in_summary_and_legacy_migration(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE verdicts (id INTEGER PRIMARY KEY, t REAL NOT NULL, kind TEXT NOT NULL, "
        "summary TEXT NOT NULL, verdict TEXT NOT NULL, reason TEXT NOT NULL, "
        "probabilities TEXT NOT NULL, latency_ms REAL NOT NULL)"
    )
    connection.execute(
        "INSERT INTO verdicts(t, kind, summary, verdict, reason, probabilities, latency_ms) "
        "VALUES (1, 'utterance', 'legacy note', 'approve', 'ok', '{}', 2)"
    )
    connection.commit()
    connection.close()

    ledger = Ledger(path, store_summaries=True)
    assert ledger.recent()[0]["summary"] == "legacy note"
    ledger.append(Action(kind="utterance", summary="opted-in note"), Verdict("approve", "ok"), {}, 1.0)
    assert ledger.recent()[0]["summary"] == "opted-in note"
    assert "legacy note" not in ledger.export_jsonl()
    assert "opted-in note" not in ledger.export_jsonl()
    assert "legacy note" in ledger.export_jsonl(include_summary=True)
    ledger.close()

    default_reader = Ledger(path)
    assert all(row["summary"].endswith(" action") for row in default_reader.recent())
    default_reader.close()


def test_ledger_rejects_nonfinite_latency_and_bad_metadata(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    action = Action(kind="inbound", summary="inbound")
    for value in [float("nan"), float("inf"), -1]:
        with pytest.raises(ValueError, match="latency"):
            ledger.append(action, Verdict("hold", "fixture"), {}, value)
    with pytest.raises(ValueError, match="outcome"):
        ledger.append(action, Verdict("hold", "fixture"), {}, 1, outcome="sent")
    ledger.append(action, Verdict("hold", "sensitive reason: 12 Oak Street"), {}, 1)
    assert ledger.recent()[0]["reason"] == "custom_reason"
    ledger.close()
