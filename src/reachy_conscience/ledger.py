"""Local, text-minimizing SQLite verdict ledger."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .guard import Action, Verdict


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS verdicts ("
            "id INTEGER PRIMARY KEY, t REAL NOT NULL, kind TEXT NOT NULL, summary TEXT NOT NULL, "
            "verdict TEXT NOT NULL, reason TEXT NOT NULL, "
            "probabilities TEXT NOT NULL, latency_ms REAL NOT NULL)"
        )
        self.connection.commit()

    def append(
        self,
        action: Action,
        verdict: Verdict,
        probabilities: dict[str, float],
        latency_ms: float,
    ) -> int:
        if latency_ms < 0:
            raise ValueError("latency must be non-negative")
        safe = {key: value for key, value in probabilities.items() if key.isidentifier() and 0 <= value <= 1}
        cursor = self.connection.execute(
            "INSERT INTO verdicts(t, kind, summary, verdict, reason, probabilities, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                action.kind,
                action.summary[:140],
                verdict.kind,
                verdict.reason,
                json.dumps(safe),
                latency_ms,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be in [1,1000]")
        rows = self.connection.execute(
            "SELECT id, t, kind, summary, verdict, reason, probabilities, latency_ms "
            "FROM verdicts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            dict(
                zip(
                    ("id", "t", "kind", "summary", "verdict", "reason", "probabilities", "latency_ms"),
                    row,
                    strict=True,
                )
            )
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()
