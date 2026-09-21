"""Local, text-minimizing SQLite verdict ledger."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from .guard import Action, Verdict


class Ledger:
    """Append-only local verdict ledger with text-free storage by default."""

    def __init__(self, path: str | Path, *, store_summaries: bool = False) -> None:
        if not isinstance(store_summaries, bool):
            raise TypeError("store_summaries must be a boolean")
        self.store_summaries = store_summaries
        self.connection = sqlite3.connect(path)
        with self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS verdicts ("
                "id INTEGER PRIMARY KEY, t REAL NOT NULL, kind TEXT NOT NULL, summary TEXT NOT NULL, "
                "verdict TEXT NOT NULL, reason TEXT NOT NULL, "
                "probabilities TEXT NOT NULL, latency_ms REAL NOT NULL, "
                "bank TEXT, model TEXT, outcome TEXT, redteam INTEGER NOT NULL DEFAULT 0)"
            )
            # Existing preview databases had eight columns. Add only missing
            # metadata; never rewrite or discard prior verdict rows.
            columns = {row[1] for row in self.connection.execute("PRAGMA table_info(verdicts)")}
            for name, definition in (
                ("bank", "TEXT"),
                ("model", "TEXT"),
                ("outcome", "TEXT"),
                ("redteam", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    self.connection.execute(f"ALTER TABLE verdicts ADD COLUMN {name} {definition}")

    def append(
        self,
        action: Action,
        verdict: Verdict,
        probabilities: dict[str, float],
        latency_ms: float,
        *,
        bank: str | None = None,
        model: str | None = None,
        outcome: str | None = None,
        redteam: bool = False,
    ) -> int:
        if (
            not isinstance(latency_ms, (int, float))
            or isinstance(latency_ms, bool)
            or not math.isfinite(latency_ms)
            or latency_ms < 0
        ):
            raise ValueError("latency must be non-negative")
        if any(
            value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 100)
            for value in (bank, model)
        ):
            raise ValueError("invalid bank or model label")
        if outcome not in {None, "executed", "confirmed", "cancelled", "refused", "timed_out"}:
            raise ValueError("invalid outcome")
        if not isinstance(redteam, bool):
            raise TypeError("redteam must be a boolean")
        safe = {
            key: float(value)
            for key, value in probabilities.items()
            if isinstance(key, str)
            and key.isidentifier()
            and len(key) <= 80
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        }
        top_three = dict(sorted(safe.items(), key=lambda item: (-item[1], item[0]))[:3])
        summary = action.summary[:140] if self.store_summaries else f"{action.kind} action"
        reason = verdict.reason if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", verdict.reason) else "custom_reason"
        cursor = self.connection.execute(
            "INSERT INTO verdicts(t, kind, summary, verdict, reason, probabilities, latency_ms, "
            "bank, model, outcome, redteam) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                action.kind,
                summary,
                verdict.kind,
                reason,
                json.dumps(top_three, sort_keys=True),
                latency_ms,
                bank,
                model,
                outcome,
                int(redteam),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be in [1,1000]")
        rows = self.connection.execute(
            "SELECT id, t, kind, summary, verdict, reason, probabilities, latency_ms, "
            "bank, model, outcome, redteam "
            "FROM verdicts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        records = [
            dict(
                zip(
                    (
                        "id",
                        "t",
                        "kind",
                        "summary",
                        "verdict",
                        "reason",
                        "probabilities",
                        "latency_ms",
                        "bank",
                        "model",
                        "outcome",
                        "redteam",
                    ),
                    row,
                    strict=True,
                )
            )
            for row in rows
        ]
        if not self.store_summaries:
            for record in records:
                record["summary"] = f"{record['kind']} action"
        for record in records:
            record["redteam"] = bool(record["redteam"])
        return records

    def export_jsonl(self, *, include_summary: bool = False) -> str:
        """Export chronological verdicts, omitting even stored summaries by default."""
        if not isinstance(include_summary, bool):
            raise TypeError("include_summary must be a boolean")
        rows = self.connection.execute(
            "SELECT id, t, kind, summary, verdict, reason, probabilities, latency_ms, "
            "bank, model, outcome, redteam FROM verdicts ORDER BY id ASC"
        )
        lines = []
        for row in rows:
            record = {
                "schema": "conscience.verdict@1",
                "id": row[0],
                "t": row[1],
                "kind": row[2],
                "verdict": row[4],
                "reason": row[5],
                "probabilities": json.loads(row[6]),
                "latency_ms": row[7],
                "bank": row[8],
                "model": row[9],
                "outcome": row[10],
                "redteam": bool(row[11]),
            }
            if include_summary:
                record["summary"] = row[3]
            lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return "\n".join(lines) + ("\n" if lines else "")

    def close(self) -> None:
        self.connection.close()
