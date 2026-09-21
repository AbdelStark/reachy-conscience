"""Read-only, expiring Reflex event hints for an operator-paced host.

These hints never authorize speech, microphone capture, tool dispatch, motion,
or an emergency stop. The relay has no replay or consumer acknowledgement.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

MAX_FRAME_BYTES = 512
MAX_EVENT_AGE_MS = 1500
MAX_FUTURE_SKEW_MS = 250
MAX_SEQUENCE = 2**53 - 1


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate event field")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ReflexHint:
    kind: str
    seq: int
    person: str | None
    probability: float | None
    received_at_monotonic: float


class ReflexEventInbox:
    """Validate one relay connection's text-free hints and expire them locally."""

    def __init__(
        self,
        *,
        wall_ms: Callable[[], float] = lambda: time.time() * 1000,
        monotonic_s: Callable[[], float] = time.monotonic,
    ) -> None:
        self._wall_ms = wall_ms
        self._monotonic_s = monotonic_s
        self._lock = threading.Lock()
        self._last_seq = 0
        self._latest: ReflexHint | None = None

    def reset_connection(self) -> None:
        """A reconnect starts a new relay sequence; old hints do not survive it."""
        with self._lock:
            self._last_seq = 0
            self._latest = None

    def accept(self, frame: str | bytes) -> bool:
        """Ignore malformed, stale, future, replayed, or non-event frames."""
        if not isinstance(frame, (str, bytes)):
            return False
        try:
            encoded = frame.encode("utf-8") if isinstance(frame, str) else frame
            if not 0 < len(encoded) <= MAX_FRAME_BYTES:
                return False
            value = json.loads(encoded, object_pairs_hook=_unique_object)
            if not isinstance(value, dict) or set(value) != {"schema", "seq", "t_ms", "event"}:
                return False
            seq = value["seq"]
            stamp = value["t_ms"]
            event = value["event"]
            if (
                value["schema"] != "reflex.event@1"
                or isinstance(seq, bool)
                or not isinstance(seq, int)
                or not 1 <= seq <= MAX_SEQUENCE
                or isinstance(stamp, bool)
                or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp)
                or not isinstance(event, dict)
            ):
                return False
            age_ms = self._wall_ms() - stamp
            if not -MAX_FUTURE_SKEW_MS <= age_ms <= MAX_EVENT_AGE_MS:
                return False
            kind = event.get("type")
            person = event.get("person")
            p = event.get("p")
            if kind == "attention":
                if set(event) != {"type", "person"}:
                    return False
                probability = None
            elif kind == "user_addressed":
                if set(event) != {"type", "person", "p"}:
                    return False
                probability = p
            elif kind in {"yield", "interrupt"}:
                if set(event) != {"type", "p"}:
                    return False
                person = None
                probability = p
            else:
                return False
            if person is not None and (not isinstance(person, str) or not re.fullmatch(r"p[1-9]", person)):
                return False
            if kind in {"attention", "user_addressed"} and person is None:
                return False
            if probability is not None and (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                return False
            received = self._monotonic_s()
            with self._lock:
                if seq <= self._last_seq:
                    return False
                self._last_seq = seq
                self._latest = ReflexHint(
                    kind, seq, person, float(probability) if probability is not None else None, received
                )
            return True
        except (UnicodeError, ValueError, TypeError, OverflowError):
            return False

    def latest(self) -> ReflexHint | None:
        with self._lock:
            hint = self._latest
        if (
            hint is None
            or not 0 <= self._monotonic_s() - hint.received_at_monotonic <= MAX_EVENT_AGE_MS / 1000
        ):
            return None
        return hint


class ReflexEventMonitor:
    """Optional authenticated loopback subscriber, with no output handles."""

    def __init__(self, relay_url: str, token: str) -> None:
        parts = urlsplit(relay_url)
        if (
            parts.scheme != "ws"
            or parts.hostname != "127.0.0.1"
            or parts.port is None
            or parts.path != "/v1/events"
            or parts.query
            or parts.fragment
            or parts.username
            or parts.password
        ):
            raise ValueError("Reflex event URL must be ws://127.0.0.1:<port>/v1/events")
        if (
            not isinstance(token, str)
            or not 32 <= len(token) <= 256
            or not token.isascii()
            or not token.isprintable()
        ):
            raise ValueError("Reflex event subscriber token must be 32..256 printable ASCII characters")
        self._url = relay_url
        self._token = token
        self._inbox = ReflexEventInbox()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._connection: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None or self._closed.is_set():
            raise RuntimeError("Reflex event monitor cannot be started again")
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("install the events extra to monitor Reflex hints") from exc
        self._thread = threading.Thread(
            target=self._run, args=(connect,), name="conscience-reflex-events", daemon=True
        )
        self._thread.start()

    def _run(self, connect: Callable[..., Any]) -> None:
        while not self._closed.is_set():
            self._inbox.reset_connection()
            try:
                with connect(
                    self._url,
                    additional_headers={"Authorization": f"Bearer {self._token}"},
                    proxy=None,
                    compression=None,
                    max_size=MAX_FRAME_BYTES,
                    max_queue=2,
                    open_timeout=2,
                    close_timeout=1,
                ) as connection:
                    with self._lock:
                        self._connection = connection
                    while not self._closed.is_set():
                        try:
                            frame = connection.recv(timeout=0.25)
                        except TimeoutError:
                            continue
                        self._inbox.accept(frame)
            except Exception:
                pass  # Missing or failed advisory feed must never affect owned output.
            finally:
                with self._lock:
                    self._connection = None
                self._inbox.reset_connection()
            self._closed.wait(0.5)

    def latest(self) -> ReflexHint | None:
        return self._inbox.latest()

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            connection = self._connection
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)
