"""Optional, advisory Conscience-to-Reflex speaking assertion.

An accepted enqueue means audio *may* be playing. Only the operator can assert
quiet in this half-duplex host; neither assertion is a hardware receipt.
"""

from __future__ import annotations

import json
import math
import secrets
import threading
import time
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit

from .session import Playback


class ReflexSpeakingPublisher:
    """Best-effort loopback heartbeat with monotonic sequence fencing.

    Network errors never affect the guarded conversation. Reflex expires an
    unrefreshed assertion; this publisher must not be used as a stop control.
    """

    def __init__(
        self,
        relay_url: str,
        token: str,
        *,
        send: Callable[[str, int, bool], None] | None = None,
        interval_s: float = 0.75,
    ) -> None:
        parts = urlsplit(relay_url)
        if (
            parts.scheme != "http"
            or parts.hostname != "127.0.0.1"
            or parts.port is None
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
            or parts.username
            or parts.password
        ):
            raise ValueError("speaking relay must be a numeric 127.0.0.1 HTTP origin")
        if (
            not isinstance(token, str)
            or not 32 <= len(token) <= 256
            or not token.isascii()
            or not token.isprintable()
        ):
            raise ValueError("speaking writer token must be 32..256 printable ASCII characters")
        if not math.isfinite(interval_s) or not 0.5 <= interval_s <= 1:
            raise ValueError("speaking heartbeat interval must be 0.5..1 second")
        self._endpoint = f"http://127.0.0.1:{parts.port}/v1/speaking"
        self._token = token
        self._send = send or self._send_http
        self._interval_s = interval_s
        self._session = secrets.token_urlsafe(18)
        self._condition = threading.Condition()
        self._state: bool | None = None
        self._generation = 0
        self._sent_generation = -1
        self._next_heartbeat = float("inf")
        self._sequence = 0
        self._closed = False
        self._thread: threading.Thread | None = None

    def _send_http(self, session: str, sequence: int, speaking: bool) -> None:
        body = json.dumps(
            {"schema": "reflex.speaking@1", "session": session, "seq": sequence, "speaking": speaking},
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
            method="POST",
        )
        # Bypass user proxy settings for this numeric loopback-only channel.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=0.5) as response:
            if response.status != 202:
                raise OSError("speaking relay did not accept the assertion")

    def start(self) -> None:
        with self._condition:
            if self._closed or self._thread is not None:
                raise RuntimeError("speaking publisher cannot be started again")
            self._thread = threading.Thread(target=self._run, name="conscience-reflex-speaking", daemon=True)
            self._thread.start()

    def _set(self, state: bool) -> None:
        with self._condition:
            if self._closed or self._state is state:
                return
            self._state = state
            self._generation += 1
            self._condition.notify_all()

    def may_be_speaking(self) -> None:
        """Call only after owned audio enqueue returned successfully."""
        self._set(True)

    def operator_confirmed_quiet(self) -> None:
        """An explicit operator assertion, not a playback-complete receipt."""
        self._set(False)

    def _run(self) -> None:
        while True:
            with self._condition:
                while True:
                    if self._closed:
                        return
                    now = time.monotonic()
                    due = self._state is not None and (
                        self._generation != self._sent_generation
                        or (self._state and now >= self._next_heartbeat)
                    )
                    if due:
                        state = self._state
                        generation = self._generation
                        self._sequence += 1
                        sequence = self._sequence
                        break
                    wait_s = (
                        max(0, self._next_heartbeat - now)
                        if self._state and math.isfinite(self._next_heartbeat)
                        else None
                    )
                    self._condition.wait(wait_s)
            try:
                self._send(self._session, sequence, state)
            except Exception:
                pass  # A failed advisory feed must expire, never gate owned output.
            with self._condition:
                self._sent_generation = generation
                self._next_heartbeat = time.monotonic() + self._interval_s if state else float("inf")

    def close(self) -> None:
        """Stop heartbeats; the relay's last assertion expires without a silence claim."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=1)


class SpeakingObservedPlayback:
    """Notify the advisory publisher only after the owned SDK enqueue succeeds."""

    def __init__(self, backend: Playback, publisher: ReflexSpeakingPublisher) -> None:
        self.backend = backend
        self.publisher = publisher

    async def arm_playback(self) -> None:
        await self.backend.arm_playback()

    async def enqueue(self, audio: bytes) -> None:
        await self.backend.enqueue(audio)
        self.publisher.may_be_speaking()

    async def halt_audio(self) -> None:
        await self.backend.halt_audio()
