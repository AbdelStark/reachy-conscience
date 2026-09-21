"""One-shot, exact-argument owner approval shared by an async turn and local UI.

The broker never executes a tool. The authenticated host must expose its
pending request to a human owner and call ``decide`` with the same ID/digest.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate tool argument")
        value[key] = item
    return value


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    tool: str
    arguments_json: str
    digest: str
    remaining_s: float


@dataclass(slots=True)
class _Pending:
    request_id: str
    tool: str
    arguments_json: str
    digest: str
    deadline: float
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[bool]


class OwnerApprovalBroker:
    """Fail-closed, one-pending-request bridge for an authenticated local UI.

    ``authorize`` implements the pipeline's OwnerApproval port. It is an
    async wait, not an automatic approval. Only ``decide`` can resolve it
    true, and decisions bind the exact tool name and canonical arguments.
    Both timeout and coroutine cancellation remove the pending request.
    """

    def __init__(self, *, timeout_s: float = 10.0) -> None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= 30:
            raise ValueError("owner approval timeout must be within (0,30]")
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._pending: _Pending | None = None

    @staticmethod
    def _validate(tool: str, arguments_json: str) -> None:
        if not isinstance(tool, str) or not tool.isidentifier() or len(tool) > 80:
            raise ValueError("invalid tool name")
        if not isinstance(arguments_json, str) or len(arguments_json.encode("utf-8")) > 4_096:
            raise ValueError("tool arguments exceed limit")
        try:
            arguments = json.loads(arguments_json, object_pairs_hook=_unique_object)
            canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("invalid tool arguments") from exc
        if not isinstance(arguments, dict) or canonical != arguments_json:
            raise ValueError("tool arguments must be a canonical JSON object")

    async def authorize(self, tool: str, arguments_json: str) -> bool:
        self._validate(tool, arguments_json)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        request_id = secrets.token_urlsafe(24)
        digest = hashlib.sha256(f"{tool}\0{arguments_json}".encode()).hexdigest()
        pending = _Pending(
            request_id,
            tool,
            arguments_json,
            digest,
            time.monotonic() + self.timeout_s,
            loop,
            future,
        )
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("another owner approval is pending")
            self._pending = pending
        try:
            try:
                return await asyncio.wait_for(future, self.timeout_s)
            except TimeoutError:
                return False
        finally:
            with self._lock:
                if self._pending is pending:
                    self._pending = None

    def pending(self) -> ApprovalRequest | None:
        with self._lock:
            entry = self._pending
            if entry is None or time.monotonic() >= entry.deadline:
                return None
            return ApprovalRequest(
                entry.request_id,
                entry.tool,
                entry.arguments_json,
                entry.digest,
                max(0.0, entry.deadline - time.monotonic()),
            )

    @staticmethod
    def _resolve(entry: _Pending, approved: bool) -> None:
        def finish() -> None:
            if not entry.future.done():
                entry.future.set_result(approved)

        try:
            entry.loop.call_soon_threadsafe(finish)
        except RuntimeError:
            pass  # A closed loop cannot dispatch the action.

    def decide(self, request_id: str, digest: str, *, approve: bool) -> bool:
        """Consume only the currently pending exact request; never dispatch."""
        if not isinstance(approve, bool):
            return False
        with self._lock:
            entry = self._pending
            if (
                entry is None
                or time.monotonic() >= entry.deadline
                or not isinstance(request_id, str)
                or not isinstance(digest, str)
                or not secrets.compare_digest(request_id, entry.request_id)
                or not secrets.compare_digest(digest, entry.digest)
            ):
                return False
            self._pending = None
        self._resolve(entry, approve)
        return True

    def cancel_all(self) -> None:
        """Invalidate a pending request, including one shown in a stale tab."""
        with self._lock:
            entry = self._pending
            self._pending = None
        if entry is not None:
            self._resolve(entry, False)
