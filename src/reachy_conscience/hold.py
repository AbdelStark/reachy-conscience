"""One-shot, expiring owner confirmations. No motor/tool execution occurs here."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable

from .guard import Action


class HoldRegistry:
    def __init__(self, timeout_s: float = 10, clock: Callable[[], float] = time.monotonic) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout must be positive")
        self.timeout_s = timeout_s
        self.clock = clock
        self._pending: dict[str, tuple[float, Action]] = {}

    def hold(self, action: Action) -> str:
        self.expire()
        token = secrets.token_urlsafe(24)
        self._pending[token] = (self.clock() + self.timeout_s, action)
        return token

    def confirm(self, token: str) -> Action | None:
        self.expire()
        entry = self._pending.pop(token, None)
        return entry[1] if entry else None

    def cancel(self, token: str) -> bool:
        return self._pending.pop(token, None) is not None

    def expire(self) -> int:
        now = self.clock()
        expired = [token for token, (deadline, _) in self._pending.items() if now >= deadline]
        for token in expired:
            del self._pending[token]
        return len(expired)
