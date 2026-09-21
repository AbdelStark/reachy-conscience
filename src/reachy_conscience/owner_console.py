"""Authenticated loopback owner console for policy, preview, and ledger.

This control plane has no planner, robot, tool, motion, or audio output handle.
An optional approval broker can resolve an exact pending tool request; it does
not execute the tool and is absent from the standalone console command.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from .guard import GuardPolicy
from .ledger import Ledger
from .owner_approval import OwnerApprovalBroker
from .pipeline import Guard
from .policy_store import PolicyStore, dry_run_policy, policy_from_lines
from .redteam import run_redteam

MAX_BODY_BYTES = 8_192
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _policy_payload(
    policy: GuardPolicy,
    registered_tools: frozenset[str],
    *,
    preview_available: bool,
    approval_available: bool,
) -> dict[str, Any]:
    return {
        "rules": "\n".join(policy.rules),
        "confirm_before": sorted(policy.confirm_before),
        "registered_tools": sorted(registered_tools),
        "block_threshold": policy.block_threshold,
        "hold_floor": policy.hold_floor,
        "preview_available": preview_available,
        "approval_available": approval_available,
    }


def _candidate(value: Any, store: PolicyStore) -> GuardPolicy:
    if not isinstance(value, dict) or set(value) != {
        "rules",
        "confirm_before",
        "block_threshold",
        "hold_floor",
    }:
        raise ValueError("invalid policy fields")
    if not isinstance(value["confirm_before"], list):
        raise ValueError("confirm-before must be a list")
    base = policy_from_lines(value["rules"], confirm_before=value["confirm_before"])
    if not base.confirm_before <= store.registered_tools:
        raise ValueError("confirm-before tool is not registered")
    block, hold = value["block_threshold"], value["hold_floor"]
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (block, hold)):
        raise ValueError("invalid policy thresholds")
    return GuardPolicy(
        rules=base.rules,
        confirm_before=base.confirm_before,
        block_threshold=float(block),
        hold_floor=float(hold),
    )


class OwnerConsole:
    """Serve one owner console on 127.0.0.1; disabled until explicitly started.

    The bearer token is deliberately kept only in memory by the browser UI.
    A real guard factory is optional and only called by an owner-confirmed
    five-case preview or 20-case synthetic red-team run. The caller owns the
    token and must protect terminal output, local processes, the policy
    directory, and the ledger database.
    """

    def __init__(
        self,
        policy_store: PolicyStore,
        ledger_path: str | Path,
        *,
        guard_factory: Callable[[GuardPolicy], Guard] | None = None,
        approval_broker: OwnerApprovalBroker | None = None,
        token: str | None = None,
        port: int = 0,
    ) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65_535:
            raise ValueError("invalid owner-console port")
        if token is None:
            token = secrets.token_urlsafe(32)
        if not isinstance(token, str) or len(token) < 32 or not token.isascii():
            raise ValueError("owner token must be at least 32 ASCII characters")
        self.token = token
        self.policy_store = policy_store
        self.ledger_path = Path(ledger_path)
        self.guard_factory = guard_factory
        self.approval_broker = approval_broker
        self._preview_lock = threading.Lock()
        self._policy_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._server.daemon_threads = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("owner console already started")
        self._thread = threading.Thread(target=self._server.serve_forever, name="conscience-owner-console")
        self._thread.start()

    def close(self) -> None:
        if self.approval_broker is not None:
            self.approval_broker.cancel_all()
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> OwnerConsole:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        console = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "ConscienceOwnerConsole/0.1"
            sys_version = ""

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(5)

            def log_message(self, _format: str, *_args: object) -> None:
                # Do not log requests or bearer credentials.
                return

            def _send(self, status: int, payload: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(payload)

            def _json(self, status: int, value: Any) -> None:
                payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
                self._send(status, payload, "application/json")

            def _error(self, status: int, message: str) -> None:
                self._json(status, {"error": message})

            def _host_ok(self) -> bool:
                return self.headers.get("Host") == f"127.0.0.1:{console._server.server_port}"

            def _authenticated(self) -> bool:
                supplied = self.headers.get("Authorization", "")
                expected = f"Bearer {console.token}"
                return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))

            def _body(self) -> Any:
                if self.headers.get("Transfer-Encoding") is not None:
                    raise ValueError("chunked requests are not supported")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("expected application/json")
                raw_length = self.headers.get("Content-Length", "")
                if not raw_length.isascii() or not raw_length.isdecimal():
                    raise ValueError("invalid content length")
                length = int(raw_length)
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("request body exceeds limit")
                raw = self.rfile.read(length)
                try:
                    return json.loads(raw, object_pairs_hook=_unique_object)
                except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
                    raise ValueError("invalid JSON") from exc

            def do_GET(self) -> None:
                if not self._host_ok():
                    self._error(403, "invalid host")
                    return
                if self.path in _ASSETS:
                    name, content_type = _ASSETS[self.path]
                    payload = files("reachy_conscience").joinpath("assets", name).read_bytes()
                    self._send(200, payload, content_type)
                    return
                if not self._authenticated():
                    self._error(401, "owner token required")
                    return
                try:
                    if self.path == "/api/policy":
                        with console._policy_lock:
                            policy = console.policy_store.load()
                        self._json(
                            200,
                            _policy_payload(
                                policy,
                                console.policy_store.registered_tools,
                                preview_available=console.guard_factory is not None,
                                approval_available=console.approval_broker is not None,
                            ),
                        )
                    elif self.path == "/api/approval":
                        if console.approval_broker is None:
                            self._error(503, "owner approval is not configured")
                            return
                        pending = console.approval_broker.pending()
                        self._json(
                            200,
                            {
                                "pending": None
                                if pending is None
                                else {
                                    "request_id": pending.request_id,
                                    "tool": pending.tool,
                                    "arguments_json": pending.arguments_json,
                                    "digest": pending.digest,
                                    "remaining_s": pending.remaining_s,
                                }
                            },
                        )
                    elif self.path in {"/api/ledger", "/api/export"}:
                        ledger = Ledger(console.ledger_path)
                        try:
                            if self.path == "/api/ledger":
                                self._json(200, {"rows": ledger.recent(200)})
                            else:
                                self._send(200, ledger.export_jsonl().encode(), "application/x-ndjson")
                        finally:
                            ledger.close()
                    else:
                        self._error(404, "not found")
                except Exception:
                    self._error(503, "local data unavailable")

            def do_POST(self) -> None:
                if not self._host_ok():
                    self._error(403, "invalid host")
                    return
                if not self._authenticated():
                    self._error(401, "owner token required")
                    return
                origin = self.headers.get("Origin")
                if origin is not None and origin != console.url.rstrip("/"):
                    self._error(403, "invalid origin")
                    return
                if self.path not in {"/api/policy", "/api/preview", "/api/redteam", "/api/approval"}:
                    self._error(404, "not found")
                    return
                try:
                    body = self._body()
                    if self.path == "/api/policy":
                        policy = _candidate(body, console.policy_store)
                        with console._policy_lock:
                            console.policy_store.save(policy)
                        self._json(
                            200,
                            _policy_payload(
                                policy,
                                console.policy_store.registered_tools,
                                preview_available=console.guard_factory is not None,
                                approval_available=console.approval_broker is not None,
                            ),
                        )
                        return
                    if self.path == "/api/approval":
                        if console.approval_broker is None:
                            self._error(503, "owner approval is not configured")
                            return
                        if not isinstance(body, dict) or set(body) != {"request_id", "digest", "approve"}:
                            raise ValueError("invalid owner decision fields")
                        if not isinstance(body["approve"], bool):
                            raise ValueError("owner decision must be a boolean")
                        accepted = console.approval_broker.decide(
                            body["request_id"], body["digest"], approve=body["approve"]
                        )
                        if not accepted:
                            self._error(409, "approval request is stale or mismatched")
                            return
                        self._json(200, {"accepted": True})
                        return
                    if console.guard_factory is None:
                        self._error(503, "live preview is not configured")
                        return
                    if not isinstance(body, dict) or set(body) != {"policy", "confirm_live_calls"}:
                        raise ValueError("invalid guard-run fields")
                    if body["confirm_live_calls"] is not True:
                        self._error(409, "confirm guard calls before running synthetic cases")
                        return
                    policy = _candidate(body["policy"], console.policy_store)
                    if not console._preview_lock.acquire(blocking=False):
                        self._error(409, "preview already running")
                        return
                    try:
                        guard = console.guard_factory(policy)
                        if self.path == "/api/redteam":
                            report = asyncio.run(run_redteam(guard))
                        else:
                            results = asyncio.run(dry_run_policy(guard))
                    finally:
                        console._preview_lock.release()
                    if self.path == "/api/redteam":
                        self._json(
                            200,
                            {
                                "schema": "conscience.synthetic-redteam-report@1",
                                "synthetic": True,
                                "counts": report.counts,
                                "p95_latency_ms": report.p95_latency_ms,
                                "results": [
                                    {
                                        "case_id": item.case_id,
                                        "channel": item.channel,
                                        "intent": item.intent,
                                        "verdict": item.verdict.kind,
                                        "reason": item.verdict.reason,
                                        "latency_ms": item.latency_ms,
                                    }
                                    for item in report.results
                                ],
                            },
                        )
                        return
                    self._json(
                        200,
                        {
                            "results": [
                                {
                                    "case_id": item.case_id,
                                    "verdict": item.verdict.kind,
                                    "reason": item.verdict.reason,
                                    "probabilities": item.probabilities,
                                    "latency_ms": item.latency_ms,
                                }
                                for item in results
                            ]
                        },
                    )
                except (TypeError, ValueError) as exc:
                    self._error(400, str(exc))
                except Exception:
                    self._error(503, "local operation unavailable")

        return Handler
