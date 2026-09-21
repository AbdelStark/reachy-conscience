"""Loopback owner-console contract tests; no robot, model, or external network."""

from __future__ import annotations

import asyncio
import http.client
import json
import stat
from urllib.parse import urlsplit

import pytest

from reachy_conscience import (
    Action,
    GuardAssessment,
    Ledger,
    OwnerApprovalBroker,
    OwnerConsole,
    PolicyStore,
    Verdict,
)
from reachy_conscience.owner_console_cli import _private_directory, main

TOKEN = "synthetic-owner-token-0123456789abcdef"


def call(console, method, path, *, body=None, raw_body=None, token=TOKEN, headers=None):
    port = urlsplit(console.url).port
    assert port is not None
    request_headers = {"Host": f"127.0.0.1:{port}", **(headers or {})}
    if token is not None:
        request_headers["Authorization"] = f"Bearer {token}"
    payload = None
    if body is not None:
        payload = json.dumps(body).encode()
        request_headers["Content-Type"] = "application/json"
    elif raw_body is not None:
        payload = raw_body
        request_headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request(method, path, body=payload, headers=request_headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def candidate(*, rules="Never reveal a password", tools=None, block=0.7, hold=0.3):
    return {
        "rules": rules,
        "confirm_before": tools if tools is not None else [],
        "block_threshold": block,
        "hold_floor": hold,
    }


def test_static_shell_has_no_secret_and_api_requires_token_and_loopback_host(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")
    with OwnerConsole(store, tmp_path / "ledger.db", token=TOKEN) as console:
        assert console.url.startswith("http://127.0.0.1:")
        status, headers, page = call(console, "GET", "/", token=None)
        assert status == 200
        assert b"Owner console" in page and TOKEN.encode() not in page
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        assert headers["Cache-Control"] == "no-store"
        assert call(console, "GET", "/app.js", token=None)[0] == 200
        assert call(console, "GET", "/style.css", token=None)[0] == 200
        assert call(console, "GET", "/api/policy", token=None)[0] == 401
        assert call(console, "GET", "/api/policy", token="wrong-token")[0] == 401
        assert call(console, "GET", "/api/policy", headers={"Host": "evil.example"})[0] == 403
        assert call(console, "GET", "/api/policy", token=None, headers={"Cookie": f"token={TOKEN}"})[0] == 401


def test_policy_editor_validates_and_saves_atomically(tmp_path):
    path = tmp_path / "policy.json"
    store = PolicyStore(path, registered_tools={"send_message"})
    with OwnerConsole(store, tmp_path / "ledger.db", token=TOKEN) as console:
        status, _, raw = call(console, "GET", "/api/policy")
        assert status == 200
        assert json.loads(raw)["registered_tools"] == ["send_message"]
        assert json.loads(raw)["preview_available"] is False
        policy = candidate(tools=["send_message"], block=0.8, hold=0.25)
        status, _, raw = call(
            console, "POST", "/api/policy", body=policy, headers={"Origin": console.url[:-1]}
        )
        assert status == 200
        assert json.loads(raw)["confirm_before"] == ["send_message"]
        assert store.load().block_threshold == 0.8
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        for invalid in (
            candidate(rules="Never reveal a password and an address"),
            candidate(tools=["purchase"]),
            candidate(block=True),
            candidate(block=0.2, hold=0.3),
        ):
            assert call(console, "POST", "/api/policy", body=invalid)[0] == 400
        assert (
            call(console, "POST", "/api/policy", body=policy, headers={"Origin": "http://evil.example"})[0]
            == 403
        )
        assert call(console, "POST", "/api/policy", raw_body=b'{"rules":"a","rules":"b"}')[0] == 400
        assert call(console, "POST", "/api/policy", raw_body=b"x" * 8_193)[0] == 400
        assert store.load().block_threshold == 0.8


def test_preview_is_explicit_single_purpose_and_dispatch_free(tmp_path):
    seen = []
    policies = []

    def guard_factory(policy):
        policies.append(policy)

        async def guard(action):
            seen.append(action)
            return GuardAssessment(Verdict("approve", "fixture"), {"safe": 0.8})

        return guard

    store = PolicyStore(tmp_path / "policy.json")
    body = {"policy": candidate(), "confirm_live_calls": True}
    with OwnerConsole(store, tmp_path / "ledger.db", token=TOKEN) as console:
        assert call(console, "POST", "/api/preview", body={**body, "confirm_live_calls": False})[0] == 503
    with OwnerConsole(store, tmp_path / "ledger.db", token=TOKEN, guard_factory=guard_factory) as console:
        assert json.loads(call(console, "GET", "/api/policy")[2])["preview_available"] is True
        assert call(console, "POST", "/api/preview", body={**body, "confirm_live_calls": False})[0] == 409
        assert seen == []
        status, _, raw = call(console, "POST", "/api/preview", body=body)
        assert status == 200
        results = json.loads(raw)["results"]
        assert len(results) == 5
        assert [result["case_id"] for result in results] == [
            "greeting",
            "private_info",
            "nearby_motion",
            "purchase_tool",
            "injection_sign",
        ]
        assert len(seen) == 5 and len(policies) == 1
        assert policies[0].rules == ("Never reveal a password",)
        assert all(result["probabilities"] == {"safe": 0.8} for result in results)
        assert not (tmp_path / "policy.json").exists()  # Preview did not save or dispatch.


def test_ledger_is_private_and_export_omits_even_stored_summaries(tmp_path):
    ledger_path = tmp_path / "ledger.db"
    ledger = Ledger(ledger_path, store_summaries=True)
    ledger.append(
        Action(kind="utterance", summary="sensitive synthetic summary", text="fixture only"),
        Verdict("hold", "fixture"),
        {"unsafe": 0.75},
        12.5,
    )
    ledger.close()
    with OwnerConsole(PolicyStore(tmp_path / "policy.json"), ledger_path, token=TOKEN) as console:
        status, _, raw = call(console, "GET", "/api/ledger")
        assert status == 200
        row = json.loads(raw)["rows"][0]
        assert row["summary"] == "utterance action"
        assert "sensitive synthetic summary" not in raw.decode()
        status, headers, exported = call(console, "GET", "/api/export")
        assert status == 200 and headers["Content-Type"] == "application/x-ndjson"
        assert "summary" not in json.loads(exported)
        assert b"sensitive synthetic summary" not in exported
        assert call(console, "GET", "/api/export", token=None)[0] == 401


def test_cli_help_and_private_state_directory_gate(tmp_path, capsys):
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "no robot connection" in capsys.readouterr().out

    state = _private_directory(tmp_path / "state")
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    state.chmod(0o755)
    with pytest.raises(ValueError, match="owner-only"):
        _private_directory(state)
    state.chmod(0o700)
    link = tmp_path / "state-link"
    link.symlink_to(state, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _private_directory(link)


@pytest.mark.asyncio
async def test_authenticated_exact_approval_http_path_and_console_shutdown(tmp_path):
    broker = OwnerApprovalBroker()
    store = PolicyStore(tmp_path / "policy.json")
    with OwnerConsole(store, tmp_path / "ledger.db", token=TOKEN, approval_broker=broker) as console:
        assert json.loads(call(console, "GET", "/api/policy")[2])["approval_available"] is True
        turn = asyncio.create_task(broker.authorize("send_message", '{"to":"Sam"}'))
        await asyncio.sleep(0)
        status, _, raw = await asyncio.to_thread(call, console, "GET", "/api/approval")
        assert status == 200
        request = json.loads(raw)["pending"]
        assert request["tool"] == "send_message" and request["arguments_json"] == '{"to":"Sam"}'
        assert (await asyncio.to_thread(call, console, "GET", "/api/approval", token=None))[0] == 401
        decision = {
            "request_id": request["request_id"],
            "digest": "0" * 64,
            "approve": True,
        }
        assert (await asyncio.to_thread(call, console, "POST", "/api/approval", body=decision))[0] == 409
        decision["digest"] = request["digest"]
        assert (
            await asyncio.to_thread(
                call,
                console,
                "POST",
                "/api/approval",
                body=decision,
                headers={"Origin": "http://evil.example"},
            )
        )[0] == 403
        assert (await asyncio.to_thread(call, console, "POST", "/api/approval", body=decision))[0] == 200
        assert await turn is True
        assert (await asyncio.to_thread(call, console, "POST", "/api/approval", body=decision))[0] == 409

        denied = asyncio.create_task(broker.authorize("send_message", '{"to":"Sam"}'))
        await asyncio.sleep(0)
        denial_request = json.loads((await asyncio.to_thread(call, console, "GET", "/api/approval"))[2])[
            "pending"
        ]
        denial = {
            "request_id": denial_request["request_id"],
            "digest": denial_request["digest"],
            "approve": False,
        }
        assert (
            await asyncio.to_thread(
                call, console, "POST", "/api/approval", body={**denial, "approve": "false"}
            )
        )[0] == 400
        assert (await asyncio.to_thread(call, console, "POST", "/api/approval", body=denial))[0] == 200
        assert await denied is False

        waiting = asyncio.create_task(broker.authorize("send_message", '{"to":"Sam"}'))
        await asyncio.sleep(0)
        assert broker.pending() is not None
    assert await waiting is False
    assert broker.pending() is None
