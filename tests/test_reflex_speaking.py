"""Synthetic local speaking bridge tests; no robot, TypeSafe, or live model."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reachy_conscience.reflex_speaking import ReflexSpeakingPublisher, SpeakingObservedPlayback

TOKEN = "w" * 40


def wait_for(predicate, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("speaking assertion was not observed")


def test_owned_enqueue_assertion_heartbeats_then_operator_quiet_is_one_shot():
    sent = []
    lock = threading.Lock()

    def record(session, sequence, speaking):
        with lock:
            sent.append((session, sequence, speaking))

    publisher = ReflexSpeakingPublisher("http://127.0.0.1:8048", TOKEN, send=record, interval_s=0.5)
    publisher.start()
    try:
        assert sent == []  # No assertion before an enqueue or operator command.
        publisher.may_be_speaking()
        wait_for(lambda: len(sent) >= 2)
        publisher.operator_confirmed_quiet()
        wait_for(lambda: sent[-1][2] is False)
        count = len(sent)
        time.sleep(0.6)
        assert len(sent) == count  # Quiet is not a perpetual inferred sensor reading.
        assert all(entry[0] == sent[0][0] for entry in sent)
        assert [entry[1] for entry in sent] == list(range(1, len(sent) + 1))
        assert len(sent[0][0]) >= 8
        assert sent[0][2] is True
    finally:
        publisher.close()
    publisher.close()  # Idempotent cleanup.


def test_network_failure_does_not_raise_into_owned_audio_path():
    calls = []

    def fail(_session, sequence, speaking):
        calls.append((sequence, speaking))
        raise OSError("fixture relay down")

    publisher = ReflexSpeakingPublisher("http://127.0.0.1:8048", TOKEN, send=fail, interval_s=0.5)
    publisher.start()
    try:
        publisher.may_be_speaking()
        wait_for(lambda: len(calls) >= 1)
        publisher.operator_confirmed_quiet()
        wait_for(lambda: calls[-1][1] is False)
    finally:
        publisher.close()


@pytest.mark.parametrize(
    ("url", "token"),
    [
        ("https://127.0.0.1:8048", TOKEN),
        ("http://localhost:8048", TOKEN),
        ("http://127.0.0.1:8048/other", TOKEN),
        ("http://127.0.0.1:8048/?token=secret", TOKEN),
        ("http://user:password@127.0.0.1:8048", TOKEN),
        ("http://127.0.0.1:8048", "short"),
    ],
)
def test_only_numeric_loopback_and_distinct_secret_shape(url, token):
    with pytest.raises(ValueError):
        ReflexSpeakingPublisher(url, token)


@pytest.mark.asyncio
async def test_audio_observer_reports_only_successful_owned_enqueue():
    events = []

    class Backend:
        fail = False

        async def arm_playback(self):
            events.append("arm")

        async def enqueue(self, _audio):
            events.append("enqueue")
            if self.fail:
                raise OSError("partial fixture enqueue")

        async def halt_audio(self):
            events.append("halt")

    class Publisher:
        def may_be_speaking(self):
            events.append("speaking_assertion")

    backend = Backend()
    observed = SpeakingObservedPlayback(backend, Publisher())
    await observed.arm_playback()
    await observed.enqueue(b"fixture")
    await observed.halt_audio()
    assert events == ["arm", "enqueue", "speaking_assertion", "halt"]
    backend.fail = True
    with pytest.raises(OSError):
        await observed.enqueue(b"fixture")
    assert events.count("speaking_assertion") == 1


def test_http_writer_sends_only_typed_state_to_numeric_loopback():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, dict(self.headers), json.loads(body)))
            self.send_response(202)
            self.end_headers()

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        publisher = ReflexSpeakingPublisher(f"http://127.0.0.1:{server.server_port}", TOKEN)
        publisher.start()
        try:
            publisher.may_be_speaking()
            wait_for(lambda: bool(received))
        finally:
            publisher.close()
        path, headers, body = received[0]
        assert path == "/v1/speaking"
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        assert "Origin" not in headers
        assert body == {"schema": "reflex.speaking@1", "session": body["session"], "seq": 1, "speaking": True}
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)
