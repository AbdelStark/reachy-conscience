"""Text-free Reflex hint validation and optional read-only transport tests."""

from __future__ import annotations

import json
import threading
import time

import pytest

from reachy_conscience.reflex_events import ReflexEventInbox, ReflexEventMonitor

TOKEN = "s" * 40


def frame(seq: int, stamp: int, event: dict) -> str:
    return json.dumps({"schema": "reflex.event@1", "seq": seq, "t_ms": stamp, "event": event})


def test_hint_inbox_accepts_only_fresh_monotonic_text_free_events():
    clock = [2_000.0, 10.0]
    inbox = ReflexEventInbox(wall_ms=lambda: clock[0], monotonic_s=lambda: clock[1])
    assert inbox.accept(frame(1, 1900, {"type": "attention", "person": "p1"}))
    assert inbox.latest().kind == "attention"
    assert inbox.latest().person == "p1"
    assert inbox.latest().probability is None
    assert not inbox.accept(frame(1, 1900, {"type": "yield", "p": 0.8}))
    assert inbox.accept(frame(3, 1900, {"type": "user_addressed", "person": "p2", "p": 0.83}))
    assert (inbox.latest().kind, inbox.latest().probability) == ("user_addressed", 0.83)
    clock[1] = 11.51
    assert inbox.latest() is None
    inbox.reset_connection()
    assert inbox.accept(frame(1, 1900, {"type": "interrupt", "p": 0.7}))
    assert inbox.latest().kind == "interrupt"


@pytest.mark.parametrize(
    "bad",
    [
        '{"schema":"reflex.event@1","schema":"reflex.event@1","seq":1,"t_ms":2000,"event":{"type":"attention","person":"p1"}}',
        frame(1, 2000, {"type": "attention", "person": "p0"}),
        frame(1, 2000, {"type": "yield", "p": True}),
        frame(1, 2000, {"type": "yield", "p": 0.8, "text": "private"}),
        frame(1, 2000, {"type": "speaking_started"}),
        frame(1, 2000, {"type": "attention", "person": "p1", "p": 0.5}),
        frame(1, 400, {"type": "yield", "p": 0.8}),
        frame(1, 2300, {"type": "yield", "p": 0.8}),
        frame(True, 2000, {"type": "yield", "p": 0.8}),
        "{" + '"x":"a",' * 130 + '"schema":"reflex.event@1"}',
    ],
)
def test_hint_inbox_rejects_malformed_stale_or_oversized_frames(bad):
    inbox = ReflexEventInbox(wall_ms=lambda: 2000, monotonic_s=lambda: 1)
    assert not inbox.accept(bad)
    assert inbox.latest() is None


@pytest.mark.parametrize(
    ("url", "token"),
    [
        ("ws://localhost:8048/v1/events", TOKEN),
        ("wss://127.0.0.1:8048/v1/events", TOKEN),
        ("ws://127.0.0.1:8048/other", TOKEN),
        ("ws://127.0.0.1:8048/v1/events?token=secret", TOKEN),
        ("ws://user:password@127.0.0.1:8048/v1/events", TOKEN),
        ("ws://127.0.0.1:8048/v1/events", "short"),
    ],
)
def test_monitor_requires_exact_loopback_endpoint_and_secret_shape(url, token):
    with pytest.raises(ValueError):
        ReflexEventMonitor(url, token)


def test_monitor_reads_authenticated_local_websocket_without_dispatching():
    websockets = pytest.importorskip("websockets.sync.server")
    seen_headers = []
    send_now = threading.Event()

    def handler(connection):
        seen_headers.append(connection.request.headers.get("Authorization"))
        send_now.wait(2)
        connection.send(frame(1, int(time.time() * 1000), {"type": "yield", "p": 0.81}))
        time.sleep(0.3)

    with websockets.serve(handler, "127.0.0.1", 0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        port = server.socket.getsockname()[1]
        monitor = ReflexEventMonitor(f"ws://127.0.0.1:{port}/v1/events", TOKEN)
        monitor.start()
        try:
            deadline = time.monotonic() + 2
            while not seen_headers and time.monotonic() < deadline:
                time.sleep(0.01)
            assert seen_headers == [f"Bearer {TOKEN}"]
            send_now.set()
            while monitor.latest() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert monitor.latest().kind == "yield"
            assert monitor.latest().probability == 0.81
        finally:
            monitor.close()
            server.shutdown()
            worker.join(timeout=2)
        assert monitor.latest() is None
