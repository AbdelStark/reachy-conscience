"""Loopback fixture only: no model, cloud request, tool, or robot."""

import json
import shutil
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from reachy_conscience import (
    AsyncTypeSafeGuard,
    EspeakFfmpegSynthesizer,
    GuardedConversation,
    GuardPolicy,
    Ledger,
    Speech,
    Verdict,
)
from reachy_conscience.proposal_planner import LocalOllamaPlanner, decode_proposals


@pytest.fixture
def ollama_fixture():
    received = []
    reply = {
        "status": 200,
        "body": {"done": True, "response": '{"proposals":[{"kind":"speech","text":"hello"}]}'},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            received.append((self.path, json.loads(self.rfile.read(content_length))))
            body = json.dumps(reply["body"]).encode()
            self.send_response(reply["status"])
            if reply["status"] == 302:
                self.send_header("Location", "https://example.invalid/should-not-be-followed")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/api/generate", reply, received
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_decoder_requires_exact_unambiguous_typed_records():
    proposals = decode_proposals(
        json.dumps(
            {
                "proposals": [
                    {"kind": "speech", "text": "hello"},
                    {
                        "kind": "tool_call",
                        "name": "weather",
                        "arguments": {"city": "Paris"},
                        "summary": "lookup",
                    },
                    {"kind": "motion", "motion_class": "small_gesture", "target": {"yawDeg": 5}},
                ]
            }
        )
    )
    assert [type(value).__name__ for value in proposals] == ["Speech", "ToolCall", "Motion"]
    assert decode_proposals('{"proposals":[]}') == ()
    for raw in [
        '{"proposals":[],"proposals":[]}',
        '{"proposals":[{"kind":"speech","text":"hello","execute":true}]}',
        '{"proposals":[{"kind":"tool_call","name":"x","summary":"x","arguments":[]}]}',
        '{"proposals":[{"kind":"motion","motion_class":"dance","target":{},"text":"hi"}]}',
        '{"proposals":[{"kind":"speech","text":" "}]}',
        '{"proposals":[' + ",".join('{"kind":"speech","text":"x"}' for _ in range(5)) + "]}",
        "not-json",
    ]:
        with pytest.raises(ValueError):
            decode_proposals(raw)


def test_planner_refuses_non_loopback_and_bad_paths():
    for endpoint in [
        "http://localhost:11434/api/generate",
        "https://127.0.0.1:11434/api/generate",
        "http://192.168.1.2:11434/api/generate",
        "http://127.0.0.1:11434/api/chat",
        "http://127.0.0.1:11434/api/generate?token=x",
    ]:
        with pytest.raises(ValueError, match="loopback"):
            LocalOllamaPlanner("fixture", endpoint=endpoint)


def test_planner_describes_only_opted_in_local_note_tool():
    default = LocalOllamaPlanner("fixture")
    enabled = LocalOllamaPlanner("fixture", enable_local_notes=True)
    assert "No tools or motion are available" in default.system
    assert "append_local_note" not in default.system
    assert "append_local_note" in enabled.system
    assert "No other tools or motion are available" in enabled.system


@pytest.mark.asyncio
async def test_loopback_complete_response_and_guard_before_output(ollama_fixture):
    endpoint, reply, received = ollama_fixture
    planner = LocalOllamaPlanner("fixture", endpoint=endpoint)
    proposals = await planner.plan("say hello")
    assert proposals == (Speech("hello"),)
    path, body = received[0]
    assert path == "/api/generate"
    assert body["model"] == "fixture"
    assert body["stream"] is False
    assert body["format"]["type"] == "object"
    assert json.loads(body["prompt"]) == {"untrusted_user_text": "say hello"}

    events = []

    class Outputs:
        async def synthesize(self, _text):
            events.append("synthesize")
            return b"pcm"

        async def enqueue(self, _audio):
            events.append("enqueue")

        async def stop(self):
            events.append("stop")

    async def guard(action):
        events.append(f"guard:{action.kind}")
        return Verdict("block" if action.kind == "utterance" else "approve", "fixture")

    output = Outputs()
    app = GuardedConversation(
        planner=planner, guard=guard, synthesizer=output, audio=output, emergency_stop=output
    )
    result = await app.run_turn("say hello")
    assert result.status == "block"
    assert events == ["guard:inbound", "guard:utterance"]
    assert len(received) == 2

    reply["body"] = {"done": True, "response": '{"proposals":[{"kind":"speech","text":"okay"}]}'}
    assert await planner.plan("hi") == (Speech("okay"),)


@pytest.mark.asyncio
async def test_incomplete_or_redirected_response_fails_closed(ollama_fixture):
    endpoint, reply, _received = ollama_fixture
    planner = LocalOllamaPlanner("fixture", endpoint=endpoint)
    reply["body"] = {"done": False, "response": '{"proposals":[]}'}
    with pytest.raises(ValueError, match="incomplete"):
        await planner.plan("hello")
    reply["status"] = 302
    with pytest.raises(urllib.error.HTTPError):
        await planner.plan("hello")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not shutil.which("espeak-ng") or not shutil.which("ffmpeg"),
    reason="offline TTS binaries are not installed",
)
async def test_complete_fixture_path_guards_both_inbound_and_speech_before_pcm(ollama_fixture, tmp_path):
    endpoint, _reply, received = ollama_fixture
    events = []

    class FakeJev:
        def system_one(self, *, state, questions, model):
            assert model == "jev-latest"
            events.append(f"jev:{state['action']['kind']}")
            answers = {}
            for key, question in questions.items():
                if question["type"] == "score":
                    answers[key] = SimpleNamespace(type="score", score=1.0)
                else:
                    answers[key] = SimpleNamespace(
                        type="noul", noul=1.0 if key == "appropriate_for_audience" else 0.0
                    )
            return SimpleNamespace(answers=answers)

    class FakeAudio:
        async def enqueue(self, pcm):
            assert isinstance(pcm, bytes) and pcm
            events.append("enqueue")

        async def stop(self):
            events.append("stop")

    class RecordingTts:
        async def synthesize(self, text):
            events.append("synthesize")
            return await EspeakFfmpegSynthesizer().synthesize(text)

    audio = FakeAudio()
    ledger = Ledger(tmp_path / "verdicts.db")
    app = GuardedConversation(
        planner=LocalOllamaPlanner("fixture", endpoint=endpoint),
        guard=AsyncTypeSafeGuard(FakeJev(), GuardPolicy()),
        synthesizer=RecordingTts(),
        audio=audio,
        emergency_stop=audio,
        ledger=ledger,
    )
    result = await app.run_turn("hello")
    assert (result.status, result.delivered) == ("complete", 1)
    assert events == ["jev:inbound", "jev:utterance", "synthesize", "enqueue"]
    assert len(received) == 1
    rows = ledger.recent()
    assert [row["kind"] for row in rows] == ["utterance", "inbound"]
    assert rows[0]["summary"] == "utterance action"
    assert rows[0]["bank"] == "conscience.guard@0.1.0"
    assert json.loads(rows[0]["probabilities"])["appropriate_for_audience"] == 1.0
    assert "hello" not in ledger.export_jsonl()
    ledger.close()
