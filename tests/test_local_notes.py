"""A real local effectful sink behind synthetic guard and owner decisions."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from reachy_conscience import GuardedConversation, LocalNotesTool, OwnerApprovalBroker, ToolCall, Verdict


class Ports:
    def __init__(self, text: str = "Bring the charger") -> None:
        self.arguments = {"text": text}
        self.stop_count = 0

    async def plan(self, _transcript: str) -> list[ToolCall]:
        return [ToolCall("append_local_note", self.arguments, "save a local note")]

    async def synthesize(self, _text: str) -> bytes:
        raise AssertionError("no speech proposed")

    async def enqueue(self, _audio: bytes) -> None:
        raise AssertionError("no speech proposed")

    async def stop(self) -> None:
        self.stop_count += 1


async def pending(broker: OwnerApprovalBroker):
    for _ in range(30):
        request = broker.pending()
        if request is not None:
            return request
        await asyncio.sleep(0)
    raise AssertionError("no approval request")


def conversation(directory, ports, broker, *, enable=True):
    async def guard(action):
        if action.kind == "tool_call":
            expected = json.dumps(ports.arguments, sort_keys=True, separators=(",", ":"))
            assert action.tool_arguments_json == expected
            return Verdict("hold", "confirm_before")
        return Verdict("approve", "fixture")

    return GuardedConversation(
        planner=ports,
        guard=guard,
        synthesizer=ports,
        audio=ports,
        emergency_stop=ports,
        tools=LocalNotesTool(directory) if enable else None,
        effectful_tools=frozenset({"append_local_note"}) if enable else frozenset(),
        owner_approval=broker if enable else None,
        approval_timeout_s=1,
    )


@pytest.mark.asyncio
async def test_exact_approved_note_reaches_private_file_only_after_decision(tmp_path):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    ports = Ports()
    broker = OwnerApprovalBroker()
    app = conversation(state, ports, broker)
    turn = asyncio.create_task(app.run_turn("remember the charger"))
    request = await pending(broker)
    assert request.tool == "append_local_note"
    assert request.arguments_json == '{"text":"Bring the charger"}'
    assert not (state / "notes.jsonl").exists()
    ports.arguments["text"] = "Changed after review"
    assert not broker.decide(request.request_id, "0" * 64, approve=True)
    assert not (state / "notes.jsonl").exists()
    assert broker.decide(request.request_id, request.digest, approve=True)
    assert (await turn).status == "complete"
    assert (state / "notes.jsonl").read_text() == '{"text":"Bring the charger"}\n'
    assert os.stat(state / "notes.jsonl").st_mode & 0o077 == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["deny", "stop"])
async def test_denial_or_stop_never_creates_note(tmp_path, decision):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    ports = Ports()
    broker = OwnerApprovalBroker()
    app = conversation(state, ports, broker)
    turn = asyncio.create_task(app.run_turn("remember the charger"))
    request = await pending(broker)
    if decision == "deny":
        assert broker.decide(request.request_id, request.digest, approve=False)
        assert (await turn).status == "held"
    else:
        assert (await app.run_turn("stop")).status == "stopped"
        assert (await turn).status == "interrupted"
        assert not broker.decide(request.request_id, request.digest, approve=True)
    assert not (state / "notes.jsonl").exists()


@pytest.mark.asyncio
async def test_tool_rejects_other_names_shapes_and_insecure_targets(tmp_path):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    tool = LocalNotesTool(state)
    for name, arguments in (
        ("send_message", {"text": "hello"}),
        ("append_local_note", {"text": "hello", "path": "elsewhere"}),
        ("append_local_note", {"text": "hi\nthere"}),
        ("append_local_note", {"text": " "}),
        ("append_local_note", {"text": "x" * 241}),
    ):
        with pytest.raises(ValueError):
            await tool.execute(name, arguments)
    assert not (state / "notes.jsonl").exists()

    state.chmod(0o755)
    with pytest.raises(PermissionError):
        await tool.execute("append_local_note", {"text": "hello"})
    state.chmod(0o700)

    target = tmp_path / "outside"
    target.write_text("untouched")
    (state / "notes.jsonl").symlink_to(target)
    with pytest.raises(OSError):
        await tool.execute("append_local_note", {"text": "hello"})
    assert target.read_text() == "untouched"


@pytest.mark.asyncio
async def test_private_existing_file_and_size_cap_are_enforced(tmp_path):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    notes = state / "notes.jsonl"
    notes.write_text("existing\n")
    notes.chmod(0o644)
    tool = LocalNotesTool(state)
    with pytest.raises(PermissionError):
        await tool.execute("append_local_note", {"text": "hello"})
    assert notes.read_text() == "existing\n"

    notes.chmod(0o600)
    os.truncate(notes, 1_048_576)
    with pytest.raises(ValueError, match="size cap"):
        await tool.execute("append_local_note", {"text": "hello"})
    assert notes.stat().st_size == 1_048_576
