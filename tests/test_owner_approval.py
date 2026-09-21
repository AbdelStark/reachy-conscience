"""Exact owner approval and hard-stop races; all tools are in-memory spies."""

from __future__ import annotations

import asyncio

import pytest

from reachy_conscience import GuardedConversation, OwnerApprovalBroker, ToolCall, Verdict


async def pending(broker: OwnerApprovalBroker):
    for _ in range(20):
        request = broker.pending()
        if request is not None:
            return request
        await asyncio.sleep(0)
    raise AssertionError("approval request did not appear")


@pytest.mark.asyncio
async def test_exact_request_is_one_shot_and_wrong_digest_cannot_approve():
    broker = OwnerApprovalBroker()
    task = asyncio.create_task(broker.authorize("send_message", '{"text":"Running late","to":"Sam"}'))
    request = await pending(broker)
    assert request.tool == "send_message"
    assert request.arguments_json == '{"text":"Running late","to":"Sam"}'
    assert len(request.digest) == 64 and request.remaining_s > 0
    assert not broker.decide(request.request_id, "0" * 64, approve=True)
    assert not broker.decide("stale-id", request.digest, approve=True)
    assert broker.pending() is not None
    assert broker.decide(request.request_id, request.digest, approve=True)
    assert await task is True
    assert broker.pending() is None
    assert not broker.decide(request.request_id, request.digest, approve=True)


@pytest.mark.asyncio
async def test_denial_timeout_cancellation_and_second_pending_fail_closed():
    broker = OwnerApprovalBroker(timeout_s=0.02)
    first = asyncio.create_task(broker.authorize("send_message", '{"to":"Sam"}'))
    request = await pending(broker)
    with pytest.raises(RuntimeError, match="another owner approval"):
        await broker.authorize("purchase", '{"item":"demo"}')
    assert broker.decide(request.request_id, request.digest, approve=False)
    assert await first is False

    timed_out = asyncio.create_task(broker.authorize("send_message", '{"to":"Sam"}'))
    old = await pending(broker)
    assert await timed_out is False
    assert broker.pending() is None
    assert not broker.decide(old.request_id, old.digest, approve=True)

    cancelled = asyncio.create_task(broker.authorize("send_message", '{"to":"Sam"}'))
    old = await pending(broker)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert broker.pending() is None
    assert not broker.decide(old.request_id, old.digest, approve=True)


@pytest.mark.asyncio
async def test_rejects_noncanonical_or_oversized_arguments_before_pending():
    broker = OwnerApprovalBroker()
    for raw in ('{"b":2,"a":1}', '{"a":NaN}', "[1]", '{"a":1,"a":2}', '{"a":"' + "x" * 4096 + '"}'):
        with pytest.raises(ValueError):
            await broker.authorize("send_message", raw)
    with pytest.raises(ValueError, match="tool name"):
        await broker.authorize("send-message", "{}")
    assert broker.pending() is None


class Ports:
    def __init__(self):
        self.events = []

    async def plan(self, _transcript):
        return [ToolCall("send_message", {"to": "Sam", "text": "Running late"}, "send message")]

    async def synthesize(self, _text):
        raise AssertionError("no speech proposed")

    async def enqueue(self, _audio):
        raise AssertionError("no speech proposed")

    async def execute(self, name, arguments):
        self.events.append(("execute", name, arguments))

    async def stop(self):
        self.events.append(("stop",))


def pipeline(ports, broker):
    async def guard(_action):
        return Verdict("approve", "fixture")

    return GuardedConversation(
        planner=ports,
        guard=guard,
        synthesizer=ports,
        audio=ports,
        tools=ports,
        emergency_stop=ports,
        effectful_tools=frozenset({"send_message"}),
        owner_approval=broker,
        approval_timeout_s=1,
    )


@pytest.mark.asyncio
async def test_pipeline_dispatches_only_after_exact_owner_decision():
    ports = Ports()
    broker = OwnerApprovalBroker()
    app = pipeline(ports, broker)
    turn = asyncio.create_task(app.run_turn("please message Sam"))
    request = await pending(broker)
    assert ports.events == []
    assert broker.decide(request.request_id, request.digest, approve=True)
    result = await turn
    assert result.status == "complete"
    assert ports.events == [("execute", "send_message", {"text": "Running late", "to": "Sam"})]


@pytest.mark.asyncio
async def test_hard_stop_cancels_visible_approval_and_prevents_late_dispatch():
    ports = Ports()
    broker = OwnerApprovalBroker()
    app = pipeline(ports, broker)
    turn = asyncio.create_task(app.run_turn("please message Sam"))
    request = await pending(broker)
    assert (await app.run_turn("stop")).status == "stopped"
    assert (await turn).status == "interrupted"
    assert broker.pending() is None
    assert not broker.decide(request.request_id, request.digest, approve=True)
    assert ports.events == [("stop",)]
