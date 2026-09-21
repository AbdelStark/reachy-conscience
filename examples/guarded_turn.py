"""Offline ordering demo. Fake judgments and sinks; no Jev call or robot output."""

from __future__ import annotations

import asyncio

from reachy_conscience import Action, GuardedConversation, Speech, Verdict


class FixturePorts:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def plan(self, transcript: str) -> list[Speech]:
        self.events.append("plan")
        return [Speech(f"I heard: {transcript}")]

    async def synthesize(self, text: str) -> bytes:
        self.events.append("synthesize")
        return text.encode()

    async def enqueue(self, audio: bytes) -> None:
        self.events.append("enqueue")
        assert audio

    async def stop(self) -> None:
        self.events.append("stop")


async def main() -> None:
    ports = FixturePorts()

    async def fixture_guard(action: Action) -> Verdict:
        ports.events.append("guard")
        if action.kind == "utterance":
            return Verdict("block", "fixture_rule")
        return Verdict("approve", "fixture_only")

    conversation = GuardedConversation(
        planner=ports,
        guard=fixture_guard,
        synthesizer=ports,
        audio=ports,
        emergency_stop=ports,
    )
    result = await conversation.run_turn("hello")
    assert result.status == "block"
    assert ports.events == ["guard", "plan", "guard"]
    print("fixture: blocked speech never reached synthesis or audio enqueue")


if __name__ == "__main__":
    asyncio.run(main())
