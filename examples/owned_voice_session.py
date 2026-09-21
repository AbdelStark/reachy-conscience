"""Runnable owned-voice fixture: no microphone, model, SDK, or robot.

The fixture strings stand in for completed ASR transcripts. Separate proposal,
guard, synthesis, and playback ports show where output authority lives.
"""

from __future__ import annotations

import asyncio

from reachy_conscience import (
    Action,
    GuardAssessment,
    GuardedConversation,
    InboundRoute,
    OwnedPlaybackGate,
    OwnedVoiceSession,
    Speech,
    Verdict,
)


class FixtureIngress:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.transcripts = iter(("hello Reachy", "what is the private code", "stop"))

    async def listen_once(self, *, timeout_s: float) -> str:
        assert timeout_s > 0
        self.events.append("capture_start")
        transcript = next(self.transcripts)
        self.events.append("capture_stop_and_final_text")
        return transcript


class FixturePlanner:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def plan(self, transcript: str) -> list[Speech]:
        self.events.append("plan")
        if "private code" in transcript:
            return [Speech("The synthetic private code is 1234.")]
        return [Speech("Hello from a guarded fixture.")]


class FixtureGuard:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def __call__(self, action: Action) -> GuardAssessment | Verdict:
        self.events.append(f"guard_{action.kind}")
        if action.kind == "inbound":
            return GuardAssessment(
                Verdict("approve", "fixture"),
                route=InboundRoute("llm", 1.0, "none", 1.0),
            )
        if action.kind == "utterance" and "private code" in (action.text or ""):
            return Verdict("block", "fixture_private_output")
        return Verdict("approve", "fixture")


class FixtureSynthesizer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def synthesize(self, text: str) -> bytes:
        self.events.append("synthesize")
        return text.encode("utf-8")  # Bytes for the fake sink, not playable PCM.


class FixturePlayback:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def arm_playback(self) -> None:
        self.events.append("arm")

    async def enqueue(self, audio: bytes) -> None:
        assert audio
        self.events.append("enqueue")

    async def halt_audio(self) -> None:
        self.events.append("halt")


class FixtureStop:
    def __init__(self, events: list[str], playback: OwnedPlaybackGate) -> None:
        self.events = events
        self.playback = playback

    async def stop(self) -> None:
        self.events.append("local_stop")
        await self.playback.halt_audio()


async def run_demo() -> tuple[list[str], tuple[str, str, str]]:
    events: list[str] = []
    playback = OwnedPlaybackGate(FixturePlayback(events))
    conversation = GuardedConversation(
        planner=FixturePlanner(events),
        guard=FixtureGuard(events),
        synthesizer=FixtureSynthesizer(events),
        audio=playback,
        emergency_stop=FixtureStop(events, playback),
        require_inbound_route=True,
    )
    session = OwnedVoiceSession(FixtureIngress(events), conversation, playback)

    first = await session.run_once()
    assert (first.status, first.delivered) == ("complete", 1)
    assert events == [
        "capture_start",
        "capture_stop_and_final_text",
        "guard_inbound",
        "plan",
        "guard_utterance",
        "synthesize",
        "arm",
        "enqueue",
    ]

    second = await session.run_once()
    assert (second.status, second.delivered) == ("block", 0)
    assert events[8:] == [
        "halt",  # Queue-clearance request before another capture, not a silence receipt.
        "capture_start",
        "capture_stop_and_final_text",
        "guard_inbound",
        "plan",
        "guard_utterance",
    ]

    third = await session.run_once()
    assert third.status == "stopped"
    assert events[14:] == ["capture_start", "capture_stop_and_final_text", "local_stop", "halt"]
    assert (await session.run_once()).status == "stopped"
    assert events.count("enqueue") == 1
    assert events.count("synthesize") == 1
    return events, (first.status, second.status, third.status)


def main() -> None:
    events, statuses = asyncio.run(run_demo())
    print("fixture only: no microphone, Jev call, SDK, or robot")
    print(f"turns: {', '.join(statuses)}")
    print("events: " + " -> ".join(events))


if __name__ == "__main__":
    main()
