"""Owned voice lifecycle tests with fake ports; no robot or model calls."""

import asyncio

import numpy as np
import pytest

from reachy_conscience import (
    GuardedConversation,
    OwnedAudioIngress,
    OwnedVoiceSession,
    ReachyMediaAudio,
    Speech,
    Verdict,
)


class Ports:
    def __init__(self, transcript="hello"):
        self.transcript = transcript
        self.events = []
        self.capture_started = asyncio.Event()
        self.capture_release = asyncio.Event()
        self.arm_started = asyncio.Event()
        self.arm_release = asyncio.Event()
        self.wait_capture = False
        self.wait_arm = False

    async def listen_once(self, *, timeout_s):
        self.events.append("capture_start")
        self.capture_started.set()
        try:
            if self.wait_capture:
                await self.capture_release.wait()
            self.events.append("asr_final")
            return self.transcript
        finally:
            self.events.append("capture_stop")

    async def arm_playback(self):
        self.events.append("arm")
        self.arm_started.set()
        if self.wait_arm:
            await self.arm_release.wait()

    async def plan(self, _transcript):
        self.events.append("plan")
        return [Speech("reply")]

    async def synthesize(self, _text):
        self.events.append("synthesize")
        return b"audio"

    async def enqueue(self, _audio):
        self.events.append("enqueue")

    async def stop(self):
        self.events.append("stop")


def session(ports, guard_kind="approve"):
    async def guard(action):
        ports.events.append(f"guard_{action.kind}")
        return Verdict(guard_kind, "policy")

    conversation = GuardedConversation(
        planner=ports, guard=guard, synthesizer=ports, audio=ports, emergency_stop=ports
    )
    return OwnedVoiceSession(ports, conversation, ports)


@pytest.mark.asyncio
async def test_capture_final_text_then_guard_before_synthesis_and_enqueue():
    ports = Ports()
    result = await session(ports).run_once()
    assert (result.status, result.delivered) == ("complete", 1)
    assert ports.events == [
        "capture_start",
        "asr_final",
        "capture_stop",
        "arm",
        "guard_inbound",
        "plan",
        "guard_utterance",
        "synthesize",
        "enqueue",
    ]


@pytest.mark.asyncio
async def test_block_and_no_input_cannot_reach_output():
    ports = Ports()
    result = await session(ports, "block").run_once()
    assert result.status == "block"
    assert "synthesize" not in ports.events and "enqueue" not in ports.events

    ports = Ports(None)
    result = await session(ports).run_once()
    assert result.status == "no_input"
    assert "arm" not in ports.events and "plan" not in ports.events


@pytest.mark.asyncio
async def test_spoken_final_stop_bypasses_playback_guard_and_planner():
    ports = Ports("stop")
    voice = session(ports)
    assert (await voice.run_once()).status == "stopped"
    assert ports.events == ["capture_start", "asr_final", "capture_stop", "stop"]
    assert (await voice.run_once()).status == "stopped"


@pytest.mark.asyncio
async def test_external_stop_cancels_capture_without_planning():
    ports = Ports()
    ports.wait_capture = True
    voice = session(ports)
    pending = asyncio.create_task(voice.run_once())
    await ports.capture_started.wait()
    assert (await voice.stop()).status == "stopped"
    assert (await pending).status == "interrupted"
    assert "plan" not in ports.events and "arm" not in ports.events
    assert ports.events.count("capture_stop") == 1


@pytest.mark.asyncio
async def test_stop_during_playback_arm_disarms_again_and_never_dispatches():
    ports = Ports()
    ports.wait_arm = True
    voice = session(ports)
    pending = asyncio.create_task(voice.run_once())
    await ports.arm_started.wait()
    stopper = asyncio.create_task(voice.stop())
    await asyncio.sleep(0)
    assert ports.events.count("stop") == 1  # stop bypasses the stalled arm
    ports.arm_release.set()
    assert (await stopper).status == "stopped"
    assert (await pending).status == "interrupted"
    assert ports.events.count("stop") == 2
    assert "plan" not in ports.events and "enqueue" not in ports.events


@pytest.mark.asyncio
@pytest.mark.parametrize("output_verdict", ["approve", "block"])
async def test_synthetic_pcm_crosses_owned_sdk_boundary_only_after_guard(output_verdict):
    events = []

    class Media:
        audio = object()

        def __init__(self):
            self.chunks = [
                np.concatenate(
                    [np.full(320 * 12, 0.2, dtype=np.float32), np.zeros(320 * 30, dtype=np.float32)]
                ).reshape(-1, 1)
            ]

        def get_input_audio_samplerate(self):
            return 16_000

        def get_output_audio_samplerate(self):
            return 16_000

        def get_input_channels(self):
            return 1

        def get_output_channels(self):
            return 1

        def start_recording(self):
            events.append("capture_start")

        def get_audio_sample(self):
            return self.chunks.pop(0) if self.chunks else None

        def stop_recording(self):
            events.append("capture_stop")

        def start_playing(self):
            events.append("playback_arm")

        def push_audio_sample(self, _samples):
            events.append("audio_push")

        def stop_playing(self):
            events.append("audio_stop")

    class Asr:
        async def transcribe(self, _pcm):
            events.append("asr_final")
            return "hello"

    class PlannerAndVoice:
        async def plan(self, _transcript):
            events.append("plan")
            return [Speech("reply")]

        async def synthesize(self, _text):
            events.append("synthesize")
            return np.array([0.1], dtype=np.float32).tobytes()

        async def stop(self):
            await audio.halt_audio()

    async def guard(action):
        events.append(f"guard_{action.kind}")
        return Verdict(output_verdict if action.kind == "utterance" else "approve", "fixture")

    media = Media()
    audio = ReachyMediaAudio(media)
    ingress = OwnedAudioIngress(audio, Asr(), channels=1)
    ports = PlannerAndVoice()
    conversation = GuardedConversation(
        planner=ports, guard=guard, synthesizer=ports, audio=audio, emergency_stop=ports
    )
    result = await OwnedVoiceSession(ingress, conversation, audio).run_once()
    assert result.status == ("complete" if output_verdict == "approve" else "block")
    assert events[:4] == ["capture_start", "capture_stop", "asr_final", "playback_arm"]
    assert events[4:7] == ["guard_inbound", "plan", "guard_utterance"]
    assert events[7:] == (["synthesize", "audio_push"] if output_verdict == "approve" else [])
