"""Owned voice lifecycle tests with fake ports; no robot or model calls."""

import asyncio

import numpy as np
import pytest

from reachy_conscience import (
    GuardAssessment,
    GuardedConversation,
    InboundRoute,
    OwnedAudioIngress,
    OwnedPlaybackGate,
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
        self.fail_arm = False
        self.wait_enqueue = False
        self.fail_enqueue = False
        self.wait_halt = False
        self.fail_halt = False
        self.halt_started = asyncio.Event()
        self.halt_release = asyncio.Event()
        self.enqueue_started = asyncio.Event()
        self.enqueue_release = asyncio.Event()
        self.gate = None

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
        if self.fail_arm:
            raise OSError("partially armed")

    async def halt_audio(self):
        self.events.append("halt")
        self.halt_started.set()
        if self.wait_halt:
            await self.halt_release.wait()
        if self.fail_halt:
            raise OSError("queue flush unavailable")

    async def plan(self, _transcript):
        self.events.append("plan")
        return [Speech("reply")]

    async def synthesize(self, _text):
        self.events.append("synthesize")
        return b"audio"

    async def enqueue(self, _audio):
        self.events.append("enqueue")
        self.enqueue_started.set()
        if self.wait_enqueue:
            await self.enqueue_release.wait()
        if self.fail_enqueue:
            raise OSError("partial enqueue")

    async def stop(self):
        self.events.append("stop")
        await self.gate.halt_audio()


def session(ports, guard_kind="approve"):
    async def guard(action):
        ports.events.append(f"guard_{action.kind}")
        return Verdict(guard_kind, "policy")

    ports.gate = OwnedPlaybackGate(ports)
    conversation = GuardedConversation(
        planner=ports, guard=guard, synthesizer=ports, audio=ports.gate, emergency_stop=ports
    )
    return OwnedVoiceSession(ports, conversation, ports.gate)


def test_session_rejects_a_conversation_that_bypasses_its_playback_gate():
    ports = Ports()
    gate = OwnedPlaybackGate(ports)

    async def guard(_action):
        return Verdict("approve", "fixture")

    conversation = GuardedConversation(
        planner=ports, guard=guard, synthesizer=ports, audio=ports, emergency_stop=ports
    )
    with pytest.raises(ValueError, match="owned playback gate"):
        OwnedVoiceSession(ports, conversation, gate)


@pytest.mark.asyncio
async def test_capture_final_text_then_guard_before_synthesis_and_enqueue():
    ports = Ports()
    result = await session(ports).run_once()
    assert (result.status, result.delivered) == ("complete", 1)
    assert ports.events == [
        "capture_start",
        "asr_final",
        "capture_stop",
        "guard_inbound",
        "plan",
        "guard_utterance",
        "synthesize",
        "arm",
        "enqueue",
    ]


@pytest.mark.asyncio
async def test_block_and_no_input_cannot_reach_output():
    ports = Ports()
    result = await session(ports, "block").run_once()
    assert result.status == "block"
    assert all(event not in ports.events for event in ("arm", "synthesize", "enqueue"))

    ports = Ports(None)
    result = await session(ports).run_once()
    assert result.status == "no_input"
    assert "arm" not in ports.events and "plan" not in ports.events


@pytest.mark.asyncio
async def test_ignored_inbound_turn_never_arms_the_speaker():
    ports = Ports("background conversation")
    ports.gate = OwnedPlaybackGate(ports)

    async def guard(action):
        ports.events.append(f"guard_{action.kind}")
        return GuardAssessment(Verdict("approve", "fixture"), route=InboundRoute("ignore", 0.9, "none", 0.9))

    conversation = GuardedConversation(
        planner=ports,
        guard=guard,
        synthesizer=ports,
        audio=ports.gate,
        emergency_stop=ports,
        require_inbound_route=True,
    )
    result = await OwnedVoiceSession(ports, conversation, ports.gate).run_once()
    assert result.status == "ignored"
    assert ports.events == ["capture_start", "asr_final", "capture_stop", "guard_inbound"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["arm", "enqueue"])
async def test_partial_speaker_failure_is_halted_before_next_capture(failure):
    ports = Ports()
    setattr(ports, f"fail_{failure}", True)
    voice = session(ports)
    result = await voice.run_once()
    assert (result.status, result.reason) == ("output_error", "audio_sink_failed")
    assert ports.events[-1] == "halt"
    assert voice.playback.armed is False
    assert (await voice.run_once()).status == "output_error"
    assert ports.events.count("capture_start") == 2


@pytest.mark.asyncio
async def test_next_capture_waits_for_previous_playback_queue_flush():
    ports = Ports()
    voice = session(ports)
    assert (await voice.run_once()).status == "complete"
    first_events = list(ports.events)
    ports.wait_halt = True
    second = asyncio.create_task(voice.run_once())
    await ports.halt_started.wait()
    assert ports.events == [*first_events, "halt"]
    ports.halt_release.set()
    assert (await second).status == "complete"
    assert ports.events[len(first_events) : len(first_events) + 2] == ["halt", "capture_start"]


@pytest.mark.asyncio
async def test_stop_during_next_turn_flush_never_reopens_microphone():
    ports = Ports()
    voice = session(ports)
    assert (await voice.run_once()).status == "complete"
    ports.wait_halt = True
    second = asyncio.create_task(voice.run_once())
    await ports.halt_started.wait()
    capture_count = ports.events.count("capture_start")
    stopper = asyncio.create_task(voice.stop())
    await asyncio.sleep(0)
    assert "stop" in ports.events
    ports.halt_release.set()
    assert (await stopper).status == "stopped"
    assert (await second).status == "interrupted"
    assert ports.events.count("capture_start") == capture_count


@pytest.mark.asyncio
async def test_failed_previous_playback_flush_holds_without_capture():
    ports = Ports()
    voice = session(ports)
    assert (await voice.run_once()).status == "complete"
    ports.fail_halt = True
    capture_count = ports.events.count("capture_start")
    result = await voice.run_once()
    assert (result.status, result.reason) == ("held", "playback_unavailable")
    assert voice.playback.needs_flush is True
    assert ports.events.count("capture_start") == capture_count


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["arm", "enqueue"])
async def test_failed_cleanup_after_partial_output_blocks_new_capture_until_flush_succeeds(failure):
    ports = Ports()
    setattr(ports, f"fail_{failure}", True)
    ports.fail_halt = True
    voice = session(ports)
    assert (await voice.run_once()).status == "output_error"
    assert voice.playback.armed is False
    assert voice.playback.needs_flush is True
    capture_count = ports.events.count("capture_start")
    result = await voice.run_once()
    assert (result.status, result.reason) == ("held", "playback_unavailable")
    assert ports.events.count("capture_start") == capture_count
    ports.fail_halt = False
    setattr(ports, f"fail_{failure}", False)
    assert (await voice.run_once()).status == "complete"
    assert ports.events.count("capture_start") == capture_count + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript", ["stop", "Reachy, please stop."])
async def test_spoken_final_stop_bypasses_playback_guard_and_planner(transcript):
    ports = Ports(transcript)
    voice = session(ports)
    assert (await voice.run_once()).status == "stopped"
    assert ports.events == ["capture_start", "asr_final", "capture_stop", "stop", "halt"]
    assert (await voice.run_once()).status == "stopped"


@pytest.mark.asyncio
async def test_model_routed_stop_is_terminal_for_owned_voice_session():
    ports = Ports("I want the robot to stop")

    async def guard(action):
        ports.events.append(f"guard_{action.kind}")
        return GuardAssessment(
            Verdict("approve", "fixture"), route=InboundRoute("fast_path", 0.9, "stop", 0.9)
        )

    ports.gate = OwnedPlaybackGate(ports)
    conversation = GuardedConversation(
        planner=ports,
        guard=guard,
        synthesizer=ports,
        audio=ports.gate,
        emergency_stop=ports,
        require_inbound_route=True,
    )
    voice = OwnedVoiceSession(ports, conversation, ports.gate)
    assert (await voice.run_once()).status == "stopped"
    assert ports.events == ["capture_start", "asr_final", "capture_stop", "guard_inbound", "stop", "halt"]
    assert (await voice.run_once()).status == "stopped"
    assert ports.events.count("capture_start") == 1
    with pytest.raises(RuntimeError, match="stopped"):
        await ports.gate.enqueue(b"audio")


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
    assert ports.events.count("stop") == 1
    assert ports.events.count("halt") == 2
    assert "enqueue" not in ports.events


@pytest.mark.asyncio
async def test_stop_racing_with_audio_push_rehalts_after_the_push():
    ports = Ports()
    ports.wait_enqueue = True
    voice = session(ports)
    pending = asyncio.create_task(voice.run_once())
    await ports.enqueue_started.wait()
    stopper = asyncio.create_task(voice.stop())
    await asyncio.sleep(0)
    assert ports.events[-2:] == ["stop", "halt"]
    ports.enqueue_release.set()
    assert (await pending).status == "interrupted"
    assert (await stopper).status == "stopped"
    assert ports.events[-1] == "halt"
    assert ports.events.count("halt") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("output_verdict", ["approve", "block"])
async def test_synthetic_pcm_crosses_owned_sdk_boundary_only_after_guard(output_verdict):
    events = []

    class Media:
        audio = None

        def __init__(self):
            self.audio = self
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

        def clear_player(self):
            events.append("audio_flush")

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
    gate = OwnedPlaybackGate(audio)
    conversation = GuardedConversation(
        planner=ports, guard=guard, synthesizer=ports, audio=gate, emergency_stop=ports
    )
    result = await OwnedVoiceSession(ingress, conversation, gate).run_once()
    assert result.status == ("complete" if output_verdict == "approve" else "block")
    assert events[:3] == ["capture_start", "capture_stop", "asr_final"]
    assert events[3:6] == ["guard_inbound", "plan", "guard_utterance"]
    assert events[6:] == (["synthesize", "playback_arm", "audio_push"] if output_verdict == "approve" else [])
