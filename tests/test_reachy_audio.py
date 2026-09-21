"""Reachy 1.10 media contract tests with a fake backend, never real hardware."""

import numpy as np
import pytest

from reachy_conscience import GuardedConversation, OwnedPlaybackGate, ReachyMediaAudio, Speech, Verdict


class FakeMedia:
    def __init__(self):
        self.audio = self
        self.output_rate = 16_000
        self.input_rate = 16_000
        self.output_channels = 2
        self.input_channels = 2
        self.calls = []
        self.next_chunk = None

    def get_output_audio_samplerate(self):
        return self.output_rate

    def get_output_channels(self):
        return self.output_channels

    def start_playing(self):
        self.calls.append("start_playing")

    def push_audio_sample(self, samples):
        self.calls.append(("push", samples.copy()))

    def stop_playing(self):
        self.calls.append("stop_playing")

    def clear_player(self):
        self.calls.append("clear_player")

    def get_input_audio_samplerate(self):
        return self.input_rate

    def get_input_channels(self):
        return self.input_channels

    def start_recording(self):
        self.calls.append("start_recording")

    def get_audio_sample(self):
        return self.next_chunk

    def stop_recording(self):
        self.calls.append("stop_recording")


@pytest.mark.asyncio
async def test_pcm_output_is_bounded_and_halt_requires_explicit_rearm():
    media = FakeMedia()
    output = ReachyMediaAudio(media)
    pcm = np.array([0.1, -0.1], dtype=np.float32).tobytes()
    with pytest.raises(RuntimeError, match="not armed"):
        await output.enqueue(pcm)
    await output.arm_playback()
    await output.enqueue(pcm)
    assert media.calls[0] == "start_playing"
    assert media.calls[1][0] == "push"
    assert media.calls[1][1].shape == (2,)
    await output.halt_audio()
    assert media.calls[-2:] == ["clear_player", "stop_playing"]
    with pytest.raises(RuntimeError, match="not armed"):
        await output.enqueue(pcm)
    await output.arm_playback()
    await output.enqueue(pcm)
    assert len([call for call in media.calls if isinstance(call, tuple)]) == 2


@pytest.mark.asyncio
async def test_bad_audio_and_missing_backend_fail_before_push():
    media = FakeMedia()
    output = ReachyMediaAudio(media)
    await output.arm_playback()
    for bad in (
        b"",
        b"abc",
        np.array([np.nan], dtype=np.float32).tobytes(),
        np.array([1.1], dtype=np.float32).tobytes(),
    ):
        with pytest.raises(ValueError):
            await output.enqueue(bad)
    assert all(not isinstance(call, tuple) for call in media.calls)
    media.audio = None
    with pytest.raises(RuntimeError, match="unavailable"):
        await output.enqueue(np.array([0.1], dtype=np.float32).tobytes())


@pytest.mark.asyncio
async def test_backend_without_queue_flush_cannot_arm():
    media = FakeMedia()
    media.audio = object()
    output = ReachyMediaAudio(media)
    with pytest.raises(RuntimeError, match="cannot flush"):
        await output.arm_playback()
    assert media.calls == []


@pytest.mark.asyncio
async def test_flush_failure_still_requests_stop_and_disarms():
    media = FakeMedia()
    output = ReachyMediaAudio(media)
    await output.arm_playback()

    def fail_flush():
        media.calls.append("clear_player_failed")
        raise RuntimeError("flush failed")

    media.clear_player = fail_flush
    with pytest.raises(RuntimeError, match="flush failed"):
        await output.halt_audio()
    assert media.calls[-2:] == ["clear_player_failed", "stop_playing"]
    with pytest.raises(RuntimeError, match="not armed"):
        await output.enqueue(np.array([0.1], dtype=np.float32).tobytes())


@pytest.mark.asyncio
async def test_microphone_chunks_are_copied_and_validated():
    media = FakeMedia()
    adapter = ReachyMediaAudio(media)
    await adapter.start_capture()
    assert await adapter.read_capture() is None
    samples = np.array([[0.25, -0.25], [0.0, 0.1]], dtype=np.float32)
    media.next_chunk = samples
    chunk = await adapter.read_capture()
    assert chunk is not None
    assert (chunk.sample_rate, chunk.channels) == (16_000, 2)
    assert np.array_equal(np.frombuffer(chunk.data, dtype="<f4").reshape(-1, 2), samples)
    samples[:] = 0
    assert np.frombuffer(chunk.data, dtype="<f4")[0] == pytest.approx(0.25)
    media.next_chunk = np.ones((1, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="shape or dtype"):
        await adapter.read_capture()
    await adapter.stop_capture()
    assert media.calls == ["start_recording", "stop_recording"]
    with pytest.raises(RuntimeError, match="not started"):
        await adapter.read_capture()


@pytest.mark.asyncio
async def test_guard_blocks_before_reachy_push_and_stop_halts_owned_audio():
    media = FakeMedia()
    audio = ReachyMediaAudio(media)
    playback = OwnedPlaybackGate(audio)

    class Ports:
        async def plan(self, _transcript):
            return [Speech("do not speak")]

        async def synthesize(self, _text):
            return np.array([0.1], dtype=np.float32).tobytes()

        async def stop(self):
            await playback.halt_audio()

    async def guard(action):
        return Verdict("block" if action.kind == "utterance" else "approve", "fixture")

    app = GuardedConversation(
        planner=Ports(), guard=guard, synthesizer=Ports(), audio=playback, emergency_stop=Ports()
    )
    assert (await app.run_turn("hello")).status == "block"
    assert media.calls == []
    assert (await app.run_turn("stop")).status == "stopped"
    assert media.calls[-2:] == ["clear_player", "stop_playing"]
