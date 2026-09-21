"""Synthetic PCM only: no microphone, model weights, or robot."""

import asyncio
import struct
import sys
import types
from collections import deque

import pytest

from reachy_conscience import (
    AudioChunk,
    EnergySegmenter,
    FasterWhisperTranscriber,
    GuardedConversation,
    OwnedAudioIngress,
    Speech,
    Verdict,
)


def pcm(level: float, frames: int, channels: int = 1) -> AudioChunk:
    return AudioChunk(struct.pack("<f", level) * (320 * frames * channels), 16_000, channels)


def segments(channels: int = 1) -> list[AudioChunk]:
    return [pcm(0.0, 5, channels), pcm(0.2, 12, channels), pcm(0.0, 30, channels)]


def test_energy_segmenter_preroll_silence_boundary_and_format():
    segmenter = EnergySegmenter(channels=1)
    result = [part for chunk in segments() for part in segmenter.feed(chunk)]
    assert len(result) == 1
    assert len(result[0].data) <= 47 * 320 * 4
    assert result[0].data.startswith(struct.pack("<f", 0.0))
    assert segmenter.feed(pcm(0.0, 35)) == []
    assert EnergySegmenter(channels=2).feed(pcm(0.1, 12, 2)) == []
    with pytest.raises(ValueError, match="format"):
        segmenter.feed(pcm(0.1, 1, 2))


def test_energy_segmenter_rejects_noise_and_bad_samples():
    segmenter = EnergySegmenter(channels=1)
    assert [part for chunk in [pcm(0.1, 2), pcm(0, 30)] for part in segmenter.feed(chunk)] == []
    for bad in [float("nan"), float("inf"), 1.1]:
        with pytest.raises(ValueError, match="sample"):
            segmenter.feed(pcm(bad, 1))
    with pytest.raises(ValueError, match="incomplete"):
        segmenter.feed(AudioChunk(b"abc", 16_000, 1))


def test_energy_segmenter_caps_continuous_speech():
    segmenter = EnergySegmenter(channels=1, max_segment_s=1)
    result = [part for _ in range(5) for part in segmenter.feed(pcm(0.2, 10))]
    assert len(result) == 1
    assert len(result[0].data) == 16_000 * 4


class FakeCapture:
    def __init__(self, chunks):
        self.chunks = deque(chunks)
        self.events = []

    async def start_capture(self):
        self.events.append("capture_start")

    async def read_capture(self):
        self.events.append("read")
        return self.chunks.popleft() if self.chunks else None

    async def stop_capture(self):
        self.events.append("capture_stop")


class FakeAsr:
    def __init__(self, capture, text="hello"):
        self.capture = capture
        self.text = text

    async def transcribe(self, segment):
        assert self.capture.events[-1] == "capture_stop"
        assert isinstance(segment, AudioChunk)
        self.capture.events.append("asr")
        return self.text


@pytest.mark.asyncio
async def test_owned_ingress_stops_capture_before_asr_and_guarded_output():
    capture = FakeCapture(segments())
    ingress = OwnedAudioIngress(capture, FakeAsr(capture), channels=1)
    transcript = await ingress.listen_once()
    assert transcript == "hello"
    assert capture.events[-2:] == ["capture_stop", "asr"]

    class Ports:
        async def plan(self, _text):
            return [Speech("reply")]

        async def synthesize(self, _text):
            capture.events.append("synthesize")
            return b"audio"

        async def enqueue(self, _audio):
            capture.events.append("enqueue")

        async def stop(self):
            capture.events.append("stop")

    async def guard(_action):
        capture.events.append("guard")
        return Verdict("approve", "ok")

    ports = Ports()
    result = await GuardedConversation(
        planner=ports, guard=guard, synthesizer=ports, audio=ports, emergency_stop=ports
    ).run_turn(transcript)
    assert result.status == "complete"
    assert capture.events[-4:] == ["guard", "guard", "synthesize", "enqueue"]


@pytest.mark.asyncio
async def test_final_asr_stop_bypasses_planner_and_guard():
    capture = FakeCapture(segments())
    transcript = await OwnedAudioIngress(capture, FakeAsr(capture, "stop"), channels=1).listen_once()

    class Ports:
        async def plan(self, _text):
            pytest.fail("hard stop must not call planner")

        async def synthesize(self, _text):
            pytest.fail("hard stop must not synthesize")

        async def enqueue(self, _audio):
            pytest.fail("hard stop must not enqueue")

        async def stop(self):
            capture.events.append("stop")

    async def guard(_action):
        pytest.fail("hard stop must not call Jev")

    ports = Ports()
    result = await GuardedConversation(
        planner=ports, guard=guard, synthesizer=ports, audio=ports, emergency_stop=ports
    ).run_turn(transcript)
    assert result.status == "stopped"
    assert capture.events[-3:] == ["capture_stop", "asr", "stop"]


@pytest.mark.asyncio
async def test_ingress_failure_does_not_transcribe_and_always_stops_capture():
    capture = FakeCapture([pcm(float("nan"), 1)])
    ingress = OwnedAudioIngress(capture, FakeAsr(capture), channels=1)
    with pytest.raises(ValueError, match="sample"):
        await ingress.listen_once()
    assert capture.events[-1] == "capture_stop"

    class PartialStart(FakeCapture):
        async def start_capture(self):
            self.events.append("capture_start")
            raise RuntimeError("capture failed after partial start")

    partial = PartialStart([])
    with pytest.raises(RuntimeError, match="partial start"):
        await OwnedAudioIngress(partial, FakeAsr(partial), channels=1).listen_once()
    assert partial.events == ["capture_start", "capture_stop"]

    class Stalled(FakeCapture):
        async def read_capture(self):
            await asyncio.sleep(0.1)
            return None

    stalled = Stalled([])
    ingress = OwnedAudioIngress(stalled, FakeAsr(stalled), channels=1, read_timeout_s=0.001)
    with pytest.raises(TimeoutError):
        await ingress.listen_once(timeout_s=1)
    assert stalled.events[-1] == "capture_stop"

    class SlowAsr(FakeAsr):
        async def transcribe(self, _segment):
            await asyncio.sleep(0.1)
            return "late"

    capture = FakeCapture(segments())
    ingress = OwnedAudioIngress(capture, SlowAsr(capture), channels=1, asr_timeout_s=0.001)
    with pytest.raises(TimeoutError):
        await ingress.listen_once()
    assert capture.events[-1] == "capture_stop"


def test_offline_asr_adapter_mixdown_and_limits(tmp_path, monkeypatch):
    class Model:
        def transcribe(self, samples, **options):
            assert len(samples) == 320 * 12
            assert abs(float(samples[0]) - 0.2) < 1e-6
            assert options == {
                "language": "en",
                "beam_size": 1,
                "condition_on_previous_text": False,
                "vad_filter": False,
            }
            return [type("Segment", (), {"text": " hello "})()], object()

    asr = FasterWhisperTranscriber(Model())
    assert asr._transcribe(pcm(0.2, 12, 2)) == "hello"
    with pytest.raises(ValueError, match="PCM"):
        asr._transcribe(AudioChunk(b"", 16_000, 1))
    with pytest.raises(ValueError, match="local model must contain"):
        FasterWhisperTranscriber.from_local_model(tmp_path)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"fixture")

    def load_model(path, **options):
        assert path == str(tmp_path)
        assert options == {"device": "cpu", "compute_type": "int8", "local_files_only": True}
        return Model()

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = load_model
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    assert isinstance(FasterWhisperTranscriber.from_local_model(tmp_path).model, Model)
