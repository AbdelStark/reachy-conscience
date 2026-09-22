"""Offline synthesis never plays sound; no robot or LLM is used."""

import asyncio
import math
import shutil
import struct
import sys

import pytest

from reachy_conscience import EspeakFfmpegSynthesizer, GuardedConversation, Speech, Verdict


def test_tts_rejects_unbounded_or_ambiguous_settings():
    with pytest.raises(ValueError, match="voice"):
        EspeakFfmpegSynthesizer(voice="--stdout")
    with pytest.raises(ValueError, match="settings"):
        EspeakFfmpegSynthesizer(channels=6)


@pytest.mark.asyncio
async def test_tts_rejects_oversized_text_before_process_spawn():
    tts = EspeakFfmpegSynthesizer(espeak_binary="not-installed")
    with pytest.raises(ValueError, match="outside"):
        await tts.synthesize("x" * 241)
    with pytest.raises(FileNotFoundError):
        await tts.synthesize("Hello")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
async def test_tts_stops_a_child_as_soon_as_its_output_exceeds_the_cap(stream):
    tts = EspeakFfmpegSynthesizer(timeout_s=3)
    code = f"import sys,time; sys.{stream}.buffer.write(b'x'*1048576); sys.{stream}.flush(); time.sleep(10)"
    with pytest.raises(RuntimeError, match="output cap"):
        await tts._call(sys.executable, "-c", code, data=b"", max_bytes=1024)


@pytest.mark.asyncio
async def test_tts_cancellation_reaps_its_child(monkeypatch):
    spawned = []
    spawned_event = asyncio.Event()
    create = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await create(*args, **kwargs)
        spawned.append(process)
        spawned_event.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    tts = EspeakFfmpegSynthesizer(timeout_s=20)
    task = asyncio.create_task(
        tts._call(sys.executable, "-c", "import time; time.sleep(10)", data=b"", max_bytes=1024)
    )
    await asyncio.wait_for(spawned_event.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert spawned[0].returncode is not None


@pytest.mark.asyncio
async def test_tts_timeout_reaps_its_child(monkeypatch):
    spawned = []
    create = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await create(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    tts = EspeakFfmpegSynthesizer(timeout_s=0.05)
    with pytest.raises(TimeoutError):
        await tts._call(sys.executable, "-c", "import time; time.sleep(10)", data=b"", max_bytes=1024)
    assert len(spawned) == 1
    assert spawned[0].returncode is not None


@pytest.mark.asyncio
@pytest.mark.skipif(
    not shutil.which("espeak-ng") or not shutil.which("ffmpeg"),
    reason="offline TTS binaries are not installed",
)
async def test_offline_tts_yields_bounded_reachy_pcm_only_after_guard():
    events = []
    tts = EspeakFfmpegSynthesizer()

    class Ports:
        async def plan(self, _transcript):
            return [Speech("Hello, Reachy.")]

        async def enqueue(self, pcm):
            events.append(("enqueue", pcm))

        async def stop(self):
            events.append(("stop",))

    class RecordingTts:
        async def synthesize(self, text):
            events.append(("synthesize", text))
            return await tts.synthesize(text)

    async def blocked(action):
        events.append(("guard", action.kind))
        return Verdict("block" if action.kind == "utterance" else "approve", "fixture")

    ports = Ports()
    app = GuardedConversation(
        planner=ports,
        guard=blocked,
        synthesizer=RecordingTts(),
        audio=ports,
        emergency_stop=ports,
    )
    assert (await app.run_turn("hello")).status == "block"
    assert events == [("guard", "inbound"), ("guard", "utterance")]

    async def approved(action):
        events.append(("guard", action.kind))
        return Verdict("approve", "fixture")

    app.guard = approved
    assert (await app.run_turn("hello")).status == "complete"
    assert [event[0] for event in events[-4:]] == ["guard", "guard", "synthesize", "enqueue"]
    pcm = events[-1][1]
    assert isinstance(pcm, bytes)
    assert 0 < len(pcm) <= 16_000 * 4 * 30
    assert len(pcm) % 4 == 0
    samples = [sample for (sample,) in struct.iter_unpack("<f", pcm)]
    assert all(math.isfinite(sample) and abs(sample) <= 1 for sample in samples)
    assert any(abs(sample) > 0.01 for sample in samples)
