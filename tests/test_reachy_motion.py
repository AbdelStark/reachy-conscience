"""SDK 1.10 motion protocol tests with an in-memory client, never hardware."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytest.importorskip("reachy_mini")

from reachy_conscience import (  # noqa: E402
    GuardedConversation,
    Motion,
    MotionContextSnapshot,
    ReachyOutputStop,
    ReachySdkMotion,
    Verdict,
    bounded_pose,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls = []
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.fail_wait = False

    def send_task_request(self, request):
        self.calls.append(("send_task", request))
        return "task-1"

    def wait_for_task_completion(self, task_id, timeout):
        self.calls.append(("wait", task_id, timeout))
        self.waiting.set()
        self.release.wait(timeout)
        if self.fail_wait:
            raise TimeoutError("daemon did not complete")

    def send_command(self, command):
        self.calls.append(("command", command.type))
        self.release.set()


class FakeRobot:
    def __init__(self) -> None:
        self.client = FakeClient()


class Context:
    async def snapshot(self):
        return MotionContextSnapshot("far", "normal", "cool", time.monotonic())


POSE = {
    "yawDeg": 10,
    "pitchDeg": -3,
    "rollDeg": 2,
    "zMm": 3,
    "rightAntennaDeg": 20,
    "leftAntennaDeg": -20,
    "durationS": 0.7,
}


def test_complete_pose_is_bounded_and_no_body_rotation() -> None:
    pose = bounded_pose("small_gesture", POSE)
    assert (pose.yaw_deg, pose.duration_s) == (10, 0.7)
    for name, target in (
        ("dance", POSE),
        ("small_gesture", {**POSE, "yawDeg": 16}),
        ("small_gesture", {**POSE, "zMm": float("nan")}),
        ("small_gesture", {**POSE, "durationS": 0.1}),
        ("small_gesture", {**POSE, "bodyYawDeg": 10}),
        ("small_gesture", {"yawDeg": 10}),
    ):
        with pytest.raises(ValueError):
            bounded_pose(name, target)
    assert bounded_pose("full_range_head", {**POSE, "yawDeg": 20, "durationS": 1.0}).yaw_deg == 20


@pytest.mark.asyncio
async def test_motion_is_disarmed_by_default_and_stop_interrupts_wait():
    robot = FakeRobot()
    adapter = ReachySdkMotion(robot)
    with pytest.raises(RuntimeError, match="not armed"):
        await adapter.execute("small_gesture", POSE)
    assert robot.client.calls == []

    adapter.arm_motion()
    pending = asyncio.create_task(adapter.execute("small_gesture", POSE))
    assert await asyncio.to_thread(robot.client.waiting.wait, 2)
    request = robot.client.calls[0][1]
    assert (request.body_yaw, request.duration, request.method.value) == (None, 0.7, "minjerk")
    assert len(request.head) == 16
    assert request.antennas[0] > 0 > request.antennas[1]
    await adapter.stop_motion()
    await pending
    assert robot.client.calls[-1] == ("command", "stop_move")
    with pytest.raises(RuntimeError, match="not armed"):
        await adapter.execute("small_gesture", POSE)


@pytest.mark.asyncio
async def test_pipeline_guards_motion_before_sdk_and_composite_stop_halts_both_ports():
    robot = FakeRobot()
    motion = ReachySdkMotion(robot)
    motion.arm_motion()
    audio_events = []

    class Audio:
        async def halt_audio(self):
            audio_events.append("halt_audio")

    stop = ReachyOutputStop(Audio(), motion)

    class Ports:
        async def plan(self, _transcript):
            return [Motion("small_gesture", POSE)]

        async def synthesize(self, _text):
            raise AssertionError("no speech")

        async def enqueue(self, _audio):
            raise AssertionError("no speech")

    async def guard(action):
        return Verdict("block" if action.kind == "motion" else "approve", "fixture")

    app = GuardedConversation(
        planner=Ports(),
        guard=guard,
        synthesizer=Ports(),
        audio=Ports(),
        motion=motion,
        enable_motion=True,
        motion_context=Context(),
        emergency_stop=stop,
    )
    assert (await app.run_turn("move")).status == "block"
    assert robot.client.calls == []
    assert (await app.run_turn("stop")).status == "stopped"
    assert robot.client.calls[-1] == ("command", "stop_move")
    assert audio_events == ["halt_audio"]


@pytest.mark.asyncio
async def test_pipeline_hard_stop_during_motion_reports_interrupted():
    robot = FakeRobot()
    motion = ReachySdkMotion(robot)
    motion.arm_motion()

    class Audio:
        async def halt_audio(self):
            pass

    class Ports:
        async def plan(self, _transcript):
            return [Motion("small_gesture", POSE)]

        async def synthesize(self, _text):
            raise AssertionError("no speech")

        async def enqueue(self, _audio):
            raise AssertionError("no speech")

    async def guard(_action):
        return Verdict("approve", "fixture")

    app = GuardedConversation(
        planner=Ports(),
        guard=guard,
        synthesizer=Ports(),
        audio=Ports(),
        motion=motion,
        enable_motion=True,
        motion_context=Context(),
        emergency_stop=ReachyOutputStop(Audio(), motion),
    )
    pending = asyncio.create_task(app.run_turn("move"))
    assert await asyncio.to_thread(robot.client.waiting.wait, 2)
    assert (await app.run_turn("stop")).status == "stopped"
    assert (await pending).status == "interrupted"


@pytest.mark.asyncio
async def test_motion_wait_failure_requests_stop_and_disarms():
    robot = FakeRobot()
    robot.client.fail_wait = True
    robot.client.release.set()
    motion = ReachySdkMotion(robot)
    motion.arm_motion()
    with pytest.raises(TimeoutError, match="daemon did not complete"):
        await motion.execute("small_gesture", POSE)
    assert robot.client.calls[-1] == ("command", "stop_move")
    with pytest.raises(RuntimeError, match="not armed"):
        await motion.execute("small_gesture", POSE)
