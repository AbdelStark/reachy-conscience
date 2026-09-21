"""Conservatively bounded Reachy Mini 1.10 motion output.

This adapter uses the pinned SDK's task and stop protocol, not the official
Conversation App. Its limits are application caps, not proven physical limits.
It has only been exercised with a fake SDK client, never a robot.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

_POSE_KEYS = frozenset({"yawDeg", "pitchDeg", "rollDeg", "zMm", "rightAntennaDeg", "leftAntennaDeg"})
_LIMITS = {
    "small_gesture": (15.0, 10.0, 10.0, 5.0, 25.0, 0.5, 1.5),
    "full_range_head": (25.0, 15.0, 12.0, 8.0, 35.0, 0.75, 2.0),
}


class ReachyTaskClient(Protocol):
    def send_task_request(self, request: Any) -> Any: ...
    def wait_for_task_completion(self, task_id: Any, timeout: float) -> None: ...
    def send_command(self, command: Any) -> None: ...


class ReachyMotionRobot(Protocol):
    client: ReachyTaskClient


@dataclass(frozen=True, slots=True)
class BoundedPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    z_mm: float
    right_antenna_deg: float
    left_antenna_deg: float
    duration_s: float


def bounded_pose(motion_class: str, target: Mapping[str, Any]) -> BoundedPose:
    """Reject unsupported classes, missing fields, extras, and out-of-cap poses."""
    limits = _LIMITS.get(motion_class)
    if limits is None:
        raise ValueError("motion class is not enabled by this adapter")
    if not isinstance(target, Mapping) or not _POSE_KEYS <= set(target):
        raise ValueError("motion target must contain a complete pose")
    if set(target) - _POSE_KEYS - {"durationS"}:
        raise ValueError("motion target has unknown fields")
    names = ("yawDeg", "pitchDeg", "rollDeg", "zMm", "rightAntennaDeg", "leftAntennaDeg")
    values = []
    for name in names:
        value = target[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        values.append(float(value))
    for value, limit in zip(values, (*limits[:4], limits[4], limits[4]), strict=True):
        if abs(value) > limit:
            raise ValueError("motion target exceeds application cap")
    duration = target.get("durationS", limits[5])
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not limits[5] <= duration <= limits[6]
    ):
        raise ValueError("motion duration outside application cap")
    return BoundedPose(*values, float(duration))


class ReachySdkMotion:
    """Guarded output port for complete poses; disarmed until explicitly armed.

    The SDK's public ``goto_target`` blocks until completion and does not expose
    a task handle. SDK 1.10's ``client`` task methods and ``StopMoveCmd`` are
    version-pinned here so stop can be sent while motion is in progress. They
    are not a firmware emergency stop, and their live ordering is unverified.
    """

    def __init__(self, robot: ReachyMotionRobot) -> None:
        self.client = robot.client
        self._lock = threading.Lock()
        self._armed = False

    def arm_motion(self) -> None:
        """Opt in after an operator has checked the robot and surroundings."""
        with self._lock:
            self._armed = True

    def _start(self, pose: BoundedPose) -> Any:
        from reachy_mini.io.protocol import GotoTaskRequest
        from reachy_mini.utils import create_head_pose
        from reachy_mini.utils.interpolation import InterpolationTechnique

        head = create_head_pose(
            z=pose.z_mm,
            roll=pose.roll_deg,
            pitch=pose.pitch_deg,
            yaw=pose.yaw_deg,
            mm=True,
        )
        request = GotoTaskRequest(
            head=head.flatten().tolist(),
            antennas=[math.radians(pose.right_antenna_deg), math.radians(pose.left_antenna_deg)],
            duration=pose.duration_s,
            method=InterpolationTechnique.MIN_JERK,
            body_yaw=None,
        )
        with self._lock:
            if not self._armed:
                raise RuntimeError("Reachy motion is not armed")
            return self.client.send_task_request(request)

    async def execute(self, motion_class: str, target: Mapping[str, Any]) -> None:
        pose = bounded_pose(motion_class, target)
        task_id = await asyncio.to_thread(self._start, pose)
        try:
            await asyncio.to_thread(self.client.wait_for_task_completion, task_id, pose.duration_s + 1.0)
        except BaseException:
            # A timed-out or cancelled local wait does not cancel a daemon task.
            await self.stop_motion()
            raise

    def _stop_motion(self) -> None:
        from reachy_mini.io.protocol import StopMoveCmd

        with self._lock:
            self._armed = False
            self.client.send_command(StopMoveCmd())

    async def stop_motion(self) -> None:
        """Disarm and request SDK move cancellation; hardware proof is pending."""
        await asyncio.to_thread(self._stop_motion)


class ReachyOutputStop:
    """Stop owned audio and motion ports; does not undo a dispatched tool."""

    def __init__(self, audio: Any, motion: ReachySdkMotion) -> None:
        self.audio = audio
        self.motion = motion

    async def stop(self) -> None:
        outcomes = await asyncio.gather(
            self.audio.halt_audio(), self.motion.stop_motion(), return_exceptions=True
        )
        if any(isinstance(outcome, BaseException) for outcome in outcomes):
            raise RuntimeError("one or more Reachy output stop requests failed")
