"""SDK 1.10 static API contract check. Does not connect to a robot."""

from reachy_mini.io.abstract import AbstractClient
from reachy_mini.io.protocol import GotoTaskRequest, StopMoveCmd
from reachy_mini.media.media_manager import MediaManager
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import InterpolationTechnique

REQUIRED = (
    "get_output_audio_samplerate",
    "get_output_channels",
    "start_playing",
    "push_audio_sample",
    "stop_playing",
    "get_input_audio_samplerate",
    "get_input_channels",
    "start_recording",
    "get_audio_sample",
    "stop_recording",
)

missing = [name for name in REQUIRED if not callable(getattr(MediaManager, name, None))]
if missing:
    raise SystemExit(f"Reachy Mini media API changed: {missing}")
motion_methods = ("send_task_request", "wait_for_task_completion", "send_command")
missing_motion = [name for name in motion_methods if not callable(getattr(AbstractClient, name, None))]
if missing_motion or not callable(create_head_pose) or not hasattr(InterpolationTechnique, "MIN_JERK"):
    raise SystemExit(f"Reachy Mini motion API changed: {missing_motion}")
if GotoTaskRequest.model_fields.keys() != {"head", "antennas", "duration", "method", "body_yaw"}:
    raise SystemExit("Reachy Mini GotoTaskRequest fields changed")
if StopMoveCmd().type != "stop_move":
    raise SystemExit("Reachy Mini stop command changed")
print("Reachy Mini 1.10 media and motion API contracts present; no robot connected")
