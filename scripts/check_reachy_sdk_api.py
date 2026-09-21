"""SDK 1.10 static API contract check. Does not connect to a robot."""

from reachy_mini.media.media_manager import MediaManager

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
print("Reachy Mini 1.10 media API contract present; no robot connected")
