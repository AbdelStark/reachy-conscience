"""Deterministic action gate for Reachy Mini integrations."""

from .guard import Action, GuardPolicy, Verdict, decide, guard_action, is_hard_stop, lint_rules
from .hold import HoldRegistry
from .ingress import EnergySegmenter, OwnedAudioIngress
from .jev import AsyncTypeSafeGuard, action_state, ask_typesafe, guard_questions, guard_typesafe
from .ledger import Ledger
from .local_asr import FasterWhisperTranscriber
from .local_tts import EspeakFfmpegSynthesizer
from .pipeline import GuardedConversation, Motion, OwnerApproval, Speech, ToolCall, TurnResult
from .proposal_planner import LocalOllamaPlanner, decode_proposals
from .reachy_audio import AudioChunk, ReachyMediaAudio
from .reachy_motion import BoundedPose, ReachyOutputStop, ReachySdkMotion, bounded_pose

__all__ = [
    "Action",
    "GuardPolicy",
    "Verdict",
    "decide",
    "guard_action",
    "is_hard_stop",
    "lint_rules",
    "HoldRegistry",
    "EnergySegmenter",
    "OwnedAudioIngress",
    "FasterWhisperTranscriber",
    "EspeakFfmpegSynthesizer",
    "LocalOllamaPlanner",
    "decode_proposals",
    "Ledger",
    "action_state",
    "ask_typesafe",
    "guard_questions",
    "guard_typesafe",
    "AsyncTypeSafeGuard",
    "GuardedConversation",
    "Speech",
    "ToolCall",
    "Motion",
    "OwnerApproval",
    "TurnResult",
    "AudioChunk",
    "ReachyMediaAudio",
    "BoundedPose",
    "ReachyOutputStop",
    "ReachySdkMotion",
    "bounded_pose",
]
