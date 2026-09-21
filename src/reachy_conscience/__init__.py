"""Deterministic action gate for Reachy Mini integrations."""

from .guard import (
    Action,
    GuardAssessment,
    GuardPolicy,
    Verdict,
    decide,
    guard_action,
    is_hard_stop,
    lint_rules,
)
from .hold import HoldRegistry
from .ingress import EnergySegmenter, OwnedAudioIngress
from .jev import AsyncTypeSafeGuard, action_state, ask_typesafe, guard_questions, guard_typesafe
from .ledger import Ledger
from .local_asr import FasterWhisperTranscriber
from .local_tts import EspeakFfmpegSynthesizer
from .pipeline import GuardedConversation, Motion, OwnerApproval, Speech, ToolCall, TurnResult
from .policy_store import PolicyStore, dry_run_policy, policy_from_lines, preview_cases
from .proposal_planner import LocalOllamaPlanner, decode_proposals
from .reachy_audio import AudioChunk, ReachyMediaAudio
from .reachy_motion import BoundedPose, ReachyOutputStop, ReachySdkMotion, bounded_pose
from .session import OwnedVoiceSession

__all__ = [
    "Action",
    "GuardPolicy",
    "GuardAssessment",
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
    "OwnedVoiceSession",
    "Speech",
    "ToolCall",
    "Motion",
    "OwnerApproval",
    "TurnResult",
    "PolicyStore",
    "policy_from_lines",
    "preview_cases",
    "dry_run_policy",
    "AudioChunk",
    "ReachyMediaAudio",
    "BoundedPose",
    "ReachyOutputStop",
    "ReachySdkMotion",
    "bounded_pose",
]
