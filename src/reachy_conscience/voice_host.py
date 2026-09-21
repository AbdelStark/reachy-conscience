"""Explicitly paced Reachy host for the independently owned guarded voice path.

The SDK does not acknowledge playback completion. An operator must request
each capture after the speaker is quiet; this module never starts another
microphone turn automatically. It never opens the official Conversation App.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .guard import GuardPolicy
from .ingress import OwnedAudioIngress
from .jev import AsyncTypeSafeGuard
from .ledger import Ledger
from .local_asr import FasterWhisperTranscriber
from .local_notes import TOOL_NAME, LocalNotesTool
from .local_tts import EspeakFfmpegSynthesizer
from .owner_approval import OwnerApprovalBroker
from .owner_console import OwnerConsole
from .owner_console_cli import _private_directory
from .pipeline import GuardedConversation, TurnResult
from .policy_store import PolicyStore
from .proposal_planner import LocalOllamaPlanner
from .reachy_audio import ReachyMediaAudio
from .reachy_motion import ReachyOutputStop, ReachySdkMotion
from .session import OwnedVoiceSession


class TurnSession(Protocol):
    async def run_once(self, *, listen_timeout_s: float = 30.0) -> TurnResult: ...
    async def stop(self) -> TurnResult: ...


def operator_turns(
    session: TurnSession,
    policy_store: PolicyStore,
    guard: AsyncTypeSafeGuard,
    *,
    prompt: Callable[[str], str] = input,
    report: Callable[[str], None] = print,
    listen_timeout_s: float = 30.0,
    required_confirmation: frozenset[str] = frozenset(),
) -> None:
    """Reload policy only between turns; stop owned output on every exit.

    The prompt and status contain no transcript or proposed text. A blank
    response requests exactly one capture. Any other text except ``q`` is
    ignored. An exception, EOF, or terminal stop closes the session.
    """
    if not 1 <= listen_timeout_s <= 120:
        raise ValueError("listen timeout must be within 1..120 seconds")
    try:
        while True:
            try:
                command = prompt("Wait until the speaker is quiet. Enter: listen once; q: stop > ")
            except EOFError:
                break
            if command.strip().lower() in {"q", "quit"}:
                break
            if command.strip():
                report("Unknown command; no microphone was opened.")
                continue
            guard.policy = _effective_policy(policy_store.load(), required_confirmation)
            result = asyncio.run(session.run_once(listen_timeout_s=listen_timeout_s))
            report(f"Turn: {result.status}; guarded outputs: {result.delivered}.")
            if result.status in {"stopped", "interrupted"}:
                break
    finally:
        asyncio.run(session.stop())


def _effective_policy(policy: GuardPolicy, required_confirmation: frozenset[str]) -> GuardPolicy:
    """Keep host-required confirmation even if the editable policy omits it."""
    return replace(policy, confirm_before=policy.confirm_before | required_confirmation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Operator-paced owned Reachy voice host; experimental, no automatic next listen."
    )
    parser.add_argument(
        "--state-dir", required=True, type=Path, help="Owner-only policy and ledger directory"
    )
    parser.add_argument("--model-path", required=True, type=Path, help="Existing local faster-whisper model")
    parser.add_argument("--ollama-model", required=True, help="Already installed local proposal model")
    parser.add_argument("--robot-host", required=True, help="Explicit Reachy daemon host")
    parser.add_argument(
        "--connection-mode",
        required=True,
        choices=("localhost_only", "network"),
        help="Explicit SDK connection mode; no auto-discovery fallback",
    )
    parser.add_argument("--console-port", type=int, default=0, help="Owner console loopback port")
    parser.add_argument("--listen-timeout", type=float, default=30.0, help="Seconds per capture, 1..120")
    parser.add_argument(
        "--enable-local-notes",
        action="store_true",
        help="Opt in to exact-owner-approved local notes; no external messages",
    )
    parser.add_argument(
        "--acknowledge-experimental-hardware",
        action="store_true",
        help="Required: no robot stop, playback, or voice quality validation exists",
    )
    args = parser.parse_args(argv)
    if not args.acknowledge_experimental_hardware:
        parser.error("pass --acknowledge-experimental-hardware after reviewing the hardware limitations")
    if not sys.stdin.isatty():
        parser.error("interactive operator terminal required")
    if not 1 <= args.listen_timeout <= 120:
        parser.error("listen timeout must be within 1..120 seconds")
    if args.connection_mode == "localhost_only" and args.robot_host not in {"127.0.0.1", "localhost"}:
        parser.error("localhost_only requires a local robot host")
    try:
        state_dir = _private_directory(args.state_dir)
        effectful_tools = frozenset({TOOL_NAME}) if args.enable_local_notes else frozenset()
        store = PolicyStore(state_dir / "policy.json", registered_tools=effectful_tools)
        policy = _effective_policy(store.load(), effectful_tools)
        approval_broker = OwnerApprovalBroker() if args.enable_local_notes else None
        notes_tool = LocalNotesTool(state_dir) if args.enable_local_notes else None
        transcriber = FasterWhisperTranscriber.from_local_model(args.model_path)
        try:
            from reachy_mini import ReachyMini
            from typesafe_sdk import RetryPolicy, TypeSafeClient
        except ImportError as exc:
            raise ValueError("install the robot, asr, and jev extras for the voice host") from exc
        client = TypeSafeClient(timeout=1.5, retry=RetryPolicy(max_retries=0))
        ledger = Ledger(state_dir / "ledger.db")
        try:
            with ReachyMini(
                host=args.robot_host,
                connection_mode=args.connection_mode,
                spawn_daemon=False,
                use_sim=False,
            ) as robot:
                audio = ReachyMediaAudio(robot.media)
                motion = ReachySdkMotion(robot)  # Never armed; stop still requests StopMoveCmd.
                guard = AsyncTypeSafeGuard(client, policy)
                conversation = GuardedConversation(
                    planner=LocalOllamaPlanner(args.ollama_model, enable_local_notes=args.enable_local_notes),
                    guard=guard,
                    synthesizer=EspeakFfmpegSynthesizer(),
                    audio=audio,
                    emergency_stop=ReachyOutputStop(audio, motion),
                    ledger=ledger,
                    tools=notes_tool,
                    effectful_tools=effectful_tools,
                    owner_approval=approval_broker,
                )
                session = OwnedVoiceSession(
                    OwnedAudioIngress(audio, transcriber, channels=robot.media.get_input_channels()),
                    conversation,
                    audio,
                )
                with OwnerConsole(
                    store,
                    state_dir / "ledger.db",
                    port=args.console_port,
                    approval_broker=approval_broker,
                ) as console:
                    print(f"Owner console: {console.url}")
                    print(f"Owner token: {console.token}")
                    if args.enable_local_notes:
                        print(
                            "Local notes enabled: exact owner approval required in the console. "
                            "Motion and external messages are disabled; "
                            "protect notes.jsonl and retain a physical stop."
                        )
                    else:
                        print(
                            "Tools and motion are disabled. "
                            "Keep this terminal private; retain a physical stop."
                        )
                    operator_turns(
                        session,
                        store,
                        guard,
                        listen_timeout_s=args.listen_timeout,
                        required_confirmation=effectful_tools,
                    )
        finally:
            ledger.close()
            client.close()
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
