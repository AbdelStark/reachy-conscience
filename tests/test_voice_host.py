"""Operator-paced host tests; no robot, model, or TypeSafe calls."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from reachy_conscience import GuardPolicy, LocalNotesTool, OwnerApprovalBroker, PolicyStore, TurnResult
from reachy_conscience.owner_console import OwnerConsole
from reachy_conscience.reflex_speaking import SpeakingObservedPlayback
from reachy_conscience.voice_host import main, operator_turns


class FakeSession:
    def __init__(self, guard, results=None):
        self.guard = guard
        self.results = iter(results or [TurnResult("complete", 1)])
        self.seen_rules = []
        self.stops = 0

    async def run_once(self, *, listen_timeout_s=30.0):
        self.seen_rules.append((self.guard.policy.rules, listen_timeout_s))
        return next(self.results)

    async def stop(self):
        self.stops += 1
        return TurnResult("stopped")


def test_operator_reloads_policy_between_explicit_turns_and_redacts_status(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")
    guard = SimpleNamespace(policy=store.load())
    session = FakeSession(guard, [TurnResult("complete", 1), TurnResult("hold", 0)])
    commands = iter(["unknown", "", "", "q"])
    reports = []

    def prompt(_message):
        command = next(commands)
        if len(session.seen_rules) == 1 and command == "":
            store.save(GuardPolicy(rules=("Never reveal an address",)))
        return command

    operator_turns(session, store, guard, prompt=prompt, report=reports.append, listen_timeout_s=5)
    assert session.seen_rules == [
        (("Never reveal a password", "Do not startle a nearby person"), 5),
        (("Never reveal an address",), 5),
    ]
    assert session.stops == 1
    assert reports == [
        "Unknown command; no microphone was opened.",
        "Turn: complete; guarded outputs: 1.",
        "Turn: hold; guarded outputs: 0.",
    ]
    assert "address" not in " ".join(reports)


def test_operator_keeps_host_required_note_confirmation_after_policy_edit(tmp_path):
    store = PolicyStore(tmp_path / "policy.json", registered_tools={"append_local_note"})
    guard = SimpleNamespace(policy=store.load())
    session = FakeSession(guard)
    commands = iter(["", "q"])
    operator_turns(
        session,
        store,
        guard,
        prompt=lambda _message: next(commands),
        report=lambda _message: None,
        required_confirmation=frozenset({"append_local_note"}),
    )
    assert guard.policy.confirm_before == frozenset({"append_local_note"})
    assert store.load().confirm_before == frozenset()


def test_operator_scans_sign_without_opening_microphone_or_dispatching_output(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")
    guard = SimpleNamespace(policy=store.load())
    session = FakeSession(guard)
    reports = []
    seen = []

    class Sign:
        async def read_once(self):
            seen.append("camera")
            return "stop"

    async def screen_sign(text):
        seen.append(("guard", text, guard.policy.rules))
        return TurnResult("block", reason="fixture")

    commands = iter(["s", "q"])
    operator_turns(
        session,
        store,
        guard,
        prompt=lambda _message: next(commands),
        report=reports.append,
        sign_ingress=Sign(),
        screen_sign=screen_sign,
    )
    assert seen == [
        "camera",
        ("guard", "stop", ("Never reveal a password", "Do not startle a nearby person")),
    ]
    assert session.seen_rules == []
    assert session.stops == 1
    assert reports == ["Sign: block; no output dispatched."]


def test_operator_sign_response_requires_a_separate_explicit_command(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")
    guard = SimpleNamespace(policy=store.load())
    session = FakeSession(guard)
    calls = []
    reports = []

    class Sign:
        async def read_once(self):
            calls.append("inspect")
            return None

    async def inspect(_text):
        raise AssertionError("guard-only inspection was not requested")

    async def respond():
        calls.append("respond")
        return TurnResult("complete", 1)

    commands = iter(["r", "q"])
    operator_turns(
        session,
        store,
        guard,
        prompt=lambda _message: next(commands),
        report=reports.append,
        sign_ingress=Sign(),
        screen_sign=inspect,
        sign_responder=respond,
    )
    assert calls == ["respond"]
    assert session.seen_rules == []
    assert reports == ["Sign response: complete; guarded outputs: 1."]
    assert session.stops == 1


def test_operator_ends_on_uncertain_output_error(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")
    guard = SimpleNamespace(policy=store.load())
    session = FakeSession(guard, [TurnResult("output_error", reason="audio_sink_failed")])
    prompts = []

    def prompt(message):
        prompts.append(message)
        if len(prompts) > 1:
            raise AssertionError("another capture prompt must not be shown")
        return ""

    operator_turns(session, store, guard, prompt=prompt, report=lambda _message: None)
    assert len(session.seen_rules) == 1
    assert session.stops == 1


def test_operator_quiet_assertion_precedes_capture_but_not_unknown_commands(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")
    guard = SimpleNamespace(policy=store.load())
    events = []

    class OrderedSession(FakeSession):
        async def run_once(self, *, listen_timeout_s=30.0):
            events.append("capture")
            return await super().run_once(listen_timeout_s=listen_timeout_s)

    session = OrderedSession(guard)
    commands = iter(["unknown", "", "q"])
    operator_turns(
        session,
        store,
        guard,
        prompt=lambda _message: next(commands),
        report=lambda _message: None,
        on_operator_quiet=lambda: events.append("operator_quiet"),
    )
    assert events == ["operator_quiet", "capture"]
    assert session.stops == 1


def test_sign_response_failure_requests_owned_stop_instead_of_another_turn(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")
    guard = SimpleNamespace(policy=store.load())
    session = FakeSession(guard)
    reports = []

    class Sign:
        async def read_once(self):
            return None

    async def inspect(_text):
        raise AssertionError("inspection was not requested")

    async def fail():
        raise RuntimeError("output may be partial")

    operator_turns(
        session,
        store,
        guard,
        prompt=lambda _message: "r",
        report=reports.append,
        sign_ingress=Sign(),
        screen_sign=inspect,
        sign_responder=fail,
    )
    assert session.stops == 1
    assert session.seen_rules == []
    assert reports == ["Sign response failed; requesting owned stop. Output may already have started."]


@pytest.mark.parametrize("exit_mode", ["eof", "stop", "exception"])
def test_operator_always_stops_owned_output(tmp_path, exit_mode):
    store = PolicyStore(tmp_path / "policy.json")
    guard = SimpleNamespace(policy=store.load())
    session = FakeSession(guard, [TurnResult("stopped")])

    def prompt(_message):
        if exit_mode == "eof":
            raise EOFError
        if exit_mode == "exception":
            raise KeyboardInterrupt
        return ""

    if exit_mode == "exception":
        with pytest.raises(KeyboardInterrupt):
            operator_turns(session, store, guard, prompt=prompt, report=lambda _message: None)
    else:
        operator_turns(session, store, guard, prompt=prompt, report=lambda _message: None)
    assert session.stops == 1
    assert len(session.seen_rules) == (1 if exit_mode == "stop" else 0)


def test_cli_help_and_hardware_gate_do_not_import_robot_or_start_capture(tmp_path, capsys):
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "operator-paced" in capsys.readouterr().out.lower()
    base = [
        "--state-dir",
        str(tmp_path / "state"),
        "--model-path",
        str(tmp_path / "model"),
        "--ollama-model",
        "local-model",
        "--robot-host",
        "127.0.0.1",
        "--connection-mode",
        "localhost_only",
    ]
    with pytest.raises(SystemExit) as gated:
        main(base)
    assert gated.value.code == 2
    assert "acknowledge-experimental-hardware" in capsys.readouterr().err
    assert not (tmp_path / "state").exists()
    with pytest.raises(SystemExit) as missing_camera:
        main([*base, "--acknowledge-experimental-hardware", "--enable-camera-sign-responses"])
    assert missing_camera.value.code == 2
    assert "require --enable-camera-signs" in capsys.readouterr().err


def test_reflex_speaking_option_needs_a_private_writer_token_before_state_creation(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("reachy_conscience.voice_host.sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.delenv("REFLEX_SPEAKING_WRITER_TOKEN", raising=False)
    with pytest.raises(SystemExit) as missing_token:
        main(
            [
                "--state-dir",
                str(tmp_path / "state"),
                "--model-path",
                str(tmp_path / "model"),
                "--ollama-model",
                "local-model",
                "--robot-host",
                "127.0.0.1",
                "--connection-mode",
                "localhost_only",
                "--acknowledge-experimental-hardware",
                "--reflex-speaking-relay-url",
                "http://127.0.0.1:8048",
            ]
        )
    assert missing_token.value.code == 2
    assert "speaking writer token" in capsys.readouterr().err
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize(
    ("enable_notes", "enable_signs", "enable_responses", "enable_reflex"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (True, True, False, False),
        (False, True, True, False),
        (True, True, True, False),
        (False, False, False, True),
    ],
)
def test_cli_composes_guarded_owned_ports_without_running_a_turn(
    tmp_path, monkeypatch, enable_notes, enable_signs, enable_responses, enable_reflex
):
    if enable_signs:
        pytest.importorskip("PIL")
    seen = []
    console_brokers = []

    class FakeRobot:
        def __init__(self, **options):
            seen.append(("robot_options", options))
            self.media = SimpleNamespace(get_input_channels=lambda: 1, get_frame_jpeg=lambda: None)
            self.client = SimpleNamespace()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            seen.append("robot_closed")

    class FakeTypeSafeClient:
        def __init__(self, **_options):
            seen.append("guard_client_opened")

        def close(self):
            seen.append("guard_client_closed")

    def inspect_session(session, _store, guard, **_options):
        conversation = session.conversation
        if enable_reflex:
            assert isinstance(session.playback.backend, SpeakingObservedPlayback)
            assert session.ingress.capture is session.playback.backend.backend
            assert callable(_options["on_operator_quiet"])
        else:
            assert session.ingress.capture is session.playback.backend
            assert _options["on_operator_quiet"] is None
        assert session.playback is conversation.audio
        assert conversation.guard is guard
        assert conversation.require_inbound_route is True
        assert conversation.emergency_stop.audio is conversation.audio
        assert conversation.motion is None
        assert conversation.effectful_tools == (
            frozenset({"append_local_note"}) if enable_notes else frozenset()
        )
        assert _store.registered_tools == conversation.effectful_tools
        if enable_notes:
            assert isinstance(conversation.tools, LocalNotesTool)
            assert isinstance(conversation.owner_approval, OwnerApprovalBroker)
            assert conversation.owner_approval is console_brokers[0]
            assert "append_local_note" in guard.policy.confirm_before
            assert conversation.planner.system.endswith("No other tools or motion are available.")
        else:
            assert conversation.tools is None and conversation.owner_approval is None
            assert console_brokers == [None]
            assert not guard.policy.confirm_before
            assert "No tools or motion are available" in conversation.planner.system
        assert conversation.enable_motion is False
        assert (_options["sign_ingress"] is not None) == enable_signs
        assert (_options["screen_sign"] is not None) == enable_signs
        assert (_options["sign_responder"] is not None) == enable_responses
        if enable_signs:
            assert _options["sign_ingress"].camera.media is session.ingress.capture.media
        if enable_responses:
            result = asyncio.run(_options["sign_responder"]())
            assert result.status == "no_input"
        seen.append("owned_session_composed")

    class TrackingConsole(OwnerConsole):
        def __init__(self, store, ledger_path, **options):
            assert (options["approval_broker"] is not None) == enable_notes
            console_brokers.append(options["approval_broker"])
            super().__init__(store, ledger_path, **options)

    monkeypatch.setitem(sys.modules, "reachy_mini", SimpleNamespace(ReachyMini=FakeRobot))
    monkeypatch.setitem(
        sys.modules,
        "typesafe_sdk",
        SimpleNamespace(TypeSafeClient=FakeTypeSafeClient, RetryPolicy=lambda **_options: object()),
    )
    monkeypatch.setattr(
        "reachy_conscience.voice_host.FasterWhisperTranscriber.from_local_model",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr("reachy_conscience.voice_host.operator_turns", inspect_session)
    monkeypatch.setattr("reachy_conscience.voice_host.OwnerConsole", TrackingConsole)
    monkeypatch.setattr("reachy_conscience.voice_host.sys.stdin", SimpleNamespace(isatty=lambda: True))
    if enable_reflex:
        monkeypatch.setenv("REFLEX_SPEAKING_WRITER_TOKEN", "w" * 40)
    if enable_signs:
        monkeypatch.setattr("reachy_conscience.voice_host.shutil.which", lambda _binary: "/usr/bin/tesseract")
    assert (
        main(
            [
                "--state-dir",
                str(tmp_path / "state"),
                "--model-path",
                str(tmp_path / "model"),
                "--ollama-model",
                "local-model",
                "--robot-host",
                "127.0.0.1",
                "--connection-mode",
                "localhost_only",
                "--acknowledge-experimental-hardware",
                *(["--enable-local-notes"] if enable_notes else []),
                *(["--enable-camera-signs"] if enable_signs else []),
                *(["--enable-camera-sign-responses"] if enable_responses else []),
                *(["--reflex-speaking-relay-url", "http://127.0.0.1:8048"] if enable_reflex else []),
            ]
        )
        == 0
    )
    assert seen == [
        "guard_client_opened",
        (
            "robot_options",
            {
                "host": "127.0.0.1",
                "connection_mode": "localhost_only",
                "spawn_daemon": False,
                "use_sim": False,
            },
        ),
        "owned_session_composed",
        "robot_closed",
        "guard_client_closed",
    ]
