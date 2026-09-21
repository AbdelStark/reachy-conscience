# Reachy Conscience

An inspectable action guard for LLM-driven Reachy Mini apps. Applications submit a proposed utterance, tool call, motion, or inbound text as a typed record. Jev supplies narrow judgments; deterministic code chooses approve, hold, or block. A hard stop is always handled in code before any model call.

This repository contains the tested guard core, local ledger, and an independently owned, non-streaming conversation-path foundation. It does **not** wrap or protect the official Conversation App. It is not yet a deployable robot app: ASR, LLM planner, TTS, audio, tool, and motion adapters still need integration and on-robot validation. It is **not** a physical safety system, prompt-injection cure, or substitute for motor limits, torque limits, firmware emergency stop, or owner supervision. No block-rate or latency claim has been measured.

## Example

```python
from reachy_conscience import Action, GuardPolicy, decide

policy = GuardPolicy(rules=("Never reveal the Wi-Fi password",), confirm_before=frozenset({"send_message"}))
action = Action(kind="tool_call", tool="send_message", summary="send a reminder")
verdict = decide(action, {"violates_rule_1": 0.1}, policy)
assert verdict.kind == "hold"  # confirmation wins even if model misses it
```

The caller must put this guard *before* TTS, tool dispatch, and motion execution. An after-the-fact observer cannot block an action. On missing or malformed model output, tools and motions do not execute. Holds expire without execution. The ledger stores short summaries and probabilities, not raw transcripts, by default.

## Owned conversation path

`GuardedConversation` accepts a final transcript and calls a proposal-only planner. It guards inbound text before planning, then guards each complete speech, tool, or motion proposal before calling its output adapter. Speech is synthesized only after approval; audio bytes are enqueued only after synthesis. A hold or block stops the rest of the turn, rather than allowing a later “done” utterance to run. Exact hard-stop commands bypass the planner and guard and interrupt pending work. Effectful tools are not executable in this MVP; only explicitly registered read-only tool names may dispatch. Motion is disabled by default. No hold auto-resumes.

The planner must never receive TTS, tool-dispatch, or robot-motion handles. Output adapters must enforce their own allowlists, hardware limits, queue cancellation on emergency stop, and physical stop behavior. A stop can prevent *future* enqueue calls from this pipeline; it cannot retract audio already queued or played by an adapter. There is no Conversation App streaming-audio hook here, and an already-streaming upstream pipeline cannot be made safe by observing its final transcript.

The current `ToolCall.summary` and `Motion.motion_class` are guard context, not a cryptographic binding to executable arguments or a calibrated motion safety assessment. Do not enable effectful tools or real motion from this package without an exact-argument gate, trusted owner confirmation, bounded hardware adapter, and integration tests. With no Reachy Mini available, all output-order tests use spies only. Run the [offline ordering example](examples/guarded_turn.py) with `uv run python examples/guarded_turn.py`; it uses fake verdicts and no Jev call or robot.

An irreversible tool call is held for owner confirmation even if Jev assigns low severity. This is a policy decision in code, not a claim that the model can reliably identify every external effect. Integrators must also place known effectful tools in `confirm_before`.

Run `uv sync --dev`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest`. See [CHANGELOG.md](CHANGELOG.md), [CITATION.cff](CITATION.cff), [SECURITY.md](SECURITY.md), and [CONTRIBUTING.md](CONTRIBUTING.md).
