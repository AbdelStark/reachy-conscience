# Reachy Conscience

An inspectable action guard for LLM-driven Reachy Mini apps. Applications submit a proposed utterance, tool call, motion, or inbound text as a typed record. Jev supplies narrow judgments; deterministic code chooses approve, hold, or block. A hard stop is always handled in code before any model call.

This repository currently contains the tested guard core and local ledger, not a deployable Conversation App integration. It is **not** a physical safety system, prompt-injection cure, or substitute for motor limits, torque limits, firmware emergency stop, or owner supervision. No block-rate or latency claim has been measured.

## Example

```python
from reachy_conscience import Action, GuardPolicy, decide

policy = GuardPolicy(rules=("Never reveal the Wi-Fi password",), confirm_before=frozenset({"send_message"}))
action = Action(kind="tool_call", tool="send_message", summary="send a reminder")
verdict = decide(action, {"violates_rule_1": 0.1}, policy)
assert verdict.kind == "hold"  # confirmation wins even if model misses it
```

The caller must put this guard *before* TTS, tool dispatch, and motion execution. An after-the-fact observer cannot block an action. On missing or malformed model output, tools and motions do not execute. Holds expire without execution. The ledger stores short summaries and probabilities, not raw transcripts, by default.

An irreversible tool call is held for owner confirmation even if Jev assigns low severity. This is a policy decision in code, not a claim that the model can reliably identify every external effect. Integrators must also place known effectful tools in `confirm_before`.

Run `uv sync --dev`, `uv run ruff check .`, and `uv run pytest`. See [SECURITY.md](SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
