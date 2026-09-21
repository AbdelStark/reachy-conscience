# Contributing

Changes to an approve path require adversarial and failure-mode tests. Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`, and `uv build`. For SDK adapter changes, also run `uv sync --extra robot`, `uv run --extra robot python scripts/check_reachy_sdk_api.py`, and the fake-backend audio and motion tests. Keep I/O adapters separate from guard policy, and never include private recordings or credentials in fixtures. Source-level checks cannot establish hardware safety or stop latency; label any untested behavior accordingly.
