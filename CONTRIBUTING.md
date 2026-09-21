# Contributing

Changes to an approve path require adversarial and failure-mode tests. Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`, and `uv build`. Keep I/O adapters separate from guard policy, and never include private recordings or credentials in fixtures.
