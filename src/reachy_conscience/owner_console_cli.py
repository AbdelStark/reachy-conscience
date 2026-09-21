"""Explicit local owner-console entry point; never connects to a robot."""

from __future__ import annotations

import argparse
import stat
import threading
from pathlib import Path
from typing import Any

from .jev import AsyncTypeSafeGuard
from .owner_console import OwnerConsole
from .policy_store import PolicyStore


def _private_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("state directory must not be a symlink")
    if not path.exists():
        path.mkdir(mode=0o700)
    if not path.is_dir() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("state directory must be owner-only (mode 0700)")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve the local Conscience policy and ledger console (no robot connection)."
    )
    parser.add_argument(
        "--state-dir", required=True, type=Path, help="Owner-only directory for policy and ledger"
    )
    parser.add_argument("--port", type=int, default=0, help="Loopback port; 0 chooses an available port")
    parser.add_argument(
        "--enable-typesafe-preview",
        action="store_true",
        help="Allow confirmed five-case preview or 20-case synthetic red-team calls",
    )
    args = parser.parse_args(argv)
    try:
        state_dir = _private_directory(args.state_dir)
        client: Any = None
        guard_factory = None
        if args.enable_typesafe_preview:
            try:
                from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError
            except ImportError as exc:
                raise ValueError("install reachy-conscience[jev] for TypeSafe preview") from exc
            try:
                client = TypeSafeClient(timeout=1.5, retry=RetryPolicy(max_retries=0))
            except TypeSafeError as exc:
                raise ValueError("TypeSafe preview configuration unavailable") from exc

            def guard_factory(policy: Any) -> AsyncTypeSafeGuard:
                return AsyncTypeSafeGuard(client, policy)

        try:
            with OwnerConsole(
                PolicyStore(state_dir / "policy.json"),
                state_dir / "ledger.db",
                guard_factory=guard_factory,
                port=args.port,
            ) as console:
                print(f"Owner console: {console.url}")
                print(f"Owner token: {console.token}")
                print("Keep this terminal private. Ctrl+C stops the console; no robot is connected.")
                try:
                    threading.Event().wait()
                except KeyboardInterrupt:
                    pass
        finally:
            if client is not None:
                client.close()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
