"""A deliberately small effectful tool for exercising the owned approval path.

The planner supplies only note text. The host fixes the destination and must
register this tool as effectful with an independent owner-approval broker.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TOOL_NAME = "append_local_note"
_MAX_FILE_BYTES = 1_048_576


class LocalNotesTool:
    """Append one exact, approved note to an owner-only local JSONL file.

    This adapter does not authorize itself. Register ``TOOL_NAME`` only in
    ``effectful_tools`` and pass a separate ``OwnerApprovalBroker`` to the
    guarded pipeline. It never accepts a planner-supplied path.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)

    @staticmethod
    def _record(name: str, arguments: Mapping[str, Any]) -> bytes:
        if name != TOOL_NAME or not isinstance(arguments, Mapping) or set(arguments) != {"text"}:
            raise ValueError("unsupported local note tool or arguments")
        value = arguments["text"]
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 240
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("note text must be 1..240 printable characters")
        return (json.dumps({"text": value}, ensure_ascii=False, separators=(",", ":")) + "\n").encode()

    def _append(self, record: bytes) -> None:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory = os.open(self.state_dir, flags)
        try:
            directory_stat = os.fstat(directory)
            if directory_stat.st_uid != os.getuid() or stat.S_IMODE(directory_stat.st_mode) & 0o077:
                raise PermissionError("local note directory must be owner-only")
            file = os.open(
                "notes.jsonl",
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            try:
                import fcntl

                fcntl.flock(file, fcntl.LOCK_EX)
                info = os.fstat(file)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) & 0o077
                ):
                    raise PermissionError("local note file must be private and unlinked")
                if info.st_size + len(record) > _MAX_FILE_BYTES:
                    raise ValueError("local note file exceeds size cap")
                view = memoryview(record)
                while view:
                    written = os.write(file, view)
                    if written <= 0:
                        raise OSError("local note write made no progress")
                    view = view[written:]
                os.fsync(file)
            finally:
                os.close(file)
        finally:
            os.close(directory)

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> None:
        record = self._record(name, arguments)
        await asyncio.to_thread(self._append, record)
