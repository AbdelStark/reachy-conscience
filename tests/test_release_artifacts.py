"""Archive fixtures for the public source and wheel release contract."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path
from stat import S_IFLNK

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("check_release_artifacts", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
release_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_check)
SOURCE_FILES = release_check.SOURCE_FILES
WHEEL_FILES = release_check.WHEEL_FILES
check_sdist = release_check.check_sdist
check_wheel = release_check.check_wheel


def _sdist(path: Path, names: set[str], *, link: str | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(names):
            member = tarfile.TarInfo(f"reachy_conscience-0.0.1/{name}")
            member.size = 0
            archive.addfile(member, io.BytesIO())
        if link is not None:
            member = tarfile.TarInfo(f"reachy_conscience-0.0.1/{link}")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../private"
            archive.addfile(member)


def _wheel(path: Path, names: set[str], *, link: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(names):
            archive.writestr(name, b"")
        if link is not None:
            member = zipfile.ZipInfo(link)
            member.create_system = 3
            member.external_attr = (S_IFLNK | 0o777) << 16
            archive.writestr(member, "../private")


def _wheel_files() -> set[str]:
    metadata = "reachy_conscience-0.0.1.dist-info"
    return set(WHEEL_FILES) | {
        f"{metadata}/METADATA",
        f"{metadata}/entry_points.txt",
        f"{metadata}/licenses/LICENSE",
    }


def test_release_contract_accepts_complete_archives(tmp_path: Path) -> None:
    sdist = tmp_path / "source.tar.gz"
    wheel = tmp_path / "package.whl"
    _sdist(sdist, set(SOURCE_FILES))
    _wheel(wheel, _wheel_files())
    assert check_sdist(sdist, "0.0.1") == []
    assert check_wheel(wheel) == []


def test_release_contract_rejects_missing_and_local_only_files(tmp_path: Path) -> None:
    sdist = tmp_path / "source.tar.gz"
    wheel = tmp_path / "package.whl"
    _sdist(sdist, set(SOURCE_FILES) - {"SECURITY.md"} | {".env"})
    _wheel(wheel, _wheel_files() - {"reachy_conscience/py.typed"} | {".venv/token"})
    source_errors = check_sdist(sdist, "0.0.1")
    wheel_errors = check_wheel(wheel)
    assert any("lacks release files" in error and "SECURITY.md" in error for error in source_errors)
    assert any("local-only paths" in error and ".env" in error for error in source_errors)
    assert any("lacks runtime files" in error and "py.typed" in error for error in wheel_errors)
    assert any("local-only paths" in error and ".venv" in error for error in wheel_errors)


def test_release_contract_rejects_source_links(tmp_path: Path) -> None:
    sdist = tmp_path / "source.tar.gz"
    wheel = tmp_path / "package.whl"
    _sdist(sdist, set(SOURCE_FILES), link="docs/shortcut")
    _wheel(wheel, _wheel_files(), link="reachy_conscience/assets/shortcut")
    assert any("links or special entries" in error for error in check_sdist(sdist, "0.0.1"))
    assert any("wheel has links" in error for error in check_wheel(wheel))
