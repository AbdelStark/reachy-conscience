"""Check that a built source release and wheel contain the public install contract.

Run after ``uv build --out-dir dist --clear --no-build-logs``. This checks archive
contents, not the behavior of optional runtimes or a connected robot.
"""

from __future__ import annotations

import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from stat import S_IFLNK, S_IFMT

SOURCE_FILES = frozenset(
    {
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "pyproject.toml",
        "uv.lock",
        "docs/local-console.png",
        "examples/guarded_turn.py",
        "examples/owned_voice_session.py",
        "scripts/check_reachy_sdk_api.py",
        "scripts/check_typesafe_sdk_api.py",
        "scripts/check_release_artifacts.py",
        "tests/test_guard.py",
        "tests/test_pipeline.py",
        "tests/test_release_artifacts.py",
        "tests/browser_session.test.mjs",
        "src/reachy_conscience/py.typed",
        "src/reachy_conscience/assets/index.html",
        "src/reachy_conscience/assets/app.js",
        "src/reachy_conscience/assets/style.css",
        "src/reachy_conscience/assets/redteam.json",
    }
)
WHEEL_FILES = frozenset(
    {
        "reachy_conscience/__init__.py",
        "reachy_conscience/pipeline.py",
        "reachy_conscience/owner_console_cli.py",
        "reachy_conscience/voice_host.py",
        "reachy_conscience/py.typed",
        "reachy_conscience/assets/index.html",
        "reachy_conscience/assets/app.js",
        "reachy_conscience/assets/style.css",
        "reachy_conscience/assets/redteam.json",
    }
)
FORBIDDEN_PARTS = frozenset({".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist"})
FORBIDDEN_NAMES = frozenset({".env", ".env.local", "id_rsa", "id_ed25519"})


def _unsafe_paths(paths: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if path.startswith("/")
        or any(part in {"", ".", ".."} or part in FORBIDDEN_PARTS for part in path.split("/"))
        or Path(path).name.lower() in FORBIDDEN_NAMES
    ]


def check_sdist(path: Path, version: str) -> list[str]:
    errors: list[str] = []
    archive_root = f"reachy_conscience-{version}"
    prefix = f"{archive_root}/"
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        files = [member for member in members if member.isfile()]
        others = [member.name for member in members if not member.isfile() and not member.isdir()]
    names = [member.name for member in files]
    member_names = [member.name.rstrip("/") for member in members]
    if others:
        errors.append(f"source archive has links or special entries: {others}")
    if any(name != archive_root and not name.startswith(prefix) for name in member_names):
        errors.append("source archive contains an entry outside its versioned root")
    relative = [name[len(prefix) :] for name in names if name.startswith(prefix)]
    all_relative = [name[len(prefix) :] for name in member_names if name.startswith(prefix)]
    if len(relative) != len(set(relative)):
        errors.append("source archive has duplicate file names")
    if unsafe := _unsafe_paths(all_relative):
        errors.append(f"source archive has unsafe or local-only paths: {unsafe}")
    if missing := sorted(SOURCE_FILES - set(relative)):
        errors.append(f"source archive lacks release files: {missing}")
    return errors


def check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
    all_names = [entry.filename.rstrip("/") for entry in entries]
    names = [entry.filename for entry in entries if not entry.is_dir()]
    if len(all_names) != len(set(all_names)):
        errors.append("wheel has duplicate file names")
    if unsafe := _unsafe_paths(all_names):
        errors.append(f"wheel has unsafe or local-only paths: {unsafe}")
    if links := [entry.filename for entry in entries if S_IFMT(entry.external_attr >> 16) == S_IFLNK]:
        errors.append(f"wheel has links: {links}")
    if missing := sorted(WHEEL_FILES - set(names)):
        errors.append(f"wheel lacks runtime files: {missing}")
    metadata_dirs = {name.split("/")[0] for name in names if ".dist-info/" in name}
    if len(metadata_dirs) != 1:
        errors.append("wheel must have exactly one dist-info directory")
    else:
        metadata = metadata_dirs.pop()
        if missing := sorted(
            {f"{metadata}/METADATA", f"{metadata}/entry_points.txt", f"{metadata}/licenses/LICENSE"}
            - set(names)
        ):
            errors.append(f"wheel lacks metadata or license files: {missing}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    dist = root / "dist"
    sdist = dist / f"reachy_conscience-{version}.tar.gz"
    wheel = dist / f"reachy_conscience-{version}-py3-none-any.whl"
    if not sdist.is_file() or not wheel.is_file():
        print("FAIL: build the source archive and wheel before checking them", file=sys.stderr)
        return 1
    errors = check_sdist(sdist, version) + check_wheel(wheel)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {sdist.name} and {wheel.name} include the release contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
