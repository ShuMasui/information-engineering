"""実験を再現するための実行環境の情報 (どのコードを、どの環境で動かしたか)。"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

_PACKAGE_DIR = Path(__file__).resolve().parent


def collect_provenance() -> dict[str, Any]:
    return {
        "package_version": _package_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git": _git_info(),
    }


def _package_version() -> str | None:
    try:
        return metadata.version("lights-out-rl")
    except metadata.PackageNotFoundError:
        return None


def _git_info() -> dict[str, Any] | None:
    """このコードがあるリポジトリのコミットと未コミットの変更の有無。git が使えなければ None。"""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": dirty}


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(_PACKAGE_DIR), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
