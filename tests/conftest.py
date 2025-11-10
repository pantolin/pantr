"""Pytest configuration to make `src` importable without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_sys_path() -> None:
    """Prepend the repository `src` directory to `sys.path` if missing."""
    repo_root: Path = Path(__file__).resolve().parents[1]
    src_path: Path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_sys_path()
