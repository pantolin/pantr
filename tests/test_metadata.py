"""Smoke tests for package metadata.

Validates public attributes exposed via the package API.
"""

from __future__ import annotations

import importlib
from typing import Final

import pantr


def test_package_all_exports() -> None:
    """Ensure all expected symbols are exported."""
    expected: Final[set[str]] = {"__version__", "__license__", "__author__"}
    assert set(pantr.__all__) == expected


def test_package_metadata_values() -> None:
    """Validate the package metadata constants."""
    assert pantr.__version__ == "0.1.0"
    assert pantr.__license__ == "MIT"
    assert pantr.__author__ == "Pablo Antolin <pablo.antolin@epfl.ch>"


def test_metadata_import_stability() -> None:
    """Verify metadata survives module reloads."""
    module = importlib.reload(pantr)
    assert module.__version__ == "0.1.0"
