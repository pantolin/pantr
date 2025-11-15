#!/usr/bin/env python3
"""Wrapper script that conditionally fails based on branch name."""

import subprocess
import sys
from contextlib import suppress


def get_current_branch() -> str:
    """Get the current git branch name."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # If we can't determine the branch, default to strict (main behavior)
        return "main"
    return result.stdout.strip()


def main() -> int:
    """Run the command and conditionally fail based on branch."""
    if len(sys.argv) < 2:  # noqa: PLR2004
        print("Usage: conditional_check.py <command> [args...]", file=sys.stderr)
        return 1

    branch = get_current_branch()
    is_main = branch == "main"

    # Handle mypy specially - call as module if command is 'mypy'
    cmd = sys.argv[1:]
    if cmd[0] == "mypy":
        # Try as module first, fallback to direct call
        with suppress(Exception):
            cmd = [sys.executable, "-m", "mypy", *cmd[1:]]

    # Run the actual command
    result = subprocess.run(cmd, check=False)

    # If on main, fail if command failed
    # If not on main, always succeed (but show output)
    if is_main:
        return result.returncode
    else:
        if result.returncode != 0:
            print(
                f"\n⚠️  Check failed on branch '{branch}' but allowing commit "
                "(only 'main' branch enforces strict checks)",
                file=sys.stderr,
            )
        return 0


if __name__ == "__main__":
    sys.exit(main())
