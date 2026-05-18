#!/usr/bin/env python3
"""Print a small JSON snapshot of git repository state."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, (proc.stdout or proc.stderr).strip()


def main() -> int:
    is_repo, _ = git(["rev-parse", "--is-inside-work-tree"])
    if is_repo != 0:
        print(json.dumps({"ok": False, "git": "not a repository"}, indent=2))
        return 0

    _, branch = git(["branch", "--show-current"])
    _, status = git(["status", "--short"])
    _, recent = git(["log", "--oneline", "-5"])
    _, remote = git(["remote", "-v"])
    dirty_files = [line for line in status.splitlines() if line.strip()]
    result = {
        "ok": True,
        "branch": branch,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "recent_commits": recent.splitlines() if recent else [],
        "remotes": remote.splitlines() if remote else [],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
