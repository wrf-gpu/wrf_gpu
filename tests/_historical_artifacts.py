"""Availability checks for external, retained experiment evidence.

Only absence is a reason to skip. Existing evidence still has to pass every
checksum, schema, numerical, and source-authority assertion in its test.
"""

from pathlib import Path

import pytest


def require_historical(*paths: str | Path) -> None:
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        pytest.skip("Retained historical evidence unavailable: " + ", ".join(missing))
