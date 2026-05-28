"""Public test collection policy.

The historical sprint suite contains tests that require private reference
assets, generated WRF savepoints, optional CUDA backends, or scripts that are
not part of the public v0.0.1 source snapshot. Public installs should still be
able to run the portable CPU tests with ``pytest -q tests/ -k 'not gpu'``.
Set ``WRF_GPU_FULL_TEST_COLLECTION=1`` in a development checkout that has the
external assets if you want pytest to collect those historical tests.
"""

from __future__ import annotations

import os
from pathlib import Path


PUBLIC_KEEP_FILES = {
    "test_m7_honest_speedup.py",
    "test_m7_profiler_window.py",
    "test_m7_rca_helpers.py",
    "test_m7_wrfout_io_compat.py",
}


def pytest_ignore_collect(collection_path, config):  # type: ignore[no-untyped-def]
    if os.environ.get("WRF_GPU_FULL_TEST_COLLECTION") == "1":
        return None
    path = Path(str(collection_path))
    name = path.name
    if name.startswith("test_") and name not in PUBLIC_KEEP_FILES:
        return True
    return None
