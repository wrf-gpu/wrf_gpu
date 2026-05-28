"""Repository path helpers for public installs.

The public package must not assume the development workstation's data layout.
Reference data defaults to a repo-relative ``reference_data/`` directory unless
``WRF_GPU_REFERENCE_ROOT`` is set.
"""

from __future__ import annotations

import os
from pathlib import Path


REFERENCE_ROOT_ENV = "WRF_GPU_REFERENCE_ROOT"
CACHE_ROOT_ENV = "WRF_GPU_CACHE_ROOT"


def reference_root() -> Path:
    """Return the configured reference-data root."""

    return Path(os.environ.get(REFERENCE_ROOT_ENV, "reference_data")).expanduser()


def reference_path(*parts: str) -> Path:
    """Return a path below the configured reference-data root."""

    return reference_root().joinpath(*parts)


def cache_root() -> Path:
    """Return the repo-local runtime cache root used by lightweight scripts."""

    return Path(os.environ.get(CACHE_ROOT_ENV, ".cache/gpuwrf")).expanduser()
