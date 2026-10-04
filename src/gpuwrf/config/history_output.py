"""Shared selection of the WRF history stream and its resident diagnostics."""
from __future__ import annotations

import os


def full_wrfout_variables_enabled() -> bool:
    """Full WRF history by default; the canonical explicit setting wins."""
    for name in ("GPUWRF_FULL_WRFOUT_VARIABLES", "GPUWRF_FULL_WRFOUT"):
        value = os.environ.get(name, "").strip().lower()
        if value in {"0", "false", "no", "off"}:
            return False
        if value in {"1", "true", "yes", "on"}:
            return True
    return True
