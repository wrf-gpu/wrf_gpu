"""CuPy raw-CUDA M2 bakeoff backend."""

from __future__ import annotations

from .column import run_column
from .stencil import run_stencil

__all__ = ["run_column", "run_stencil"]
