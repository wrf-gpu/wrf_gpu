"""Fused Pallas kernel modules (M2 kernel-granularity collapse).

New modules under this package are standalone: they never edit existing
``gpuwrf`` operators. Each module documents its integration seam (the exact
production call site it would replace) and is adoption-gated by a pre-registered
device bake-off harness under ``scripts/v025/``.
"""

from gpuwrf.kernels import fused_vertical_implicit

__all__ = ["fused_vertical_implicit"]
