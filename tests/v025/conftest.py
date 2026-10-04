"""Pin the whole v025 suite to CPU before any test module imports JAX.

This exists because of a real incident, not as a precaution. `test_step1_driver`
imported `gpuwrf.cli` to parse-check a generated command; that pulled in JAX,
which probed the CUDA driver and logged `cuda_executor.cc: Could not get kernel
mode driver version`. The v025 sprint is under a CPU-only constraint, so an
import that reaches the driver is a contract breach even though it allocates
nothing.

The generators under `scripts/v025/` pin these variables at their own import, so
they were never the exposure. Test modules that import production code directly
were, and nothing covered them: `tests/conftest.py` sets no platform.

pytest imports a directory's `conftest.py` before collecting its test modules,
so assigning here happens before any `import jax` in this package.
"""

from __future__ import annotations

import os

# Set, not defaulted: a stale JAX_PLATFORMS=cuda in the caller's environment
# must not leak into a CPU-only test run.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
