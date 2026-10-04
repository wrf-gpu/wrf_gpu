"""FORCE this process to CPU before JAX loads. Import FIRST, before jax or gpuwrf.

Contract §13: no GPU import/query/compile/run without dual-manager coordination.

Why this module forces rather than defaults
-------------------------------------------
The first version used `os.environ.setdefault`, reasoning that a deliberately
coordinated GPU run sets `JAX_PLATFORMS` itself and the guard should not override
it. The manager identified that as **fail-open** (main `965fe281`), and it is: a
CPU-only generator inheriting `JAX_PLATFORMS=cuda` from the caller's shell would
keep it, import JAX against the GPU, and the guard would have silently approved
the thing it exists to prevent. The dangerous direction was the unprotected one.

The correct split is by *entry point*, not by environment:

* **CPU-only entry points** import this module. It **forces** CPU, and if forcing
  is impossible it **refuses** — never proceeds.
* **Coordinated GPU entry points** (`run_gpu_arm.py`, `step1_driver.py`) do
  **not** import this module at all. They enforce §13 through the single-use
  receipt and the canonical lock, which is a stronger check than an environment
  variable, and importing the guard would only make their intent ambiguous.

So `JAX_PLATFORMS=cuda` in the environment is not an override to be honoured
here; in a CPU-only entry point it is a mistake, and it is corrected loudly.
"""

from __future__ import annotations

import os
import sys

#: Set by :func:`force_cpu` at import; records what the caller had asked for.
INHERITED: dict[str, str | None] = {}

_GPU_PLATFORMS = {"cuda", "gpu", "rocm", "tpu"}


class GpuPlatformRefused(RuntimeError):
    """A CPU-only entry point was asked to run on an accelerator."""


def _jax_already_imported() -> bool:
    """After `import jax`, changing these variables has no effect."""
    return "jax" in sys.modules or "jaxlib" in sys.modules


def force_cpu(*, strict: bool = True) -> dict[str, str | None]:
    """Force CPU-only execution. Returns what was inherited, for the record.

    Raises when the situation cannot be made safe rather than continuing:

    * JAX is already imported, so the variables can no longer take effect;
    * ``strict`` and the caller explicitly requested an accelerator platform —
      refused loudly, because silently rewriting an explicit request would hide a
      real configuration error from whoever set it.
    """
    inherited = {
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    INHERITED.clear()
    INHERITED.update(inherited)

    if _jax_already_imported():
        raise GpuPlatformRefused(
            "cpu_guard was imported AFTER jax/jaxlib. JAX reads JAX_PLATFORMS and "
            "CUDA_VISIBLE_DEVICES at import, so forcing them now would be a no-op that "
            "looks like protection. Import cpu_guard before any jax/gpuwrf import."
        )

    requested = (inherited["JAX_PLATFORMS"] or "").strip().lower()
    if strict and requested and any(p in _GPU_PLATFORMS for p in requested.split(",")):
        raise GpuPlatformRefused(
            f"JAX_PLATFORMS={inherited['JAX_PLATFORMS']!r} requests an accelerator, but this is "
            f"a CPU-only entry point under contract §13. Refusing rather than silently "
            f"overriding, so the caller's configuration error is visible. Coordinated GPU work "
            f"runs through run_gpu_arm.py with a single-use receipt and the canonical lock."
        )

    # FORCE, not setdefault: an inherited value must not survive.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    return inherited


def platform_is_pinned_to_cpu() -> bool:
    """True when this process cannot reach a GPU through JAX."""
    return (os.environ.get("JAX_PLATFORMS") == "cpu"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "")


force_cpu()
