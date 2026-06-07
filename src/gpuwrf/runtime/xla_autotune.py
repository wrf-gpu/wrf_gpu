"""Persistent XLA GPU autotuning cache + parallel-compile wiring (v0.13 #2/#4).

WHY
---
XLA AOT-compiles the whole jitted timestep graph before the first run, and a
large chunk of that wall-clock for the GPU backend is **autotuning**: XLA runs
candidate GEMM / convolution kernels at compile time to pick the fastest, which
is heavier in fp64 (fp64 is forced operationally; fp32 detonates the acoustic
solver, so the fp64-inherent autotune cost stays). The v0.13 speed roadmap
(#2) calls for *persisting* those autotune results to disk so even a new-but-
similar graph reuses the tuned kernels instead of re-running the autotuner.

This is the autotune-cache analogue of :mod:`gpuwrf.runtime.compile_cache`
(which persists the *compiled executable*). The two are complementary:

* compile_cache  -> warm hit  = identical HLO + backend + flags (whole-program
  executable served from disk).
* this module     -> warm hit = same fusion's tuned kernel choice reused even
  when the surrounding program differs, so partial / first-of-a-family compiles
  still skip the autotuner.

NUMERICS
--------
**Numerically inert.** Autotuning only selects *which* kernel implementation
computes a GEMM/conv; persisting that choice changes nothing about the math.
The autotune *level* (``--xla_gpu_autotune_level``) is left at the XLA default
(4 = on+init+reinit+check) for production so the picked kernels are still the
fastest correct ones; only the *caching* of the result is added here. A
separate dev-only fast-compile knob (roadmap #5) may lower the level, but that
is opt-in and NOT enabled by this module.

HOW
---
XLA debug-options flags are parsed **once**, the first time XLA initialises, out
of the ``XLA_FLAGS`` environment string. They are not re-read per compile and
the JAX ``config.update`` mechanism does NOT cover these GPU-plugin flags. So we
must inject them into ``XLA_FLAGS`` **before the first compile** -- i.e. at
package import, from the same import hook that enables x64 and the compile
cache. This module appends our flags to any ``XLA_FLAGS`` the operator already
exported (we never clobber operator flags; we only add ones not already set).

PLATFORM GUARD
--------------
These are GPU-plugin flags (``--xla_gpu_*``). They are valid only when a CUDA
backend is loaded; on a CPU-only jaxlib build XLA logs ``unknown flag`` and (in
older builds) can abort init. We therefore inject them ONLY when a GPU backend
is the likely target, detected WITHOUT importing/initialising JAX (importing
jax here would lock the flags in before we set them). Detection precedence:

1. ``GPUWRF_XLA_AUTOTUNE_CACHE=0`` -> fully disabled (no-op).
2. ``JAX_PLATFORMS`` / ``JAX_PLATFORM_NAME`` naming ``cpu`` first -> CPU run,
   skip (the dev/CI box is CPU-only; these proofs run there).
3. A CUDA device is visible (``CUDA_VISIBLE_DEVICES`` non-empty-and-not "-1",
   or an ``nvidia`` device node exists) -> inject.
4. ``GPUWRF_XLA_AUTOTUNE_CACHE=1`` (explicit force) -> inject regardless.

Best-effort: any failure leaves ``XLA_FLAGS`` untouched and never raises at
import.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "AUTOTUNE_STATUS",
    "configure_autotune_cache",
    "resolve_autotune_cache_dir",
    "autotune_env_help",
]

_DISABLE_VALUES = {"0", "false", "off", "no"}
_FORCE_VALUES = {"1", "true", "on", "yes", "force"}

# Record of what was configured, for audit / proof scripts.
AUTOTUNE_STATUS: dict[str, object] = {
    "enabled": False,
    "dir": None,
    "mode": None,
    "parallelism": None,
    "injected_flags": None,
    "source": None,
    "reason": None,
    "error": None,
}


def resolve_autotune_cache_dir() -> Path | None:
    """Return the autotune-cache directory, or ``None`` if disabled.

    Mirrors :func:`gpuwrf.runtime.compile_cache.resolve_cache_dir` precedence so
    the two caches sit side by side under one project cache root. Pure apart
    from recording ``AUTOTUNE_STATUS['source']``.
    """
    flag = os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE", "").strip().lower()
    if flag in _DISABLE_VALUES:
        AUTOTUNE_STATUS["source"] = "disabled-by-GPUWRF_XLA_AUTOTUNE_CACHE"
        return None

    explicit = os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE_DIR", "").strip()
    if explicit:
        AUTOTUNE_STATUS["source"] = "env:GPUWRF_XLA_AUTOTUNE_CACHE_DIR"
        return Path(explicit).expanduser()

    root = os.environ.get("GPUWRF_CACHE", "").strip()
    if root:
        AUTOTUNE_STATUS["source"] = "env:GPUWRF_CACHE"
        return Path(root).expanduser() / "autotune"

    # Sit beside the JIT cache under the same per-user default root.
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else (Path.home() / ".cache")
    AUTOTUNE_STATUS["source"] = "default"
    return base / "gpuwrf" / "autotune"


def _gpu_is_target() -> tuple[bool, str]:
    """Decide whether a GPU backend is the target, WITHOUT importing jax.

    Returns ``(is_gpu, reason)``. Importing jax here would freeze ``XLA_FLAGS``
    before we add ours, so detection is purely environmental.
    """
    forced = os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE", "").strip().lower() in _FORCE_VALUES

    # Explicit platform pin takes precedence even over force: a CPU/TPU pin means
    # the --xla_gpu_* flags are unknown to the active backend, and on a CPU-only
    # jaxlib build XLA *fatally aborts* on an unknown flag (verified). So never
    # inject when the operator pinned cpu/tpu, even with force=1.
    platforms = (
        os.environ.get("JAX_PLATFORMS", "")
        or os.environ.get("JAX_PLATFORM_NAME", "")
    ).strip().lower()
    if platforms:
        first = platforms.split(",")[0].strip()
        if first in ("cpu", "tpu"):
            return False, f"platform-pinned:{first}"
        if first in ("cuda", "gpu", "rocm"):
            return True, f"platform-pinned:{first}"

    if forced:
        return True, "forced-by-GPUWRF_XLA_AUTOTUNE_CACHE"

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cvd = cvd.strip()
        if cvd == "" or cvd == "-1":
            return False, "CUDA_VISIBLE_DEVICES-empty"
        return True, "CUDA_VISIBLE_DEVICES-set"

    # Fall back to probing for an nvidia device node (cheap, no jax import).
    for node in ("/dev/nvidia0", "/dev/nvidiactl"):
        if os.path.exists(node):
            return True, "nvidia-device-node"
    return False, "no-gpu-detected"


def _parse_existing_flags(xla_flags: str) -> set[str]:
    """Return the set of bare flag names already present in ``XLA_FLAGS``."""
    names: set[str] = set()
    for tok in xla_flags.split():
        if tok.startswith("--"):
            names.add(tok[2:].split("=", 1)[0])
    return names


def configure_autotune_cache() -> dict[str, object]:
    """Inject persistent-autotune-cache + parallel-compile flags into XLA_FLAGS.

    Must be called BEFORE the first JAX/XLA compile (i.e. at package import,
    before any jitted fn is built). Idempotent, best-effort, never raises.
    Returns :data:`AUTOTUNE_STATUS`.
    """
    # Reset to a clean slate so a re-call reflects the CURRENT environment (the
    # status dict is module-global; a prior enable must not leak into a later
    # disabled/skip call).
    AUTOTUNE_STATUS.update(
        {
            "enabled": False,
            "dir": None,
            "mode": None,
            "parallelism": None,
            "injected_flags": None,
            "source": None,
            "reason": None,
            "error": None,
        }
    )

    cache_dir = resolve_autotune_cache_dir()
    if cache_dir is None:
        AUTOTUNE_STATUS["reason"] = "disabled"
        return AUTOTUNE_STATUS

    is_gpu, reason = _gpu_is_target()
    AUTOTUNE_STATUS["reason"] = reason
    if not is_gpu:
        # CPU/TPU run: these --xla_gpu_* flags are unknown to the backend and
        # would log noise (or abort on old builds). Skip silently; the JIT
        # compile cache (compile_cache.py) still applies on CPU.
        return AUTOTUNE_STATUS

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        AUTOTUNE_STATUS["error"] = f"mkdir failed: {exc}"
        return AUTOTUNE_STATUS

    existing_raw = os.environ.get("XLA_FLAGS", "")
    existing = _parse_existing_flags(existing_raw)

    # Mode: "update" reads existing results AND writes newly-tuned ones back, so
    # the cache fills incrementally and is reused across runs. (The alternative
    # "read" never updates; "update" is the steady-state-warming choice.)
    mode = os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE_MODE", "update").strip() or "update"

    candidate: list[tuple[str, str]] = [
        # Per-fusion autotune cache directory: persists each fusion's tuned
        # kernel choice keyed by the fusion's HLO, so a new program reusing the
        # same fusion skips the autotuner. This is the durable cross-run cache.
        ("xla_gpu_per_fusion_autotune_cache_dir", str(cache_dir)),
        # Cache mode: update = load existing + write new entries back.
        ("xla_gpu_experimental_autotune_cache_mode", mode),
    ]

    # Optional whole-results dump/load (a single results proto). Off by default
    # because the per-fusion dir above is the primary durable cache; enable via
    # env for cross-machine transfer or debugging.
    results_file = os.environ.get("GPUWRF_XLA_AUTOTUNE_RESULTS_FILE", "").strip()
    if results_file:
        candidate.append(("xla_gpu_dump_autotune_results_to", results_file))
        if os.path.exists(results_file):
            candidate.append(("xla_gpu_load_autotune_results_from", results_file))

    # Parallel compilation (roadmap #4): parallelise kernel compiles + autotune
    # across host cores. 0 = let XLA decide; an explicit count caps it (respect
    # the project core budget). Only added when the operator opts in.
    par = os.environ.get("GPUWRF_XLA_COMPILE_PARALLELISM", "").strip()
    parallelism: int | None = None
    if par:
        try:
            parallelism = int(par)
            candidate.append(("xla_gpu_force_compilation_parallelism", str(parallelism)))
        except ValueError:
            AUTOTUNE_STATUS["error"] = f"bad GPUWRF_XLA_COMPILE_PARALLELISM={par!r}"

    # Only inject flags the operator has not already set (never clobber).
    injected = [f"--{name}={value}" for name, value in candidate if name not in existing]
    if injected:
        new_flags = (existing_raw + " " + " ".join(injected)).strip()
        os.environ["XLA_FLAGS"] = new_flags

    AUTOTUNE_STATUS["enabled"] = True
    AUTOTUNE_STATUS["dir"] = str(cache_dir)
    AUTOTUNE_STATUS["mode"] = mode
    AUTOTUNE_STATUS["parallelism"] = parallelism
    AUTOTUNE_STATUS["injected_flags"] = injected
    return AUTOTUNE_STATUS


def autotune_env_help() -> str:
    """One-line human summary of the env vars for ``--help`` / docs / logs."""
    return (
        "Persistent XLA autotune cache env vars (GPU only): "
        "GPUWRF_XLA_AUTOTUNE_CACHE=0 disables / =1 forces; "
        "GPUWRF_XLA_AUTOTUNE_CACHE_DIR sets the dir (default $GPUWRF_CACHE/autotune "
        "or $HOME/.cache/gpuwrf/autotune); "
        "GPUWRF_XLA_AUTOTUNE_CACHE_MODE=update|read; "
        "GPUWRF_XLA_AUTOTUNE_RESULTS_FILE adds dump/load of a single results proto; "
        "GPUWRF_XLA_COMPILE_PARALLELISM=N enables N-way parallel compile."
    )
