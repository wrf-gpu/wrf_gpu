"""Bootstrap package for the wrf_gpu2 AgentOS."""

# Bind the version BEFORE the import hooks below run: the compile-cache hook
# version-keys its cache dir by ``gpuwrf.__version__`` (B1), so the attribute
# must already exist when ``configure_compilation_cache()`` runs at import. (It
# is re-exported via ``__all__`` at the foot of the module.)
__version__ = "0.22.0"

# ADR-002: the dynamical core + physics are validated in float64.  JAX defaults
# to float32 and SILENTLY canonicalises float64->float32 unless x64 is enabled
# BEFORE the first array is created.  Enabling it here, at top-level package
# import, guarantees every entry path (operational/real-case, idealized, tests)
# is genuinely fp64.  Previously x64 was enabled only as a side effect of
# importing certain submodules, so the operational/real-case path
# (daily_pipeline -> operational_mode) ran fp32 and silently defeated
# force_fp64 -- see Sprint U P0-1 and the GPT firm-rule confirm-close.
# The fp32/mixed-precision operational matrix is a separately-gated perf
# decision (ADR-007 / F7-perf), applied via explicit downcast, NOT by leaving
# x64 off.
#
# The compile-cache import hook below ALSO wires two ADDITIVE v0.13 compile-speed
# knobs: the persistent GPU autotune cache (GPUWRF_XLA_AUTOTUNE_CACHE) and the
# standalone parallel-compile knob (GPUWRF_XLA_PARALLEL_COMPILE). Both are
# HARD-SAFE -- they respect the platform pin and validate each --xla_gpu_* flag
# against the installed build in an isolated subprocess before injecting, so an
# unknown flag can NEVER abort import (the v0.12.0 GPU-abort regression guard).
# B1: the autotune cache is now ON BY DEFAULT whenever the persistent compile
# cache is on (the executable cache already covers the SAME graph; the autotune
# cache adds reuse across DIFFERENT-but-similar graphs); GPUWRF_XLA_AUTOTUNE_CACHE=0
# opts out. B2: the parallel-compile knob is ALSO ON BY DEFAULT whenever the
# compile cache is on (N-way parallel XLA compile at min(cpu_count,8) -- it slashes
# the cold nest-compile wall by overlapping the many independent fusion compiles);
# GPUWRF_XLA_PARALLEL_COMPILE=0 opts out. Both remain hard-safe via the subprocess
# flag-probe + platform-pin guard above, so default-on cannot abort import.
from gpuwrf._x64_config import configure_jax_x64 as _configure_jax_x64

_JAX_X64_FORCE_STATUS = _configure_jax_x64()

# Persistent JIT/XLA compilation cache (v0.12.0 first-run usability win): the
# v0.12.0 critique measured a ~4 min 55 s cold JIT compile on EVERY fresh
# process. JAX's persistent on-disk compilation cache turns that cold compile
# into a disk read on every subsequent run/process so a fresh user pays it once,
# not on every invocation. This is NUMERICS-NEUTRAL: the cache returns the
# identical XLA executable; no float op changes. Portable per-user default dir,
# VERSION-KEYED (B1) by gpuwrf+jax+jaxlib version + backend
# (``$HOME/.cache/gpuwrf/jit/<version-tag>``) so different versions/backends never
# stale-hit each other (the paid-B200-cold-compile failure mode); best-effort,
# silently no-op on GPUWRF_JAX_CACHE=0 or if the dir can't be created (logs a
# warning, never aborts). MUST run before the first compile (i.e. here at import,
# before any submodule builds a jitted fn).
from gpuwrf.runtime.compile_cache import configure_compilation_cache as _configure_jax_cache

_JAX_CACHE_STATUS = _configure_jax_cache()

__all__ = ["__version__"]
