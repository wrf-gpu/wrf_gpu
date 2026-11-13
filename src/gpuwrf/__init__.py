"""Bootstrap package for the wrf_gpu2 AgentOS."""

# Bind the version BEFORE the import hooks below run: the compile-cache hook
# version-keys its cache dir by ``gpuwrf.__version__`` (B1), so the attribute
# must already exist when ``configure_compilation_cache()`` runs at import. (It
# is re-exported via ``__all__`` at the foot of the module.)
__version__ = "0.5.0"

# Offline entry points must work without loading JAX or configuring a backend.
import sys as _sys

_OFFLINE_COMMAND = len(_sys.argv) > 1 and _sys.argv[1] in {"doctor", "manifest", "reproduce", "ens", "export", "optimize", "calibrate", "stream-check", "perturb", "cycle", "override"}
# `gpuwrf da check|analyse` are host-only; `da cycle|members` run forecasts in-process and need the full bootstrap.
_OFFLINE_COMMAND = _OFFLINE_COMMAND or _sys.argv[1:3] in (["da", "check"], ["da", "analyse"])

# Defaults are pure environment setup, also needed by offline support reporting.
from gpuwrf._fast_defaults import apply_fast_path_defaults as _apply_fast_path_defaults

_FAST_DEFAULTS_STATUS = _apply_fast_path_defaults()

# Experimental legacy radiation is CPU-only. Inspect its CLI selection before
# the existing package bootstrap imports/configures JAX or touches a backend.
from gpuwrf.io.flg_gfdl_config import bootstrap_refusal as _legacy_radiation_refusal
_legacy_refusal = _legacy_radiation_refusal(_sys.argv)
if _legacy_refusal:
    print('gpuwrf: error: '+_legacy_refusal,file=_sys.stderr)
    raise SystemExit(2)

from gpuwrf.io.cammg_config import bootstrap_refusal as _cammg_bootstrap_refusal
_cammg_refusal = _cammg_bootstrap_refusal(_sys.argv)
if _cammg_refusal:
    print('gpuwrf: error: '+_cammg_refusal, file=_sys.stderr)
    raise SystemExit(2)

if not _OFFLINE_COMMAND:
    # v0.3 release defaults: the validated fp32 fast paths are ON unless opted out
    # (explicit =0 per flag, or GPUWRF_FAST_DEFAULTS=0 for all). MUST run first:
    # submodules and the import hooks below read these switches at import time.
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

    # Keep launch/profiler command-buffer settings together, before backend init.
    # Dynamic slice copies stay in GPU kernels; capture policy has its own executable
    # identity and is qualified by the release's actual compiled/runtime gates.
    from gpuwrf.runtime.xla_autotune import configure_command_buffers as _configure_command_buffers

    _COMMAND_BUFFER_STATUS = _configure_command_buffers(default_on=True)

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
