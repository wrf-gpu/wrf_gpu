"""Bootstrap package for the wrf_gpu2 AgentOS."""

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
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

# v0.2.0 wall-clock win #1: enable JAX's persistent on-disk compilation cache so
# repeat runs (daily forecasts AND our own re-validation) read the compiled XLA
# executable from disk instead of recompiling cold (~40%% of d02 24h wall, ~80-90%%
# of short jobs per .agent/reviews/2026-06-01-gpt-wallclock-optimization.md). This
# is NUMERICS-NEUTRAL: the cache returns the identical executable; no float op
# changes. Best-effort and silently no-op if /mnt/data is unavailable or
# GPUWRF_JAX_CACHE=0 is set. Must run before the first compile (i.e. at import).
from gpuwrf.runtime.jax_cache import configure_jax_compilation_cache as _configure_jax_cache

_JAX_CACHE_STATUS = _configure_jax_cache()

__all__ = ["__version__"]

__version__ = "0.0.0"
