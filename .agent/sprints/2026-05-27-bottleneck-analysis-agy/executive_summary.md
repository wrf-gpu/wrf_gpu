GPU 24h Canary 3km forecast achieves 22.26x speedup (732.63s wall-clock) with zero loop D2H copies (ADR-027 verified).
cuSPARSE `pcrGtsvBatchSharedMemKernelLoop` (MYNN tridiagonal) dominates GPU time (~56%), while elementwise and JAX-native Thomas scan reverse operations (`loop_reverse_fusion`) dominate kernel launch counts.
High-risk FP64 variables (`ph`, `mu`, `p`, `w`) must stay FP64 to protect geopotential PGF gradients and mass continuity; 2D surface fluxes can safely drop to FP32.
VRAM peaks at 7.28 GB for 1km domain due to XLA intermediate allocations, compile memory workspace, and BFC pre-allocation; mitigate via `JAX_MEM_FRACTION=0.4` and input array donation.
Reduce 106s cold start via `JAX_COMPILATION_CACHE_DIR` disk cache and AOT compilation, and prepare multi-GPU scaling using `jax.experimental.shard_map` for halo exchange.
