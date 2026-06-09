# 2026-06-09 v0.14 Mythos Memory/FP32 Lane Review

- Verdict: `MYTHOS_MEMORY_LANE_CLOSED_MYNN_TILING_MATERIAL_FIX_R0_LANDED_REST_MEASURED_OR_EXACT_DEFER`
- Branch: `worker/mythos/v014-memory-fp32` @ `a32efce32852`
- Merge recommendation: `MERGE_NOW`

## What changed

- MYNN BouLac leading-column tiling (RRTMG pattern, default tile 16384,
  env-gated): measured compiled-temp cut 11.53 GiB at 641x321x50 and
  4.91 GiB at 313x313x50; GPU bit-identical incl. the ragged
  production-tile case; CPU tridiag solves carry 1-2 ulp batch-width
  SIMD codegen variance on scattered columns (turbulence bit-exact;
  CPU default paths stay structurally untiled below the tile width).
- Moisture transport velocity shared per RK stage (bit-identical;
  measured non-material 0.0 GiB - XLA CSE already deduped; hygiene).
- FP32 R0 precision-mode contract landed default-inert (fail-closed,
  cache-key split, 0 timestep consumers per regenerated static audit).
- Exact-branch GPU preflight green on this lineage; final-tree rerun
  peak compute VRAM 8116 MiB.

## What was deliberately NOT done (exact reasons in proof MD)

- Acoustic carry split, pad/mask helpers, FP32 R1/R2: the open one-RK-step
  fp64 P/PH/MU divergence owns that fault surface.
- State alias reduction: ADR-gated, small, high ABI risk.
- Limiter workspace and post-physics merge: measured/headroom-based defer.

## Manager follow-ups

1. Review + merge the three separated commits on worker/mythos/v014-memory-fp32.
2. Keep the MYNN tile width 16384 unless profiling motivates retune
   (GPUWRF_MYNN_COLUMN_TILE_COLS).
3. Re-run the exact-branch preflight on the post-merge trunk before the
   next long validation (it is cheap and the lineage changed).
4. Resume FP32 R1 only after the dynamics frontier closes.

