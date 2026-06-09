# Reviewer Report

## Findings

- The MYNN tiling mirrors the proven RRTMG leading-column pattern (pad by
  repeating the last column, `lax.scan` over fixed tiles, scatter into a
  zero-initialized accumulator, slice to `ncol`). The tiled path engages only
  for flat `(B, nz)` batches with `B > tile`, so every small-batch test and
  the analytic fixtures run the unchanged whole-batch path.
- Cross-column independence of the MYNN kernel was checked at source level
  (all reductions over the level axis; the EDMF plume sum runs inside a
  per-column vmap) and proven empirically: bit-identical on GPU (incl. the
  ragged production-tile case); on CPU the tridiagonal solves carry 1-2 ulp
  batch-width SIMD codegen differences on scattered columns (turbulence chain
  bit-exact, ruling out coupling; fresh-cache discriminating run).
- The shared transport-velocity wiring keeps the internal-build fallback, so
  standalone callers (`_tiedtke_qvften_from_flux_advection`, tests) are
  unchanged; the per-stage build is shared only inside `advance_stage`.
- The R0 field rides in static aux; grep confirms zero timestep consumers
  (regenerated static audit). Unknown modes fail closed at construction.
- The preflight harness change is allowlist-only (resident Hermes bridge),
  justified because the baseline VRAM sample already accounts for it.

## Contract Compliance

- No dynamics/**, state-contract, boundary, restart, or wrfout edits.
- No clamps/masks/tolerance widening; all layout claims proven by exact
  equality, all memory claims by compiled-memory or nvidia-smi artifacts.
- GPU used only through `scripts/run_gpu_lowprio.sh`, one job at a time.
- No TOST, no Switzerland, no long validation, no Hermes messages.
- Work isolated to the mythos worktree/branch.

## Correctness Risks

- Tiling changes XLA's scheduling for large batches; values proven identical,
  but the first post-merge preflight on the merged trunk is still recommended
  (cheap) because the merge target may differ from this branch tip.
- One-time JIT cache invalidation from the static-aux extension.

## Performance Risks

- Tile width 16384 inherited from RRTMG; MYNN-specific retune unmeasured.
  The scan serializes tile execution, but the measured nested preflight shows
  NO regression: baseline 465 s -> 933 s first post-change run (pure JIT
  recompilation from the changed programs/cache keys) -> 378 s warm-cache
  rerun (faster than baseline; identical peak 8116 MiB). One-time
  recompilation cost after merge is expected and benign.

## Required Fixes

- None blocking. Optional later: MYNN tile-width tuning measurement; vectorized
  velocity-reuse claim removal from older static maps (done in roadmap update).

## Decision

Decision: APPROVE for manager merge review (`MERGE_NOW` recommendation from
the proof object; manager remains the merge authority).
