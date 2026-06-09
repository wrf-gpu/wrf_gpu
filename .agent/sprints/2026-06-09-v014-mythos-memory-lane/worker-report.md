# Worker Report

## Summary

Summary: Closed the v0.14 memory/FP32 lane on `worker/mythos/v014-memory-fp32`
(base `a32efce3`). Measured-material fix: MYNN BouLac leading-column tiling
(RRTMG pattern) cuts compiled GPU temp memory by 11.53 GiB at 641x321x50 and
4.91 GiB at a d03-like 313x313x50; tile-vs-untiled is bit-identical on GPU
(including a ragged production-tile case), while on the CPU backend the four
implicit tridiagonal-solve outputs can differ by 1-2 ulp on scattered columns
from batch-width-dependent SIMD codegen (all turbulence/diagnostic fields
bit-exact; fresh-cache discriminating run; CPU test batches stay below the
tile width so the CPU default path is structurally untiled). Bit-identical
hygiene fix: the per-RK-stage transport-velocity build is now shared between
theta/momentum and moisture flux advection; measured GPU temp delta is 0.0 GiB
(XLA CSE had already deduplicated it), so the roadmap's 0.45-0.65 GiB estimate
is reclassified non-material-measured. FP32 R0 precision-mode contract landed
default-inert (fail-closed, separate JIT cache key, 0 timestep consumers);
R1/R2 carry an exact blocker: the open one-RK-step fp64 P/PH/MU divergence
owns the same acoustic fault surface and no WRF-anchored mixed-precision gate
can pass until it closes. Exact-branch nested GPU preflight: baseline
a32efce3 peak compute 8169 MiB / 465 s; final tree 8116 MiB peak, 933 s cold
(JIT recompiles from changed programs) and 378 s warm-cache (faster than
baseline, PIPELINE_GREEN, all finite both runs). Remaining roadmap rows are
closed as measured-defer / non-material / ADR-gated with quantified reasons.

## Files Changed

- `src/gpuwrf/physics/mynn_pbl.py` — leading-column tiling (default tile
  16384, env-gated; whole-batch reference path retained)
- `src/gpuwrf/runtime/operational_mode.py` — `_stage_transport_velocities`
  shared per RK stage; R0 `acoustic_precision_mode` static-aux field
  (default-inert, fail-closed)
- `src/gpuwrf/contracts/precision.py` — `AcousticPrecisionMode` labels (R0)
- `tests/test_operational_namelist_cache_key.py` — 5-test cache-key suite
- `proofs/v014/exact_branch_memory_preflight.py` — resident-bridge allowlist
- `proofs/v014/mythos_memory_fixes_260609.py` (+ generated json/md/review)
- `proofs/v014/mythos_memory_gpu_suite_260609.py` (+ json)
- `proofs/v014/fp32_acoustic_static_audit.py` (+ json, regenerated here)
- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`,
  `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md` — update blocks
- Baseline preflight artifacts copied to
  `proofs/v014/exact_branch_memory_preflight_baseline_a32efce3.{json,md}`

## Commands Run

- `scripts/run_gpu_lowprio.sh --cores 0-23 -- python proofs/v014/exact_branch_memory_preflight.py --run-gpu ...` (baseline + final tree)
- `scripts/run_gpu_lowprio.sh --cores 0-23 -- python proofs/v014/mythos_memory_gpu_suite_260609.py`
- `JAX_PLATFORMS=cpu ... python proofs/v013/moisture_advection_wiring.py` (5 gates PASS)
- `JAX_PLATFORMS=cpu ... python proofs/v014/fp32_acoustic_static_audit.py`
- `JAX_PLATFORMS=cpu ... python proofs/v014/mythos_memory_fixes_260609.py`
- pytest: moisture advection operational + pd_rk3 (14 pass), MYNN suite
  (16 pass, 1 pre-existing env-dependent harness-build failure reproduced on
  untouched main worktree), cache-key (5 pass), pre-halo capture + m7 writer +
  m7 daily pipeline + mynnsl oracle (17 pass 1 skip), no-H2D transfer audits +
  acoustic + flux-advection map factors + pd moisture (16 pass 3 skip)
- `python -m py_compile ...`, `python -m json.tool ...`, `git diff --check`

## Proof Objects

- `proofs/v014/mythos_memory_fixes_260609.{py,json,md}` (lane verdict + tables)
- `proofs/v014/mythos_memory_gpu_suite_260609.{py,json}` (GPU measurements)
- `proofs/v014/exact_branch_memory_preflight.{json,md}` (final tree) +
  `_baseline_a32efce3` copies
- `proofs/v014/fp32_acoustic_static_audit.json`
- `proofs/v013/moisture_advection_wiring.json` (re-passed on this branch)
- `.agent/reviews/2026-06-09-v014-mythos-memory-fixes.md`

## Risks

- The MYNN tiling changes the traced operational program where the flattened
  column batch exceeds 16384; bit-identity is proven CPU+GPU (incl. ragged
  tiles) and the nested preflight is green, but the first post-merge long run
  should still treat it as a fresh-lineage preflight item (it is cheap).
- The R0 static-aux addition changes every OperationalNamelist JIT cache key:
  one-time recompilation cost after merge; no value change (0 consumers).
- The tier1 MYNN parity artifact regenerates with small numeric drift on this
  machine; reproduced on the untouched main worktree (pre-existing
  environment drift, NOT caused by this branch; artifact restored).
- Acoustic-adjacent rows (carry split, pad/mask helpers, state alias, FP32
  R1/R2) are deliberately untouched: same fault surface as the open one-RK-step
  fp64 dynamics divergence.

## Handoff

Manager: review the three separated commits, rerun gates if desired, merge, and
re-run the cheap exact-branch preflight on the post-merge trunk before the next
long validation. Resume FP32 R1 only after the dynamics frontier closes.
