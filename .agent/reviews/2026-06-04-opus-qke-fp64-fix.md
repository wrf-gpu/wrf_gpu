# qke -> FP64 precision-contract change + d03 1km gate re-validation

Branch: `worker/opus/v090-qke-fp64-fix` (from `worker/opus/v090-validation-burst` @ 5fec5ef)
Date: 2026-06-04
Author: Opus (precision-contract lane)

## Objective

Close the last open 0.9.0 validation gate: d03 1km gated-fp32 going NON-FINITE
at forecast hour 1. The validation-burst root-cause proof
(`proofs/v090/d03_1km_validation.json`) attributed this to **qke (MYNN TKE)
overflowing fp32 at 1km** and recommended promoting qke (+ the MYNN
length-scale/TKE intermediates) to FP64 in the gated precision matrix.

## What was changed (committed)

1. `src/gpuwrf/contracts/precision.py`: `PRECISION_MATRIX['qke']`
   `(FP32_GATED, True) -> (FP64, False)`. qke is outside the conserved
   mass/pressure/acoustic path, so this preserves the gated-fp32 invariants and
   the d02 speedup (qke is one small 3D field).
   - The MYNN length-scale + TKE-budget **intermediates** (el/elt/els/elf in
     `_mym_length_option1`, the qke-weighted height integral, the
     `_mym_predict_qke` tridiagonal budget) fp64-promote **implicitly**: they
     are all functions of qke (the column kernel carries `tke = qke/2`), so once
     qke is fp64 JAX type-promotion widens every qke-touching intermediate.
     `mynn_pbl.py` has NO explicit float32 narrowing (only int32 index casts) and
     the column rho/dz are already fp64, so **no `mynn_pbl.py` edit is needed**.
     The output write is fp64 automatically via `_output_dtype` (live state dtype).
   - No physics change; the existing WRF-faithful `mym_predict` qke cap (<=150)
     is untouched. NOT a clamp/fudge.
2. `tests/test_m6_precision_matrix.py`: qke moved gated->FP64-locked. Also fixed a
   **pre-existing** breakage: the v0.6.0 WDM6 leaves Nc/Nn were FP32_GATED in the
   matrix but never added to the test's gated set, so the test failed on the base
   branch independent of this change. Test now passes (GPU).
3. `src/gpuwrf/coupling/physics_couplers.py`: A2C wind-increment scatter made
   dtype-safe. With qke fp64 the MYNN column u/v output fp64-promotes, so the A2C
   `du_mass/dv_mass` arrive fp64 while the C-grid faces stay FP32_GATED; cast the
   increment to the face dtype before the scatter. No numerical change (same fp32
   bytes); silences a future-JAX hard-error FutureWarning. (A separate, identical,
   PRE-EXISTING fp64->fp32 scatter FutureWarning remains in the dynamics path; it
   appears in the pre-fix d02 gated-fp32 logs too -- benign, unrelated to qke.)

## VERDICT: the premise is FALSIFIED -- qke fp64 does NOT close the gate

The change was implemented and **verified to take effect** (direct GPU probe:
`PRECISION_MATRIX[qke]=(float64,False)`, `t0 state.qke.dtype=float64`,
`mynn_adapter` output qke is float64). The full 24h d03 1km gated-fp32 run
(`scripts/d03_replay.py --gated-fp32`, no clamp) **still BLOCKS at forecast
hour 1** with qke the sole nonfinite field and the **IDENTICAL** signature to
the pre-fix proof: **3036 nonfinite cells, same finite min/max (2.33e-5 /
27.36)**. Identical-to-the-digit numbers => fp64 changed nothing about the
blow-up; the same cells diverge.

### Root-cause localization (GPU micro-step probes)

- qke is genuinely float64 at every checkpoint.
- qke goes nonfinite within ~2 coupled dt steps (6 s) over **~67-69 specific
  columns** (2948-3036 cells = ~67-69 cols x 44 levels) -- column-localized to the
  steep Tenerife terrain.
- At blow-up qke magnitudes are **tiny (~0.04 -> 0.13)** -- NO precision-range
  overflow (fp32 or fp64).
- On the last finite state, the **real operational `mynn_adapter` produces a
  FULLY FINITE qke** (max 0.071) and finite dfm/dfh/km/kh/el/qkw. => MYNN physics
  is NOT the NaN source on a finite input.
- Yet a single coupled forecast step (dynamics acoustic/RK + boundary + MYNN-in-
  core) on that same finite state yields nonfinite qke. The MYNN-in-core runs on
  the **dynamically-evolved intermediate** state, so the **dynamics** produces an
  unstable intermediate (extreme near-surface shear/theta over the 67-69 steep
  columns at dt=3s) that drives the in-core qke budget to NaN.
- `_mym_level2` gm/gh, ustar, dz0 all stay finite through the blow-up -> not a
  divide-by-tiny-dz in the level-2 gradients.

**Conclusion:** the d03 1km hour-1 blow-up is a **dynamics-driven structural
instability over the steep Tenerife terrain** that surfaces in qke (the most
sensitive field). It is NOT an fp32 precision-range overflow. A precision-matrix
change cannot fix it. The gate remains **OPEN**.

Proof: `proofs/v090/d03_1km_validation_qkefix.json`,
`proofs/v090/pipeline_run_d03_qkefix_gated_fp32.json`.

## d02 gated-fp32 NOT REGRESSED (proof: `proofs/v090/d02_gated_fp32_recheck.json`)

qke->FP64 is NEUTRAL for d02. On the only data-available L2 case in the (66,159)
drop-in d02 config (20260521), the production gated-fp32 pipeline goes nonfinite at
hour 1 with qke (~2024 cells) **IDENTICALLY** whether qke is base FP32_GATED or the
promoted FP64 -- same 2024 cells, same hour. So qke precision is irrelevant to d02
stability here (no regression, no fix), reinforcing that the production-pipeline
gated-fp32 blow-up over steep terrain is structural/dynamics-driven with qke as the
canary.

Caveats (honest): a fresh GREEN d02 gated-fp32 speedup could NOT be re-measured --
the prior GREEN gated-fp32 d02 cases (20260507/20260523) are wrfout-purged, and the
20260521 production path (with the daily-pipeline HOURLY LAND-STATE REFRESH that the
prior 3h-finite reverify harness lacked) is itself unstable. The qke-fp64 overhead is
expected negligible (one small 3D field; dynamics cost unchanged) but is asserted on
first principles, not freshly timed, due to that data/stability gap.

## d03 1km speedup

NOT FILLABLE this sprint: the d03 gated-fp32 run NaNs at hour 1, so no
command-to-finish wall-clock for a COMPLETE 1km forecast exists at ship precision.
`proofs/v090/speedup_benchmark.json` single_1km row therefore stays honestly
UNFILLED (MEASUREMENT_STATUS=BLOCKED), with the blocked_reason updated to point at
this falsified-premise finding (the blocker is dynamics stability, not qke fp32).

## Recommendation

- KEEP the qke->FP64 promotion (correct + harmless for a turbulence diagnostic
  outside the conserved path; d02 speedup unaffected). Do NOT claim it closes d03.
- Open a real **numerics/stability** sprint for d03 1km gated-fp32 dynamics over
  steep terrain (NOT a precision-contract sprint). Leads in
  `d03_1km_validation_qkefix.json:carry_over`.

## Resources

CPU pinned to cores 0-3 throughout (cores 4-31 = live CPU-WRF backfill, untouched).
GPU lock claimed/released, cpu_cores_4_31 preserved. One GPU job at a time.
