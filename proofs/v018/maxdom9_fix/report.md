# max_dom=9 compile blowup and nested history cadence proof

Status: PASS.

## Fixes

1. `src/gpuwrf/physics/thompson_column.py`
   - Removed materialized static scan index vectors from Thompson sedimentation/fall-speed paths.
   - `_sed_implicit_q`, `_fill_down`, and `_sed_one_species` now thread scalar `int32` indices in the scan carry and scan over `None`.
   - No physics thresholds, formulas, masks, or dtype-sensitive arithmetic were changed.

2. `src/gpuwrf/integration/nested_pipeline.py`
   - Fixed a real latent writer bug: nested output cadence now uses namelist `time_control.history_interval` instead of hardcoded hourly output.
   - Writer valid time and `XTIME` now use `own_step * dt_by_domain[domain]`, so non-hourly history intervals produce correct lead seconds.
   - Existing hourly gates are unchanged because `ceil(3600 / dt)` equals the old `round(3600 / dt)` for the v0.18.2 AC1_FIT d01/d02/d03 timesteps.

## HLO Root Cause

Initial dump: `<xla-dump-dir>/module_7887.jit__advance_chunk.before_optimizations.txt`.

Evidence:

- File is 21,237,661 bytes before optimizations.
- HLO line 42617 contains `%iota.15 = s64[44]{0} iota()`.
- HLO lines 45311, 45333, 45404, 45417, 45581, 45659, and 45672 carry repeated `s64[44]` iota values through Thompson `while` tuples.
- `FileNames` maps id 1 to `src/gpuwrf/physics/thompson_column.py`.
- `FunctionNames` maps id 8 to `_fill_down` and id 10 to `_sed_one_species.<locals>.body`.
- `FileLocations` map the first `s64[44] iota` to `thompson_column.py` `_fill_down` line 1725 in the captured source frame.

Interpretation: the all-7 max_dom=9 compile was not a scheduler or nested-pipeline index issue. It was Thompson column code materializing static vertical index arrays inside scans. Across nine distinct domain shapes, XLA replicated those static `s64` iotas/ranges inside large `_advance_chunk` programs and spent unbounded time folding/lowering them. Threading scalar dynamic indices through the scan carry removes those static vectors from the scan operands while preserving iteration order and values.

## Measured Proofs

### max_dom=9 all-7 compile bounded

Input: `<all7-staging-input>`, `max_dom=9`, `history_interval=20`.

Cold cache proof artifacts:

- stderr: `proofs/v018/maxdom9_fix/maxdom9_run.stderr`
- SMI: `proofs/v018/maxdom9_fix/maxdom9_run.smi.csv`
- cache: `proofs/v018/maxdom9_fix/jax_cache_maxdom9`

All nine `_advance_chunk` domain-shape compiles completed:

| Shape compile | seconds |
| --- | ---: |
| 1 | 259.456 |
| 2 | 273.095 |
| 3 | 367.154 |
| 4 | 351.210 |
| 5 | 408.646 |
| 6 | 251.635 |
| 7 | 214.754 |
| 8 | 288.461 |
| 9 | 228.021 |

Max cold compile: 408.646 s, below the manager's 15 minute per-shape flag threshold. Cold peak VRAM: 13.84 GiB.

Warm cache proof:

- stderr: `proofs/v018/maxdom9_fix/maxdom9_warm_run.stderr`
- SMI: `proofs/v018/maxdom9_fix/maxdom9_warm_run.smi.csv`

Warm run had 9/9 persistent cache hits for `_advance_chunk`; max cache-hit compile/deserialization path was 21.786 s. It reached stable integration with mean GPU util 84.74%, p95 100%, and peak VRAM 14.17 GiB. The all-7 nest did not reach its first `history_interval=20` wrfout within the timeout because this case is compute-slow, not compile-stuck.

### Nested writer and history_interval=5

Fast nested proof case: AC1_FIT v0.18.2 max_dom=3 with a proof-local `history_interval = 5, 5, 5` namelist copy.

Artifacts:

- input mirror: `proofs/v018/maxdom9_fix/input_ac1fit_history5`
- wrapper JSON: `proofs/v018/maxdom9_fix/ac1fit_history5_run.json`
- finite check: `proofs/v018/maxdom9_fix/ac1fit_history5_wrfout_finite.json`
- wrfout: `proofs/v018/maxdom9_fix/runs/ac1fit_history5_fast/wrfout_d03_2026-04-28_18:05:00`

Result:

- `wrfout_d03_2026-04-28_18:05:00` exists, size 625,803,789 bytes.
- `Times = 2026-04-28_18:05:00`.
- `XTIME = 5.0` minutes.
- 106 numeric fields checked finite.
- No invalid/non-finite numeric fields.
- Wrapper rc is 143 because the run was intentionally terminated after the required d03 +5 min wrfout was written and validated.
- Peak VRAM for this cold AC1_FIT first-output run: 19.70 GiB. The long cold wall time is a big-domain cold compile path, not a regression; warm cache avoids it.

This proves the nested writer path honors `history_interval=5` end to end. The previous hardcoded-hourly path would not have emitted this +5 min d03 file.

### Hourly nested regression

Artifact: `proofs/v018/maxdom9_fix/history_cadence_static_check.json`.

Original AC1_FIT v0.18.2 gate namelist is hourly:

- `history_interval = 60, 60, 60`
- d01 dt 18 s: old `round(3600 / dt) = 200`, new `ceil(3600 / dt) = 200`
- d02 dt 6 s: old 600, new 600
- d03 dt 2 s: old 1800, new 1800
- First valid time remains `2026-04-28_19:00:00` on all three domains.

Status: PASS.

### Bit identity

Artifact: `proofs/v018/maxdom9_fix/bit_identity_compare.json`.

Comparison: pre-fix vs post-fix `examples/switzerland_d01`, one forecast hour.

Result:

- Status: PASS.
- 26/26 compared fields exact.
- `max_abs_diff_overall = 0.0`.
- `all_compared_exact = true`.
- Missing fields: 0.

### CPU focused tests

Command:

```bash
PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu pytest -q \
  tests/test_v015_mp_column_tiling.py \
  tests/test_m5_thompson_column_shapes.py \
  tests/test_m5_thompson_process_residuals.py \
  tests/test_thompson_precip_oracle.py
```

Result: 16 passed.

Also passed:

```bash
PYTHONPATH=src python -m py_compile \
  src/gpuwrf/integration/nested_pipeline.py \
  src/gpuwrf/physics/thompson_column.py \
  proofs/v018/maxdom9_fix/ac1fit_history5_first_output.py
```

## Proof Object Index

- `proofs/v018/maxdom9_fix/measured_summary.json`
- `proofs/v018/maxdom9_fix/history_cadence_static_check.json`
- `proofs/v018/maxdom9_fix/ac1fit_history5_wrfout_finite.json`
- `proofs/v018/maxdom9_fix/ac1fit_history5_run.json`
- `proofs/v018/maxdom9_fix/bit_identity_compare.json`
- `proofs/v018/maxdom9_fix/maxdom9_run.stderr`
- `proofs/v018/maxdom9_fix/maxdom9_warm_run.stderr`
- `<xla-dump-dir>/module_7887.jit__advance_chunk.before_optimizations.txt`

## Remaining Notes

- No all-7 wrfout was waited for after the manager convergence decision. The all-7 proof is compile-bounded plus stable integration; output writing is proven on the fast nested AC1_FIT case through the same nested writer path.
- d01 history-5 first output would be at `18:05:06` because 300 seconds is not divisible by its 18 second timestep. d03 has dt 2 s and writes exactly at `18:05:00`.
