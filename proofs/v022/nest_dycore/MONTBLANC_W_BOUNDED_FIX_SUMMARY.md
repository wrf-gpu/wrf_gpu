# Mont-Blanc W-bounded release-blocker fix

Date: 2026-06-27

## Verdict

The false-pass `9767a157` was a dynamics boundary error, not microphysics and
not a steep-slope interior blow-up. The d01 max `|W|` grew at the lateral
boundary corner while the interior stayed bounded.

The fix is WRF-faithful specified-domain `zero_grad_bdy` for `W`: after
`advance_w`, compute the acoustic work-array values whose
`small_step_finish_wrf` reconstruction has the WRF zero-gradient physical `W`.
This also follows WRF's corner source-index behavior, where y-side boundary rows
own the corners and the source column is taken from the nearest interior column.

No `W` masking, `nan_to_num`, finite guard, or value clip/clamp was added.

## Diagnosis

False-pass d01 max `|W|` locations in
`proofs/v022/nest_dycore/mb_native_jit_fixed_2h/out`:

| frame | max `|W|` m/s | max index `(k,j,i)` | edge4 max | interior4 max |
|---|---:|---|---:|---:|
| 12:20 | 44.0388 | `(29,0,128)` | 44.0388 | 2.31 |
| 12:40 | 287.702 | `(28,0,128)` | 287.702 | 4.13 |
| 13:00 | 2065.948 | `(27,0,128)` | 2065.948 | 3.57 |
| 13:20 | 14531.26 | `(26,0,128)` | 14531.26 | 3.62 |
| 13:40 | 99325.2 | `(26,0,128)` | 99325.2 | 3.59 |

This is a boundary-zone/corner runaway. It is not an interior terrain-slope max.
The Mont-Blanc terrain is severe (`d01` max one-cell slope 0.395, `d02` 0.795),
but the observed d01 runaway was at the outer specified boundary corner.

`w_damping=1` and `damp_opt=3` Rayleigh are already threaded from the namelist
into the d01 acoustic `advance_w_wrf` path. The failed behavior came from our
specified-domain `zero_grad_bdy(W)` representation: copying the small-step work
row does not copy finished physical `W` when `w_save` or dry mass differs across
the boundary row, and the old copy missed WRF's corner source-index behavior.

## Validation

Mont-Blanc exact-source native-dt fresh-JIT 2h:

```bash
scripts/with_gpu_lock.sh --timeout 36000 --label codex-mb-wbounded-exact-2h \
  env OMP_NUM_THREADS=8 PYTHONPATH=src JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_COMPILATION_CACHE_DIR=/tmp/gpuwrf_mb_wbounded_fix_exact_2h \
  GPUWRF_JAX_CACHE_DIR=/tmp/gpuwrf_mb_wbounded_fix_exact_2h \
  GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo GPUWRF_NESTED_AOT=0 \
  python -m gpuwrf.cli run --namelist <DATA_ROOT>/wrf_downscale/fixture_montblanc_256/namelist.input \
  --input-dir <DATA_ROOT>/wrf_downscale/fixture_montblanc_256 \
  --output-dir proofs/v022/nest_dycore/mb_wbounded_fix_exact_2h/out \
  --proof-dir proofs/v022/nest_dycore/mb_wbounded_fix_exact_2h/proof \
  --scratch-dir proofs/v022/nest_dycore/mb_wbounded_fix_exact_2h/scratch \
  --domain d01 --max-dom 2 --hours 2
```

Proofs:

- `proofs/v022/nest_dycore/mb_wbounded_fix_exact_2h/proof/nested_pipeline_run.json`
- `proofs/v022/nest_dycore/mb_wbounded_fix_exact_2h/proof/w_max_stats.json`
- `proofs/v022/nest_dycore/mb_wbounded_fix_exact_2h/logs/run.log`

Result: `PIPELINE_PARTIAL` only because the generic d02 expected-output counter
expects six child files; the run wrote the two actual d02 history files and both
domains have `final_state_finite=true`.

Bounded `W`:

| domain/frame | max `|W|` m/s | max index `(k,j,i)` | edge4 max | interior4 max |
|---|---:|---|---:|---:|
| d01 12:20 | 2.3112 | `(11,33,61)` | 0.8981 | 2.3112 |
| d01 12:40 | 4.1291 | `(12,32,61)` | 1.0585 | 4.1291 |
| d01 13:00 | 3.5711 | `(10,23,65)` | 1.3300 | 3.5711 |
| d01 13:20 | 3.6199 | `(11,14,49)` | 1.3960 | 3.6199 |
| d01 13:40 | 3.5820 | `(12,30,61)` | 1.4368 | 3.5820 |
| d02 13:00 | 8.0359 | `(11,27,117)` | 4.5800 | 8.0359 |
| d02 14:00 | 7.0665 | `(12,111,161)` | 3.0290 | 7.0665 |

Mont-Blanc fixture has `max_dom=2`; d03 validity was covered by both 3-domain
regression gates below.

20250121 no-regression:

- `proofs/v022/nest_dycore/20250121_wbounded_regression/proof/nested_pipeline_run.json`
- `proofs/v022/nest_dycore/20250121_wbounded_regression/proof/tuv_cpu_correlation.json`
- Verdict: `PIPELINE_GREEN`
- d01/d02/d03 `final_state_finite=true`
- T/U/V aggregate correlations vs CPU-WRF:
  - T: `0.9999951044784979`, RMSE `0.16701145377371143`
  - U: `0.9998386152497833`, RMSE `0.26809555956523523`
  - V: `0.999750770057355`, RMSE `0.22272225140013935`
- d03 19:00 correlations: T `0.9999843120720755`, U `0.9992500707691371`,
  V `0.9988731203337751`

Canary no-regression:

- `proofs/v022/nest_dycore/canary_wbounded_regression/proof/nested_pipeline_run.json`
- `proofs/v022/nest_dycore/canary_wbounded_regression/logs/run.log`
- Verdict: `PIPELINE_GREEN`
- d01/d02/d03 `final_state_finite=true`

CPU tests:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src pytest -q \
  tests/test_v014_specified_bdy_cadence.py \
  tests/test_v017_qh_hail_state.py \
  tests/test_v016_thompson_aero_threading.py \
  tests/contracts/test_v060_physics_interfaces.py \
  tests/test_v018_conditional_state_leaves.py \
  tests/test_m7_restart_checkpoint_roundtrip.py \
  tests/test_m6_precision_matrix.py
```

Result: `40 passed, 3 skipped`.
