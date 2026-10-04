# Mont-Blanc native-dt Ni onset fix summary

Date: 2026-06-26
Branch: `worker/opus/vnext-parallel-compile`

## Verdict

Root cause is the standalone root-domain specified-boundary dynamics path, not
Thompson microphysics. In the fresh-JIT native-dt failing case the first reported
non-finite leaf is still `Ni`, but the dry d01 boundary zone has already blown up
one to two root steps earlier.

The WRF-faithful fix is to run standalone wrfbdy roots with WRF specified-boundary
timestep cadence:

- `specified_bdy_cadence=True`
- `specified_adv_degrade=True`
- root `boundary_config.update_cadence_s = wrfbdy interval_seconds`
- root `boundary_config.normal_bdy_relax_strength = 1.0`
- optional scalar wrfbdy leaves for `qc/qr/qi/qs/qg/Ni/Nr`, applied only when the
  wrfbdy file contains the corresponding WRF side base/tendency arrays

No `nan_to_num`, `isfinite` guard, or non-physical finite masking was added for
this fix. The scalar boundary nonnegative floor is the existing WRF scalar
positivity treatment at lateral boundaries, not a NaN sanitizer.

## Onset Trace

Proof objects:

- `proofs/v022/nest_dycore/mb_native_jit_trace/logs/run.log`
- `proofs/v022/nest_dycore/mb_native_jit_trace/per_step.jsonl`

Fresh-JIT trace result:

- `_FirstNonFiniteHit: domain=d01 step=14 field=Ni idx=[15, 12, 124]`
- d01 step 12: `w max_abs=162.1536`, `p_total max_abs=2.714010435562908e14`, `Ni=0`
- d01 step 13: `w max_abs=1.6542927237103234e277`,
  `mu_total max_abs=9.386209596701947e279`,
  `ph_total max_abs=2.73823979802792e279`,
  `p_total max_abs=2.0478804354789848e229`, `Ni max_abs=2638705.0517592565`
- d01 step 14: `Ni` has 22 non-finite cells, but dry fields are already
  astronomical

Interpretation: `Ni` is the first leaf that trips the finite guard, but the
precursor is a dry dynamics boundary-zone runaway. Exact vertical Courant `Cz`
was not recorded in the final proof artifact because the failing fresh-JIT trace
failed before the originally expected step-55-to-67 window; max `w` and dry-state
precursors are recorded.

Negative probe:

- `proofs/v022/nest_dycore/mb_native_jit_fixed_probe/logs/run.log`
- Scalar wrfbdy leaves plus `normal_bdy_relax_strength=1.0` alone still failed:
  `_FirstNonFiniteHit: domain=d02 step=27 field=Ni`
- This ruled out microphysics/scalar-LBC alone and left missing specified-boundary
  cadence/advection degradation as the dynamics root.

## Validation

Mont-Blanc native-dt fixed 2 h proof:

- Proof: `proofs/v022/nest_dycore/mb_native_jit_fixed_2h/proof/nested_pipeline_run.json`
- Log: `proofs/v022/nest_dycore/mb_native_jit_fixed_2h/logs/run.log`
- Verdict: `PIPELINE_PARTIAL` only because d02 wrote fewer wrfout files than the
  generic expected-output counter
- Finiteness: `all_domains_finite=true`
- d01: `dt_s=18.0`, `own_steps=400`, `final_state_finite=true`,
  `wrfout_count=5/5`
- d02: `dt_s=6.0`, `own_steps=1200`, `final_state_finite=true`,
  `wrfout_count=2/6`

20250121 no-regression proof:

- Proof: `proofs/v022/nest_dycore/20250121_native_jit_regression/proof/nested_pipeline_run.json`
- CPU correlation: `proofs/v022/nest_dycore/20250121_native_jit_regression/proof/tuv_cpu_correlation.json`
- Verdict: `PIPELINE_GREEN`
- Finiteness: d01/d02/d03 all finite
- T correlation `0.9999951044773282`, RMSE `0.16701063948174175`
- U correlation `0.9998386146262473`, RMSE `0.2680964364174553`
- V correlation `0.999750773176717`, RMSE `0.22272096086936863`

Canary no-regression proof:

- Proof: `proofs/v022/nest_dycore/canary_native_jit_regression/proof/nested_pipeline_run.json`
- Log: `proofs/v022/nest_dycore/canary_native_jit_regression/logs/run.log`
- Verdict: `PIPELINE_GREEN`
- Finiteness: d01/d02/d03 all finite
- d01: `dt_s=18.0`, `own_steps=200`, `wrfout_count=2/2`
- d02: `dt_s=6.0`, `own_steps=600`, `wrfout_count=3/3`
- d03: `dt_s=2.0`, `own_steps=1800`, `wrfout_count=3/3`

Fast CPU contract tests:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src pytest -q \
  tests/test_v017_qh_hail_state.py \
  tests/test_v016_thompson_aero_threading.py \
  tests/contracts/test_v060_physics_interfaces.py \
  tests/test_v018_conditional_state_leaves.py \
  tests/test_m7_restart_checkpoint_roundtrip.py \
  tests/test_m6_precision_matrix.py
```

Result: `34 passed, 2 skipped`.
