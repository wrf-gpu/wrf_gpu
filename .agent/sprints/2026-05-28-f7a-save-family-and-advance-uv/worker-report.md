# F7A Worker Report

Verdict: **F7A_COMPLETE** for the scoped acceptance gates. The dycore is still unstable and needs F7.B.

## Objective

Implement the first scoped dycore rewrite: WRF-shaped small-step prep/finish, RK1 `_1` family carry across RK2/RK3, per-stage `*_save` rebuild, loop-entry `calc_p_rho(step=0)`, explicit RK `dts_rk`, and acoustic `advance_uv_wrf` before `advance_mu_t_core`.

## Files Changed

- `src/gpuwrf/dynamics/core/small_step_prep.py`
- `src/gpuwrf/dynamics/core/small_step_finish.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `scripts/f6_transaction_audit.py`
- `proofs/f7a/*`
- `.agent/sprints/2026-05-28-f7a-save-family-and-advance-uv/worker-report.md`

## Commands Run

- `taskset -c 0-3 python -c "import jax; print([str(d)+':'+d.platform for d in jax.devices()])"` -> `cuda:0:gpu`
- `taskset -c 0-3 python -m py_compile src/gpuwrf/dynamics/core/small_step_prep.py src/gpuwrf/dynamics/core/small_step_finish.py src/gpuwrf/dynamics/core/calc_p_rho.py src/gpuwrf/dynamics/core/acoustic.py src/gpuwrf/runtime/operational_mode.py`
- `taskset -c 0-3 pytest -q tests/unit/test_rk_scan_step_advection_active.py tests/unit/test_mu_persistence_two_substeps.py tests/unit/test_decouple_theta_state_reference.py`
- `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 1 --combination a --output-dir /tmp/f7a_smoke`
- `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12 --output-dir proofs/f7a`

## Proof Objects Produced

- `proofs/f7a/audit_combination_a.json`
- `proofs/f7a/audit_combination_b.json`
- `proofs/f7a/audit_combination_c.json`
- `proofs/f7a/audit_combination_d.json`
- `proofs/f7a/invariant_violations.json`
- `proofs/f7a/audit_summary.md`
- `proofs/f7a/regression_diff.md`
- `proofs/f7a/speedup_estimate.json`

## Results

- 3 required F6 unit tests pass.
- Combination `a` acoustic u/v max delta changed from `0.000e+00` to `3.873e+121`.
- Original F6 first critical moved from step 1/RK1/substep 1 to step 1/RK3/substep 8.
- `rk_saved_state_theta_1` no longer fires in the 12-step audit.
- Original earliest `muts_mut_work_mu_consistency` no longer fires; first combination `a` hit is later at step 2/RK2/substep 1 with max error `2.441e-04`.
- T2 RMSE was not measurable because the dycore remains unstable before a forecast product exists.

## Unresolved Risks

- `advance_uv_wrf` activates acoustic u/v, but the magnitude is unstable; this is not a GPU performance or physics-validity claim.
- Pressure bound worsens by step 2/RK1; full `advance_w_wrf` plus `calc_p_rho(step=iteration)` remains the next required repair.
- The F6 audit harness was updated to instrument the new prepared acoustic path and to avoid applying physical wind sanity bounds to coupled WRF work arrays.
- `rk_addtend_dry`, scalar flux accumulation, and full boundary cadence remain out of scope.

## Next Decision Needed

Proceed to F7.B: full `advance_w_wrf`, geopotential RHS, and `calc_p_rho(step=iteration)` to address the remaining pressure and RK3/substep8 instability.
