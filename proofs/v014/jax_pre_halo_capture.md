# V0.14 JAX Pre-Halo Capture Hook

Verdict: `HOOK_GREEN_COMPARE_BLOCKED_NO_JAX_H10_PRESTEP_CARRY`.

## Hook Proof

- CPU fixture status: `HOOK_GREEN`.
- Captured cadence: `final RK3 _carry_from_finished_stage(..., namelist) -> _maybe_exchange_sharded_carry_halos(...) -> before apply_halo(next_carry.state, halo_spec(...))`.
- Normal `_rk_scan_step` return type with hook disabled: `OperationalCarry`.
- Normal return vs capture carry exact: `True` (max_abs `0.0`).
- Captured target field views finite: `True`.

## WRF Green Target

- WRF verdict: `REFRESH_LAYER_GREEN_post_after_all_rk_steps_pre_halo`.
- Target: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.
- Domain/step: `d02`, step `6000`, `2026-05-02T04:00:00+00:00`.
- Patch counts: `{'MASS_K1': 289, 'U_K1': 306, 'V_K1': 306, 'WPH_KSTAG01': 578}`.

## H10 Compare

- Status: `BLOCKED`.
- Missing input: `CPU-loadable JAX OperationalCarry immediately before d02 step 6000/h10.`.
- No retained wrfout or JAX-vs-JAX diagnostic is used as a same-surface verdict.
