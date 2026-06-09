# V0.14 JAX After-All-RK Same-State Wrapper

Verdict: `WRAPPER_BLOCKED_NO_JAX_PRE_HALO_STATE_API`.

## Target

- WRF truth surface: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.
- Domain/lead: `d02`, step `6000`, h10 `2026-05-02T04:00:00+00:00`.
- Patch: mass `y [1,18), x [5,22)`, native U/V/W/PH staggering preserved.

## Runtime Finding

The closest JAX runtime boundary is final RK3 `_carry_from_finished_stage(..., namelist)` after `_refresh_grid_p_from_finished`, but `_acoustic_scan` immediately wraps that state in `apply_halo(...)` before returning. `_rk_scan_step` and the public forecast entries expose only that later state, followed by guard/boundary handling.

## WRF Oracle

- Parsed WRF pre-halo records: `{'MASS_K1': 289, 'U_K1': 306, 'V_K1': 306, 'WPH_KSTAG01': 578}`.
- Duplicate overlap max delta: `0.0`.

## Available Diagnostic

A retained JAX/GPU h10 wrfout was compared against the WRF truth surface only as a non-acceptance diagnostic. It is not a CPU internal pre-halo state and is not used for the verdict.
- First retained-writer mismatch by contract order: `T` max_abs `3.357818603515625` with tolerance `2e-06`.

## Missing Wrapper Prerequisite

- No production API or proof-only hook exposes the final RK3 post-refresh state before RK halo exchange.
- No CPU JAX h10 same-state savepoint/checkpoint exists in the inspected proof inputs or scratch areas.
- A full CPU forecast through the public API would return the wrong cadence surface, so it was not run.

Next decision: open a narrow source-changing/debug-hook sprint, or approve a narrower wrapper sprint that adds a CPU-only pre-halo capture around `_acoustic_scan` immediately before its `apply_halo` return.
