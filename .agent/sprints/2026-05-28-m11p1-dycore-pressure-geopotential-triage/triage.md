# M11.1 Dycore Pressure/Geopotential Triage

Decision: `PROGNOSTIC_BUG_FIX_DYCORE`

## WRF Semantics

Reference read: official WRF source tag `v4.6.1` from `https://github.com/wrf-model/WRF`.

- `PH` / `ph_perturbation` is nonhydrostatic acoustic dycore state. `solve_em.F:1049-1064` describes the small-step sequence as advancing `w` and geopotential, and `solve_em.F:1494-1501` passes `grid%ph_2` into `advance_w`. In `module_small_step_em.F:1460-1463`, `advance_w` writes `ph(i,k,j) = ...`.
- `P` / `p_perturbation` is diagnostic pressure, but it is dycore-local and refreshed during the small-step dycore path. `solve_em.F:1617-1630` calls `calc_p_rho` after the acoustic `advance_w` call, and `module_small_step_em.F:515-528` recomputes nonhydrostatic `p` from current `ph/theta/mu` quantities. The broader post-RK helper documents the same semantic: `module_big_step_utilities_em.F:1004-1006` says nonhydrostatic `calc_p_rho_phi` calculates diagnostic pressure from prognostic variables.

## GPU State Contract Match

`src/gpuwrf/contracts/state.py` carries both fields as resident WRF perturbation leaves:

- `p_perturbation`: resident WRF perturbation pressure used by pressure-gradient terms; diagnostic, but must be refreshed inside the dycore after acoustic `ph/theta/mu` changes.
- `ph_perturbation`: resident WRF perturbation geopotential advanced by the nonhydrostatic acoustic dycore.

Therefore the diagnostic harness expectation that `dycore_rk3` changes both leaves is valid. No harness verdict change is needed.

## Root Cause

Before this sprint, `src/gpuwrf/dynamics/core/acoustic.py` committed `mu`, `theta`, `ph_tend`, and `w` from `acoustic_substep_core`, but the `return state.replace(...)` path did not commit updated `ph` or `p`. That made `state.ph_perturbation` and `state.p_perturbation` flatline through `dycore_rk3`.

## Fix

Minimal dynamics-only fix in `src/gpuwrf/dynamics/core/acoustic.py`:

- Added `_advance_geopotential(...)` to commit a WRF/MPAS-family off-centered geopotential update after the implicit `w` solve.
- Added `_diagnose_pressure(...)` to refresh resident perturbation pressure from the updated dry-column-mass perturbation.
- Added `ph=ph_next` and `p=p_next` to the existing acoustic-core `state.replace(...)`.

State docs were clarified in `src/gpuwrf/contracts/state.py`. No harness or README verdict changes were made.

## Evidence

- `proofs/m11p1/diagnostic_report_after_fix.json`: `dycore_rk3` verdict is `ACTIVE`.
- In that report, `dycore_rk3` `p_perturbation` mean/max deltas are `12.678664475583766` / `92.11724913618252`.
- In that report, `dycore_rk3` `ph_perturbation` mean/max deltas are `0.44641594975908894` / `90.90342389586569`.
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py`: passed.

## Residual Risk

The pressure refresh is a reduced resident diagnostic refresh, not a full WRF `calc_p_rho` equation-of-state port with explicit `alt/c2a` intermediates. This sprint proves the silent identity path is removed and the existing 100-step guard is preserved; it does not claim full pressure-equation parity.
