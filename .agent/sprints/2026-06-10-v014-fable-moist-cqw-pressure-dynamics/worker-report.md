# Worker Report

Summary: Fable/Mythos implemented and CPU-proved a WRF-faithful moist `cqw` /
moist `pg_buoy_w` path for the operational acoustic w-equation. The production
patch adds `moist_cqw_calc_face` and `pg_buoy_w_moist`, threads the resulting
`rw_tend` and implicit-W `cqw=cq1` through `_acoustic_core_state_from_prep`, and
keeps a `GPUWRF_MOIST_CQW` switch for bisection.

Files changed by worker:

- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_v014_moist_cqw_pressure_dynamics.py`
- `proofs/v014/moist_cqw_pressure_dynamics_closure.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-fable-moist-cqw-pressure-dynamics.md`

Proof objects:

- `proofs/v014/moist_cqw_pressure_dynamics_closure.json`
- `proofs/v014/moist_cqw_pressure_dynamics_closure.md`
- `.agent/reviews/2026-06-10-v014-fable-moist-cqw-pressure-dynamics.md`

Key result:

- Pre-fix GPU `P+PB(k0)` sits on the dry hydrostatic column:
  off-dry about `8 Pa`, off-moist about `203 Pa`.
- CPU/WRF sits on the moist hydrostatic column:
  off-moist about `13.5 Pa`.
- The missing WRF term is the water-mass loading in `pg_buoy_w`:
  `-cq2*(c1f*mub+c2f)`.
- Moist path is bit-identical to dry when total moisture is zero and the
  implicit-W coefficients remain finite/well-conditioned.

Unresolved worker risk: GPU validation was not run by Fable; manager had to run
the h1-h4 GPU gate before accepting default ON.
