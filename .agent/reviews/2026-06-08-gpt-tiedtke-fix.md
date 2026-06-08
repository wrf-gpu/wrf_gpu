# GPT Tiedtke Operational Coupling Fix - 2026-06-08

## Objective

Fix the v0.13 functional-coverage defect where `cu_physics=6` modified Tiedtke was
advertised as scan-wired but operationally inert because
`coupling/scan_adapters.py::tiedtke_adapter` fed hard-zero `QVFTEN/QVPBLTEN`.

## Diagnosis

The Tiedtke JAX column kernel was not the defect. It triggers when supplied a
nonzero moisture-convergence forcing. The broken layer was the operational
coupling: the scan adapter previously assembled every column with zero
`QVFTEN/QVPBLTEN`, so the closure saw no large-scale vapor forcing and produced
zero convective tendency/precipitation on the convective smoke column.

Operationally available faithful sources:

- `RQVFTEN`: the existing runtime flux-form moisture-advection helper
  `_moisture_coupled_tendencies` returns the coupled `d(mut*qv)/dt` qv tendency.
  Dividing by WRF mass weight `c1h*mu + c2h` gives the same kg kg-1 s-1 forcing
  WRF passes to Tiedtke.
- `RQVBLTEN`: at the PBL slot, the runtime has the state before and after PBL.
  `(qv_after_pbl - qv_before_pbl) / dt` is the PBL qv tendency handed to Tiedtke.

Because this only has a faithful `RQVFTEN` source when the runtime's active
flux-form moisture-advection path is enabled, `cu_physics=6` now fails closed if
`use_flux_advection` is not true or `moist_adv_opt` is not `1`/`2`. This avoids
advertising a silently inert scheme.

## WRF Reference

- `/home/enric/src/wrf_pristine/WRF/dyn_em/module_em.F:1366-1375`:
  for Tiedtke/New-Tiedtke and `P_QV`, WRF writes scalar advection tendency into
  `RQVFTEN` via `set_tend(RQVFTEN, advect_tend, msfty, ...)`.
- `/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:5255-5263`:
  WRF mass-normalizes `RQVBLTEN` by `(c1(k)*MUT(I,J)+c2(k))`.
- `/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:5287-5305`:
  WRF mass-normalizes `RQVFTEN` by `(c1(k)*MUT(I,J)+c2(k))` for Tiedtke-family
  cumulus schemes.
- `/home/enric/src/wrf_pristine/WRF/dyn_em/module_first_rk_step_part1.F:1500-1506`:
  `cumulus_driver` receives `RQVBLTEN=grid%rqvblten` and
  `RQVFTEN=grid%rqvften`.
- `/home/enric/src/wrf_pristine/WRF/phys/module_cumulus_driver.F:1388-1407`:
  `CU_TIEDTKE` is called with `QVPBLTEN=RQVBLTEN` and `QVFTEN=RQVFTEN`.
- `/home/enric/src/wrf_pristine/WRF/phys/module_cu_tiedtke.F:115-120`:
  the Tiedtke interface takes `qvften,qvpblten`.

## Path Chosen

Chose path 2a: make `cu_physics=6` genuinely functional in the operational scan.

Rationale:

- The operational runtime already had a WRF-formulated flux-form qv tendency
  helper matching the `RQVFTEN` physics hand-off once mass-normalized.
- The PBL qv tendency is directly recoverable at the PBL slot without adding a
  new carry field.
- The fix is fail-closed when the RQVFTEN source is not active, so the default
  suite and any inert Tiedtke configuration remain honest.

## Files Changed

- `src/gpuwrf/coupling/scan_adapters.py`: `tiedtke_adapter` now accepts optional
  `qvften` and `qvpblten` arrays and passes them into `tiedtke_column_jax`.
- `src/gpuwrf/runtime/operational_mode.py`: added the WRF-style RQVFTEN
  diagnostic, PBL qv tendency hand-off, Tiedtke-specific cumulus dispatch, and
  fail-closed guard requiring active flux-form moisture advection for cu=6.
- `tests/test_v013_operational_smoke.py`: flipped the inert xfail into a passing
  nonzero-trigger test and added a fail-closed no-RQVFTEN-source test.
- `src/gpuwrf/coupling/physics_dispatch.py`, `src/gpuwrf/io/namelist_check.py`,
  `src/gpuwrf/io/scheme_catalog.py`, `docs/namelist-compatibility.md`,
  `README.md`: documented the cu=6 RQVFTEN requirement.

## Physical Sanity Gate

CPU-only convective smoke column with `cu_physics=6`, `use_flux_advection=True`,
`moist_adv_opt=2`, `dt_s=900`, and microphysics/PBL/sfclay disabled:

- `qvften_min = -3.6977159055278756e-05`
- `qvften_max = 3.002404752028934e-05`
- `qvften_mean = -2.7108835104175055e-08`
- `rainc_acc_min = 0.0`
- `rainc_acc_max = 4.192830293641778`
- `theta_delta_maxabs = 19.929579979571884`
- `qv_delta_maxabs = 0.0018026560998280032`
- `qv_after_min = 0.0008996352378965887`
- `all_finite = True`

This is not a JAX-vs-JAX self-compare. It verifies that the operational scan now
feeds a real WRF-style signed moisture-convergence tendency and that Tiedtke
produces finite, nonzero convective precipitation and tendencies on the strong
convective column. Existing Tiedtke kernel savepoint parity remains the
`proofs/v060/tiedtke_gpubatch_savepoint_parity.json` kernel-level oracle; this
lane fixed the runtime coupling gap.

## Default Byte-Unchanged Proof

Static proof:

- No default values were changed. `OperationalNamelist.use_flux_advection` remains
  `False`, `moist_adv_opt` remains `0`, and default `cu_physics` remains `0`.
- The new RQVFTEN diagnostic is called only in `_physics_step_forcing` when
  `cu_opt == 6`.
- Default `cu_physics=0` bypasses the entire new branch.

Executable proof:

- The combined gate included
  `tests/dynamics/test_moisture_advection_operational.py::test_default_moist_adv_opt0_byte_identical_and_passthrough`,
  which passed.

## Commands Run

All Python commands used CPU only with:
`JAX_PLATFORMS=cpu PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 taskset -c 24-27`.

- `python -m py_compile src/gpuwrf/coupling/scan_adapters.py src/gpuwrf/runtime/operational_mode.py src/gpuwrf/coupling/physics_dispatch.py src/gpuwrf/io/namelist_check.py src/gpuwrf/io/scheme_catalog.py tests/test_v013_operational_smoke.py`
  - Passed.
- `python -m pytest -q tests/test_v013_operational_smoke.py::test_cumulus_tiedtke_operational_triggers_with_real_qvften tests/test_v013_operational_smoke.py::test_cumulus_tiedtke_requires_qvften_source --tb=short`
  - `2 passed, 5 warnings in 4.74s`.
- `python -m pytest -q tests/test_v013_operational_smoke.py tests/test_v060_physics_dispatch.py tests/contracts/test_v060_physics_interfaces.py tests/test_namelist_check.py tests/dynamics/test_moisture_advection_operational.py::test_default_moist_adv_opt0_byte_identical_and_passthrough --tb=short`
  - Final rerun: `97 passed, 5 warnings in 65.24s`.
- Convective-column sanity script using the same smoke-test setup.
  - Produced the numerical values listed above.

Warnings were pre-existing Python deprecation warnings in
`physics/cumulus_tiedtke_jax.py` for boolean `~` usage.

## Proof Objects Produced

- This report: `.agent/reviews/2026-06-08-gpt-tiedtke-fix.md`.
- Passing CPU-only tests listed above.
- Updated operational smoke test that asserts nonzero cu=6 convective precipitation
  and tendency with real RQVFTEN, plus fail-closed behavior when the source is
  absent.

## Unresolved Risks

- I did not build a new pristine-Fortran single-column oracle for this synthetic
  operational coupling column. The existing Tiedtke kernel parity proof covers
  the kernel; this lane validates the operational hand-off and physical trigger.
- `cu_physics=6` is now operational only for configurations with active flux-form
  moisture advection (`use_flux_advection=True`, `moist_adv_opt=1/2`). Without
  that source it deliberately fails closed.

## Next Decision Needed

None for this defect. A future hardening lane could add a pristine-WRF coupled
savepoint specifically around `cumulus_driver` for the operational Tiedtke hand-off.
