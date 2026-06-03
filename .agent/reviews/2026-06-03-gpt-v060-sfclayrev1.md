# GPT v0.6.0 Revised-MM5 Surface-Layer Handoff

## objective

Port WRF revised-MM5 / Jimenez surface layer (`sf_sfclay_physics=1`,
`module_sf_sfclayrev.F` -> `sf_sfclayrev_run`) to the frozen v0.6.0 physics
interface and gate it against real WRF-module savepoints for unstable,
stable, neutral, low-ust, land, and water regimes.

## files changed

- `src/gpuwrf/physics/sfclay_revised_mm5.py`
  - Added a column-level revised-MM5 surface-layer kernel and adapter returning
    `PhysicsStepResult`.
  - Writes surface flux handles via `state_replacements`: `ustar`,
    `theta_flux`, `qv_flux`, `tau_u`, `tau_v`, `rhosfc`, `fltv`.
  - Returns diagnostics for exchange coefficients, `UST`, `TSTAR`, `QSTAR`,
    `T2/Q2/U10/V10`, `HFX/QFX/LH`, `ZNT`, stability, and Monin-Obukhov fields.
- `tests/test_v060_sfclay_revised_mm5.py`
  - Added CPU-only WRF savepoint parity gate and frozen-interface key check.
- `proofs/v060/oracle/*`
  - Added reproducible Fortran oracle harness linked against unmodified WRF
    `module_sf_sfclayrev.F`, `physics_mmm/sf_sfclayrev.F90`, and
    `ccpp_kind_types.f90`.
- `proofs/v060/savepoints/sfclayrev1_case_1.json` through
  `sfclayrev1_case_6.json`
  - WRF-module oracle savepoints.
- `proofs/v060/savepoints/sfclayrev1_wrf_source_checksums.txt`
  - SHA-256 provenance for copied WRF source files.
- `proofs/v060/sfclayrev1_savepoint_parity_report.json`
  - Machine-readable PASS report.
- `.agent/reviews/2026-06-03-gpt-v060-sfclayrev1.md`
  - This handoff.

## commands run

- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu python -m py_compile src/gpuwrf/physics/sfclay_revised_mm5.py tests/test_v060_sfclay_revised_mm5.py proofs/v060/oracle/dump_to_json.py`
- `taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh`
- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu pytest -q tests/test_v060_sfclay_revised_mm5.py --tb=short`
- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu python -m gpuwrf.contracts.physics_interfaces`
- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu python -m gpuwrf.contracts.physics_registry`

## proof objects produced

- `proofs/v060/sfclayrev1_savepoint_parity_report.json`
  - Verdict: `PASS`.
  - Oracle: single-column Fortran driver linked against pristine WRF
    `module_sf_sfclayrev.F` + `sf_sfclayrev.F90`.
  - `full_wrf_exe`: `false`.
  - Cases: 6 total, covering `unstable_convective_land_water`,
    `stable_nocturnal_land_water`, `neutral_land_water`,
    `previous_unstable_stable_guard`, `low_ust_water_charnock`, and
    `mixed_regime_persisted_qsfc`.
  - Worst residuals by case:
    - Case 1: worst abs `HFX = 0.0013002897910894262`.
    - Case 2: worst abs `ZOL = 0.0005022899785487311`.
    - Case 3: worst abs `LH = 0.0009238879450776949`.
    - Case 4: worst abs `HFX = 0.000338324099764975`.
    - Case 5: worst abs `ZOL = 0.03035061162790953`; relative residual passes.
    - Case 6: worst abs `HFX = 0.0011322365999717476`.
- `proofs/v060/savepoints/sfclayrev1_case_*.json`
- `proofs/v060/savepoints/sfclayrev1_wrf_source_checksums.txt`

## unresolved risks

- The oracle is a real WRF-module oracle, not a full coupled `wrf.exe` run.
  Integrated mixed-suite dispatch remains a later manager-owned gate.
- The port covers the default lane path: `isfflx=1`, `isftcflx=0`,
  `iz0tlnd=0`, SCM-forced flux off. Alternate WRF optional branches are not
  claimed here.
- No GPU performance claim is made; all JAX verification used CPU per
  instruction.

## next decision needed

Manager review: accept the revised-MM5 surface-layer lane as PASS and decide
when to wire `sf_sfclay_physics=1` into the mixed-suite physics dispatcher for
YSU/ACM2 pairings.
