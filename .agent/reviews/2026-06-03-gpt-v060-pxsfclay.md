# GPT v0.6.0 Pleim-Xiu Surface Layer Handoff

## Objective

Port WRF Pleim-Xiu surface layer (`sf_sfclay_physics=7`) to JAX using an
unmodified-WRF Fortran savepoint oracle, then gate the JAX adapter against
regime savepoints through the frozen v0.6.0 physics interface.

## Files Changed

- `src/gpuwrf/physics/sfclay_pleim_xiu.py`
- `tests/test_v060_sfclay_pleim_xiu.py`
- `proofs/v060/oracle/.gitignore`
- `proofs/v060/oracle/build_and_run.sh`
- `proofs/v060/oracle/dump_to_json.py`
- `proofs/v060/oracle/pxsfclay_oracle_driver.f90`
- `proofs/v060/savepoints/*`
- `proofs/v060/savepoints_fp64/*`
- `proofs/v060/pxsfclay_savepoint_parity_report.json`
- `.agent/reviews/2026-06-03-gpt-v060-pxsfclay.md`

## Commands Run

- `taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh`
- `python -m py_compile src/gpuwrf/physics/sfclay_pleim_xiu.py`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu PYTHONPATH=src pytest -q tests/test_v060_sfclay_pleim_xiu.py`
- `taskset -c 0-3 env PYTHONPATH=src python -m gpuwrf.contracts.physics_interfaces`
- `taskset -c 0-3 env PYTHONPATH=src python -m gpuwrf.contracts.physics_registry`
- `taskset -c 0-3 python .agent/skills/writing-gpu-kernels/scripts/static_kernel_check.py src/gpuwrf/physics/sfclay_pleim_xiu.py tests/test_v060_sfclay_pleim_xiu.py`
- `taskset -c 0-3 env PYTHONPATH=src pytest -q tests/contracts/test_v060_physics_interfaces.py`
- `python -m json.tool proofs/v060/pxsfclay_savepoint_parity_report.json`

## Proof Objects Produced

- `proofs/v060/pxsfclay_savepoint_parity_report.json`
  - Verdict: `PASS`
  - Canonical WRF default-real savepoint pass: `true`
  - WRF `-fdefault-real-8` precision audit pass: `true`
  - Oracle source checksum: `cd1b2095f8e093d4bc423ebbc4a3bfbce58a7d17b34f1f41d770c6061b68d0a4`
  - `full_wrf_exe`: `false`; the oracle compiles and calls unmodified
    `/home/enric/src/wrf_pristine/WRF/phys/module_sf_pxsfclay.F`, but is not a
    coupled `wrf.exe` run.
- Savepoints:
  - `proofs/v060/savepoints/pxsfclay_case_{1..6}.json`
  - `proofs/v060/savepoints_fp64/pxsfclay_case_{1..6}.json`
- Build/checksum manifests:
  - `proofs/v060/savepoints/pxsfclay_build_manifest.txt`
  - `proofs/v060/savepoints/pxsfclay_wrf_source_checksums.txt`
  - `proofs/v060/savepoints_fp64/pxsfclay_build_manifest.txt`
  - `proofs/v060/savepoints_fp64/pxsfclay_wrf_source_checksums.txt`

## Unresolved Risks

- This is a standalone WRF-module savepoint oracle, not a full coupled
  `wrf.exe` savepoint. The report carries `full_wrf_exe=false`.
- `PXSFCLAY` itself does not output `T2/TH2/Q2`; the oracle driver derives them
  with the post-surface-layer WRF `module_surface_driver.F` diagnostic formula
  from `CHS2/CQS2`, `HFX/QFX`, and `QSFC`.
- WRF fp32 has one neutral-boundary `REGIME` label dust case where `BR` is
  `O(1e-6)` and the branch label flips between 2 and 3. The predeclared rule
  accepts only 2/3 label differences when `abs(BR) <= 2e-6`; continuous fields
  remain tolerance-gated. The fp64 audit has exact `REGIME` parity.
- Not wired into runtime dispatcher or frozen registry metadata; lane ownership
  explicitly excluded frozen interface edits.

## Next Decision Needed

Manager review/merge decision, then a separate manager-owned integration patch
can decide whether to expose this module through the operational surface-layer
dispatcher and reconcile the S0 placeholder owner name for option 7.
