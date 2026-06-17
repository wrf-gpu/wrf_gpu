# GPT critic final proof: v0.18 RA tail family

Branch: `worker/gpt/v018-ra`

Final VERDICT: **ACCEPT** (original critic verdict was FIX-then-ACCEPT; both
must-fixes are now resolved).

## Path Taken

No clean full-`wrf.exe` rebuild was used. Per manager decision, the dirty
provenance was audited as the project-standard `WRFGPU2_ORACLE` savepoint
instrumentation.

Audit result: **physics-pristine, oracle-instrumented, documented**.

- Exact RA modules for 3/5/7/99 are upstream-identical to WRF HEAD:
  `module_ra_cam.F`, `module_ra_goddard.F`, `module_ra_flg.F`,
  `module_ra_gfdleta.F`.
- `module_radiation_driver.F` has dump-only `WRFGPU2_ORACLE` hooks: a `USE`
  line plus guarded `oracle_open_scheme` / `oracle_dump2d` /
  `oracle_dump3d` / `oracle_close_scheme` calls. The existing radiation call
  argument lists are not edited, and the dumper arrays are `INTENT(IN)`.
- Raw oracle run logs, raw `wrfout` files, compact savepoints, exact module
  checksums, the instrumented driver checksum, and the dumper checksum are now
  committed in `proofs/v018/savepoints/ra_tail_wrf/raw_hash_manifest.txt`.
- Report wording was corrected from "unmodified wrf.exe" to
  "physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe".

## Must-Fixes Resolved

1. Oracle provenance:
   - Added `proofs/v018/savepoints/ra_tail_wrf/raw_hash_manifest.txt`.
   - Updated `wrf_source_checksums.txt` to name the instrumentation module and
     clarify that the exact RA modules are upstream-identical while the driver
     is oracle-instrumented.
   - Regenerated RA 3/5/7/99 compact savepoint JSONs from the raw `wrfout`
     files so their `exact_module_rule` records the corrected provenance.

2. Metadata:
   - Updated `src/gpuwrf/contracts/physics_interfaces.py` so RA 5 and RA 99
     cite the v0.18 exact-driver real-WRF savepoints:
     `ra5_wrf_real.json` and `ra99_wrf_real.json`.
   - Added contract coverage asserting all RA 3/5/7/99 LW/SW specs cite their
     v0.18 exact-driver savepoint and remain `REFERENCE-ONLY`.

## Commands Run

- `python proofs/v018/oracle/ra_tail_wrf/dump_ra_oracle.py --scheme N --run-dir proofs/v018/oracle/ra_tail_wrf/run/raN --out proofs/v018/savepoints/ra_tail_wrf/raN_wrf_real.json` for `N=3,5,7,99`
- `python proofs/v018/oracle/ra_tail_wrf/make_ra_family_status.py`
- `PYTHONPATH=src pytest -q tests/test_v018_ra_tail_oracle.py tests/test_scheme_catalog_fail_closed.py tests/test_namelist_check.py tests/contracts/test_v060_physics_interfaces.py`
- `scripts/with_gpu_lock.sh --timeout 1800 --label gpt-v018-ra-fix -- env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; print(jax.default_backend()); print([str(d) for d in jax.devices()])"`

## Proof Results

- RA-tail/catalog/namelist/interface CPU gate: `90 passed in 3.26s`.
- GPU smoke via lock: backend `gpu`, device `cuda:0`.
- `proofs/v018/ra_family_status.json`: `full_ship_gate = true`,
  `oracle_raw_hash_manifest = proofs/v018/savepoints/ra_tail_wrf/raw_hash_manifest.txt`.

## Out Of Scope

I did not investigate the operational-radiation NaN blocker. That remains owned
by the separate worker named in the sprint instruction.
