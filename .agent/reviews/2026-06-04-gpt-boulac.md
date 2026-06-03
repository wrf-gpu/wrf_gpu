# GPT BouLac PBL Handoff — 2026-06-04

## Objective

Port WRF Bougeault-Lacarrere PBL (`bl_pbl_physics=8`) as a pure-JAX,
`jit`/`vmap`-able column kernel, validate it against an unmodified pristine WRF
`module_bl_boulac.F` oracle, and extend the v0.6.0 frozen PBL menu to accept and
scan-wire `bl=8`.

## Files Changed

- `src/gpuwrf/physics/pbl_boulac.py`
- `src/gpuwrf/coupling/scan_adapters.py`
- `src/gpuwrf/coupling/physics_dispatch.py`
- `src/gpuwrf/contracts/physics_registry.py`
- `src/gpuwrf/contracts/physics_interfaces.py`
- `src/gpuwrf/io/namelist_check.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/contracts/test_v060_physics_interfaces.py`
- `tests/test_namelist_check.py`
- `proofs/v060/oracle/boulac_*`
- `proofs/v060/savepoints/boulac_*`
- `proofs/v060/run_boulac_parity.py`
- `proofs/v060/boulac_pbl_savepoint_parity.json`

## Commands Run

- `taskset -c 0-3 bash proofs/v060/oracle/boulac_build_and_run.sh`
- `taskset -c 0-3 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true JAX_COMPILATION_CACHE_DIR=/tmp/gpuwrf_boulac_jax_cache PYTHONPATH=src python proofs/v060/run_boulac_parity.py`
- `taskset -c 0-3 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src python -m py_compile src/gpuwrf/physics/pbl_boulac.py src/gpuwrf/coupling/scan_adapters.py src/gpuwrf/contracts/physics_registry.py src/gpuwrf/contracts/physics_interfaces.py src/gpuwrf/coupling/physics_dispatch.py src/gpuwrf/io/namelist_check.py src/gpuwrf/runtime/operational_mode.py proofs/v060/run_boulac_parity.py`
- `taskset -c 0-3 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src pytest -q tests/contracts/test_v060_physics_interfaces.py tests/test_namelist_check.py tests/test_v060_physics_dispatch.py --tb=short`
- `taskset -c 0-3 python scripts/validate_agentos.py`

## Proof Objects Produced

- `proofs/v060/boulac_pbl_savepoint_parity.json`
  - Verdict: `PASS`
  - Worst residual: `EXCH_H max_abs = 2.2737367544323206e-13`
  - Predeclared tolerances: tendencies/TKE/exchange `abs=5e-10, rel=5e-10`; PBLH `abs=5e-9, rel=5e-12`
  - WRF source: `/home/enric/src/wrf_pristine/WRF/phys/module_bl_boulac.F`
  - WRF source SHA256: `68285f1457fe62e37ad1d76680d3c69e4a48a6baca93348f6f8a15c5fb40d871`
- `proofs/v060/savepoints/boulac_case_{1..6}.json`
- `proofs/v060/savepoints/boulac_wrf_source_checksums.txt`

## Unresolved Risks

- Oracle is a short standalone WRF-module driver compiled from unmodified
  `module_bl_boulac.F` with fp64 default real promotion; it is not a full coupled
  `wrf.exe` savepoint dump.
- Operational `bl=8` reuses the existing `State.qke` leaf as BouLac prognostic
  TKE storage to avoid a `State.__slots__`/restart schema change. This is a
  deliberate frozen-contract extension and should be reviewed at merge with
  concurrent MYJ/PBL menu changes.
- The scan adapter consumes revised-MM5 surface-layer `HFX/QFX/UST` forcing, same
  assembler pattern as YSU/ACM2. Integrated forecast skill/behavior is not claimed
  by this savepoint parity proof.

## Next Decision Needed

Merge conflict policy for concurrent PBL lanes (`bl=1/2/7/8`) in the shared
registry, dispatcher, and scan adapter tables.
