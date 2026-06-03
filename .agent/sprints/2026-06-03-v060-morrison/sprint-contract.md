# Sprint Contract: v0.6.0 Morrison 2-moment microphysics JAX port (mp_physics=10)

Date: 2026-06-03
Branch: `worker/opus/v060-morrison`
Frontrunner: Opus 4.8 MAX
Lane: v0.6.0 per-scheme lane #4 (Morrison two-moment) per `.agent/decisions/V0.6.0-S0-PLAN.md`

## Objective

Port WRF Morrison 2-moment bulk microphysics (`phys/module_mp_morr_two_moment.F`,
`mp_physics=10`) to JAX, WRF-faithful, savepoint-parity-gated. Use the proven
per-scheme lane pattern (standalone Fortran oracle driving the UNMODIFIED WRF
scheme -> per-column gold savepoints across regimes -> faithful JAX port ->
predeclared-tolerance parity gate). No new State leaves: Morrison's number
species `Ni/Nr/Ns/Ng` pre-exist in State per S0.

## Scope (frozen)

1. ORACLE: Fortran factory compiling the unmodified Morrison source + real deps
   (`module_mp_radar.F`, `share/module_model_constants.F`) with a minimal
   `module_wrf_error` stub (scheme references no module-scope name from it).
   6 regimes, fp32 (canonical) + fp64 builds, source checksums recorded.
2. PORT: JAX kernel faithful to the WRF process order (setup, subsaturation
   removal, T>=273.15 warm branch, T<273.15 cold branch, saturation adjustment,
   split-step sedimentation, final update + instantaneous freeze/melt + Reff),
   constant droplet number (iinum=1), graupel mode (IHAIL=0, IGRAUP=0).
3. PARITY GATE: predeclared per-field tolerances (frozen up front), honest
   verdict, fp64 reference for the fields where the fp32 reference leaves
   sedimentation-NSTEP / threshold round-off dust, vs-fp32 recorded.

## File ownership (created, no shared-file edits)

- `src/gpuwrf/physics/microphysics_morrison.py` (main kernel + adapter)
- `src/gpuwrf/physics/_morrison_cold.py` (cold-branch process rates)
- `src/gpuwrf/physics/_morrison_sed.py` (sedimentation + finalize)
- `src/gpuwrf/physics/morrison_constants.py` (init constants)
- `proofs/v060/oracle/{morrison_oracle_driver.f90, morrison_stub_modules.f90, build_and_run.sh, dump_to_json.py}`
- `proofs/v060/savepoints{,_fp64}/morrison_case_*.json` + checksums
- `proofs/v060/run_morrison_parity.py`, `proofs/v060/morrison_savepoint_parity_report.json`
- `tests/savepoint/test_morrison_parity.py`
- `src/gpuwrf/physics/__init__.py` (lazy export of morrison_run/morrison_tendency only)

NOT touched: frozen interfaces, other schemes, State.__slots__, registry.

## Resource

Cores 0-3 (`taskset -c 0-3`), JAX on CPU (`JAX_PLATFORM_NAME=cpu`), no GPU.

## Acceptance criteria

- Oracle compiles the UNMODIFIED Morrison source; 6 regimes; checksums recorded;
  `full_wrf_exe=False` flag honest in the report.
- JAX fp64 port reproduces the fp64 oracle to a machine-precision band on EVERY
  field (theta, qv, qc, qr, qi, qs, qg, Ni, Ns, Nr, Ng) + surface precip across
  all 6 regimes (BINDING faithfulness gate).
- Prognostic mass + surface precip vs the canonical fp32 oracle within
  predeclared physical tolerance (BINDING operational gate).
- Adapter returns a frozen `PhysicsTendency` validating against the S0 interface.
- Parity report `overall_pass=true`; pytest passes.

## Proof objects

- `proofs/v060/morrison_savepoint_parity_report.json`
- `proofs/v060/savepoints{,_fp64}/` gold savepoints + `wrf_source_checksums.txt`
- `tests/savepoint/test_morrison_parity.py`

## Result: PASS

All 6 regimes pass the fp64 faithfulness gate (~1e-13..1e-16 rel) and the fp32
mass/precip operational gate. Number-vs-fp32 in the extreme graupel-core column
(case 4 Ns) is documented non-blocking fp32 reference dust (fp64 matches to
8.3e-15). Batched multi-column execution is bit-identical to per-column.
