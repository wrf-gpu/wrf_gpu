# Sprint Contract — v0.3.0 S0 (PLAN + SCHEMA FREEZE + recon) [GATE]

## Objective
Unblock the parallel v0.3.0 implementation lanes by (1) freezing the
metgrid-equivalent artifact schema, (2) resolving the two input TBDs, (3) producing
exact, file-disjoint S1–S5 sprint contracts with predeclared oracles/tolerances and
the AIFS→met_em variable map. No ingest implementation.

## Non-Goals
- No forcing decode, geog, interp, parity, or integration code (those are S1–S5).
- No GPU work; no edits to existing `src/gpuwrf` model code.

## File Ownership (this sprint produced these — now FROZEN)
- `src/gpuwrf/init/__init__.py`, `src/gpuwrf/init/metgrid_schema.py` — FROZEN schema.
- `proofs/v030/RECON.md`, `proofs/v030/recon_inventory.py`,
  `proofs/v030/recon_inventory.json` — recon.
- `.agent/sprints/2026-06-02-v030-s{0..5}-*/sprint-contract.md` — the contracts.
- `.agent/sprints/2026-06-02-v030-s0-plan-schema-freeze/artifacts/file-ownership-map.md`.

## Inputs
Sprint plan `.agent/decisions/SPRINT-PLAN-V030-V040.md`; master plan
`.agent/decisions/MASTER-PLAN-POST-V020.md`; oracle
`/mnt/data/canairy_meteo/runs/wps_cases/<case>/l3/`; AIFS GRIB
`<case>/ungrib/step_NNN.grib2`; WPS source
`/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/`; real.exe
`/home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F`.

## Acceptance Criteria
- [x] Both TBDs resolved with on-disk paths (RECON.md §1).
- [x] Frozen schema imports CPU-only; validates against real oracle dims; rejects
  shape violations; every schema field present in the oracle.
- [x] S1–S5 contracts written with DISJOINT file ownership + predeclared tolerances.
- [x] AIFS→met_em map with gaps flagged (RECON.md §3).

## Validation Commands
```
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 proofs/v030/recon_inventory.py
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 -c "import sys;sys.path.insert(0,'src');\
from gpuwrf.init.metgrid_schema import metem_field_specs;print(len(metem_field_specs()),'specs')"
```

## Proof Object
`proofs/v030/RECON.md` + `recon_inventory.json` + the frozen
`src/gpuwrf/init/metgrid_schema.py` (importable, self-validating).

## Risks
- BRANCH DIVERGENCE: schema/contracts land on `worker/opus/v030-s0` (base
  `worker/opus/v030-native-init`@b301baa = v0.2.0 consolidation), but the v0.3.0
  sprint plan was committed on `v020-validation-gate`@370f948 (a DIFFERENT
  lineage). Manager must reconcile the true v0.3.0 base before merging S1–S5
  (recommendation: rebase the plan commit onto the consolidated base, since the
  impl lanes need the consolidated `src/`). See handoff.

## Handoff Requirements
Frozen schema summary, S1–S5 contracts, unblocked-lane map, branch-divergence flag.
