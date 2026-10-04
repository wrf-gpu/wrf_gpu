# G3 Report - Urban BEP/BEM + Lake

## Outcome

G3 is delivered as **REFERENCE_ONLY / fail-closed**, not as an operational
physics port. This is the honest outcome required by the sprint brief: BEP,
BEP+BEM, and WRF lake are large source-specific WRF schemes requiring new state
carry, static tables, fresh WRF single-column savepoints, and faithful kernels.

No numerical WRF parity is claimed. `gfortran` is available through the
`wrfbuild` conda environment (`GNU Fortran (conda-forge gcc 14.3.0-19) 14.3.0`),
and the proof includes a compile-only probe for `module_sf_lake.F`,
`module_bep_bem_helper.f90`, `module_sf_urban.F`, `module_sf_bep.F`, and
`module_sf_bem.F` (all rc=0). A fresh numerical standalone oracle was deferred
because the schemes need the full WRF initialization/static-table/state ABI
(urban maps/BEM tables, `bepscheme`/`bep_bemscheme` carry, `lakeini` and
lake/snow/soil/lake-column carry) before a single-column driver can produce
meaningful WRF reference outputs.

## Per-Scheme Verdict

| Scheme | Namelist | Status | Oracle result | Coverage | Limitation |
| --- | --- | --- | --- | --- | --- |
| BEP urban canopy | `sf_urban_physics=2` | `REFERENCE_ONLY` | `reference_only_inventory`; numerical status `standalone_numerical_oracle_deferred_wrf_dependency_depth` | WRF `module_sf_bep.F`, `module_sf_urban.F`, `module_bep_bem_helper.F` source/object/module files present; compile-only probe rc=0; Registry `bepscheme` subset match; namelist reference path accepts; operational validator and scan fail-close; stub raises | No numerical WRF single-column parity, no JAX kernel, no operational scan wiring |
| BEP+BEM urban canopy | `sf_urban_physics=3` | `REFERENCE_ONLY` | `reference_only_inventory`; numerical status `standalone_numerical_oracle_deferred_wrf_dependency_depth` | WRF BEP+BEM source/object/module files present; compile-only probe rc=0; Registry `bep_bemscheme` subset match; 57 carry members checked; namelist reference path accepts; operational validator and scan fail-close; stub raises | No numerical WRF single-column parity, no BEM building-energy state/kernel, no operational scan wiring |
| WRF lake model | `sf_lake_physics=1` | `REFERENCE_ONLY` | `reference_only_inventory`; numerical status `standalone_numerical_oracle_deferred_wrf_dependency_depth` | WRF `module_sf_lake.F` source/object/module present; compile-only probe rc=0; `sf_lake_physics` and `lake_depth` Registry terms present; 23 lake carry members inventoried; namelist reference path accepts; operational validator and scan fail-close; stub raises | No numerical WRF single-column parity, no `lakeini`/lake-column carry/kernel, no operational scan wiring |

Default `sf_urban_physics=0` and `sf_lake_physics=0` remain implemented and unchanged.

## Files Changed

- `src/gpuwrf/contracts/physics_registry.py`
- `src/gpuwrf/io/scheme_catalog.py`
- `src/gpuwrf/io/namelist_check.py`
- `docs/namelist-compatibility.md`
- `tests/test_namelist_check.py`
- `tests/test_scheme_catalog_fail_closed.py`
- `tests/test_v022_g3city_urban_lake.py`
- `proofs/v023/feature_sprints/G3_GPT_BRIEF.md`
- `proofs/v023/feature_sprints/G3_DONE`
- `proofs/v023/feature_sprints/g3_urban_lake_oracle_check.py`
- `proofs/v023/feature_sprints/g3_urban_lake_fixture_manifest.yaml`
- `proofs/v023/feature_sprints/g3_urban_lake_oracle_check.json`
- `proofs/v023/feature_sprints/G3_REPORT.md`

Note: `src/gpuwrf/physics/urban_bep_bem.py` and
`src/gpuwrf/physics/lake_model.py` were not changed; they remain fail-closed
stub modules.

## Commands Run

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  python proofs/v023/feature_sprints/g3_urban_lake_oracle_check.py

taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  python .agent/skills/building-wrf-oracles/scripts/validate_fixture_manifest.py \
  proofs/v023/feature_sprints/g3_urban_lake_fixture_manifest.yaml

taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  pytest -q tests/test_v022_g3city_urban_lake.py \
  tests/test_namelist_check.py::test_registry_records_supported_active_suite \
  tests/test_namelist_check.py::test_operational_validator_rejects_reference_only_scheme \
  tests/test_scheme_catalog_fail_closed.py::test_catalog_is_internally_consistent \
  tests/test_scheme_catalog_fail_closed.py::test_implemented_set_matches_frozen_accept_matrix \
  tests/test_scheme_catalog_fail_closed.py::test_urban_bep_bem_and_lake_are_reference_only_fail_closed_operationally

taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  pytest -q tests/test_namelist_check.py tests/test_scheme_catalog_fail_closed.py \
  tests/test_v022_g3city_urban_lake.py
```

Results:

- Oracle/fail-closed gate: PASS.
- Fixture manifest validator: PASS.
- Focused tests: 20 passed.
- Full affected test files: 76 passed.

## Proof Objects

- `proofs/v023/feature_sprints/g3_urban_lake_oracle_check.json`
- `proofs/v023/feature_sprints/g3_urban_lake_fixture_manifest.yaml`
- `proofs/v023/feature_sprints/G3_REPORT.md`

## Unresolved Risks

- There is no fresh numerical WRF-Fortran single-column output for BEP/BEM/lake
  in this lane. The compiler and module compile-only path are available; the
  missing piece is a meaningful standalone WRF driver with complete urban/lake
  initialization and carry/static ABI.
- Existing WRF object files are inventoried, but object inventory is not a
  substitute for WRF parity.
- A faithful port needs a dedicated milestone to freeze BEP/BEM/lake state
  interfaces and build real WRF savepoints before any JAX kernel work.

## Next Decision Needed

Approve a dedicated Urban/Lake oracle milestone: first build fresh pristine-WRF
single-column harnesses/savepoints for BEP, BEP+BEM, and Lake; then freeze state
carry and only then port faithful kernels.
