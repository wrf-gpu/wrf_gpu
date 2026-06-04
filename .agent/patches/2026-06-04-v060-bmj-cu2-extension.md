# Patch: Extend v0.6.0 S0 Cumulus Menu with BMJ (`cu_physics=2`)

Date: 2026-06-04
Author: GPT-5.5 xhigh
Status: implementation patch pending manager review
Policy: `AGENTS.md` stable contract patch protocol

## Scope

Adds WRF Betts-Miller-Janjic cumulus (`cu_physics=2`) as an explicit extension
to the frozen v0.6.0 S0 cumulus menu.

## Rationale

The frozen S0 contract accepted `cu_physics={0,1,3,6,16}`. The principal sprint
contract for this lane explicitly requests BMJ (`cu_physics=2`) be ported and
registered, with the extension noted as a frozen-contract extension.

## Implemented Code Surfaces

- `src/gpuwrf/contracts/physics_registry.py`
  - Adds `ACCEPTED_CU_PHYSICS=(0,1,2,3,6,16)`.
  - Adds BMJ scheme metadata.
  - Adds BMJ tendency members `rthcuten,rqvcuten`.
  - Adds BMJ carry member `cldefi`.
- `src/gpuwrf/contracts/physics_interfaces.py`
  - Adds `PhysicsStepSpec(family="cumulus", option=2, name="Betts-Miller-Janjic")`.
- `src/gpuwrf/io/namelist_check.py`
  - Accepts and reports `2=Betts-Miller-Janjic`.
- `src/gpuwrf/coupling/physics_dispatch.py`
  - Routes `cu=2` to `gpuwrf.physics.cumulus_bmj.step_bmj_column`.
- `src/gpuwrf/coupling/scan_adapters.py`
  - Adds `CU_SCAN_ADAPTERS[2]` and BMJ `CLDEFI` carry seeding.
- `src/gpuwrf/runtime/operational_mode.py`
  - Accepts scan-wired `cu=2` and threads BMJ carry separately from KF carry.

## Evidence Required Before Merge

- `proofs/v060/bmj_savepoint_parity.json` against unmodified
  `/home/enric/src/wrf_pristine/WRF/phys/module_cu_bmj.F`.
- CPU-only import/interface tests:
  - `python -m gpuwrf.contracts.physics_registry`
  - `python -m gpuwrf.contracts.physics_interfaces`
  - `pytest -q tests/contracts/test_v060_physics_interfaces.py tests/test_namelist_check.py tests/test_v060_physics_dispatch.py`

## Risk

The extension is mechanically small, but BMJ's algorithm is not a mass-flux
scheme. Merge should depend on the parity proof verdict, not on registry
presence alone.
