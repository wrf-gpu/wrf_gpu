# Phase B File-Ownership Map (FROZEN — Gate-1)

Status: FROZEN at Gate-1 (commit `f7b9fcb`). Paths verified against the actual
tree on `worker/opus/gate1`.

Rule: a lane works in an **isolated worktree** and edits ONLY files it OWNS.
SHARED-CORE files require a manager merge (no lane edits them directly). If a lane
believes it needs a SHARED-CORE change, it raises it to the manager (see
`coupler_interface.md` §6 for the already-known ones).

Two lanes must never edit the same file. The lists below are disjoint by
construction.

------------------------------------------------------------------------------
## SHARED-CORE — manager merge only (NO lane edits)
------------------------------------------------------------------------------

- `src/gpuwrf/runtime/operational_mode.py` — the RK step + `_physics_boundary_step*` + the physics/boundary call order + radiation cadence split.
- `src/gpuwrf/runtime/operational_state.py` — `OperationalCarry` (scan carry).
- `src/gpuwrf/contracts/state.py` — `State`, `Tendencies`, `BaseState`, `BoundaryState`, shapes.
- `src/gpuwrf/contracts/precision.py` — `STATE_FIELD_ORDER`, `PRECISION_MATRIX`, dtypes.
- `src/gpuwrf/contracts/grid.py`, `contracts/halo.py` — grid/metrics/halo contracts.
- `src/gpuwrf/coupling/physics_couplers.py` — **SHARED-CORE but lane-extensible by SECTION.** The adapter *call order* and shared helpers (`_to_columns`, `_field_dtype`, `_temperature_from_theta`, `_rho_from_state`, `_column_dz_from_state`, `_grid_lat_lon`, `_compute_coszen`) are manager-owned. Each lane MAY edit the body of its OWN adapter(s) in this file (B1: `thompson_adapter` + `_thompson_*`; B2: `surface_adapter`, `mynn_adapter`, `_apply_surface_flux_bottom_bc`, `_surface_*`; B3: `rrtmg_adapter`, `rrtmg_radiation_diagnostics`, `_rrtmg_*`, `_surface_radiation_properties`). **Because this is one shared file, adapter-body edits land via manager merge of the lane's worktree diff to avoid textual collisions.** Lanes should keep their kernel logic in their owned `physics/*` files and keep the adapter thin.
- `src/gpuwrf/dynamics/**` — the closed dry dycore. NO lane edits.
- `src/gpuwrf/validation/savepoint_schema.py`, `savepoint_io.py`, `tolerance_ladder.json`, `phase_b_savepoint.py` — shared oracle schema/loader/ladder.
- `src/gpuwrf/diagnostics/comprehensive_harness.py` — shared diagnostic harness.

------------------------------------------------------------------------------
## B1 — Thompson microphysics
------------------------------------------------------------------------------

OWNS (full edit rights in its worktree):
- `src/gpuwrf/physics/thompson_column.py`
- `src/gpuwrf/physics/thompson_constants.py`
- `src/gpuwrf/physics/thompson_saturation.py`
- `src/gpuwrf/physics/thompson_tables.py`
- `src/gpuwrf/physics/thompson_column_debug_stripped.py`
- `src/gpuwrf/validation/tier1_thompson.py`
- `src/gpuwrf/validation/tier2_thompson.py`
- adapter body of `thompson_adapter` + `_thompson_*` / `_state_from_thompson_output` / `_thompson_tendency_side_channel` in `physics_couplers.py` (via manager merge — shared file)

READS (must not edit): `State` moisture + `theta`/`p`; precip accumulators.

------------------------------------------------------------------------------
## B2 — surface layer + MYNN PBL (+ bottom-BC adapter)
------------------------------------------------------------------------------

OWNS:
- `src/gpuwrf/physics/surface_layer.py`
- `src/gpuwrf/physics/surface_constants.py`
- `src/gpuwrf/physics/mynn_pbl.py`
- `src/gpuwrf/physics/mynn_constants.py`
- `src/gpuwrf/physics/mynn_surface_stub.py`
- `src/gpuwrf/physics/tridiagonal_solver.py` (PBL implicit vertical solve)
- `src/gpuwrf/validation/tier1_mynn.py`
- `src/gpuwrf/validation/tier2_mynn.py`
- adapter bodies of `surface_adapter`, `mynn_adapter`, `_apply_surface_flux_bottom_bc`, `_surface_*` in `physics_couplers.py` (via manager merge — shared file)

READS: winds, theta, qv, p, geopotential (dz), prescribed land fields, `qke`.
WRITES: surface flux handles (`ustar,theta_flux,qv_flux,tau_u,tau_v,rhosfc,fltv`)
then PBL prognostics (`u,v,w,theta,qv,qke`). MUST emit operational diagnostics
HFX, LH, PBLH, T2, U10, V10 (see `coupler_interface.md` §4).

Coordination: MYNN's C-grid wind reconstruction at non-periodic edges interacts
with B4 lateral boundaries (`coupler_interface.md` §6 item 4).

------------------------------------------------------------------------------
## B3 — RRTMG radiation + land / diurnal driver
------------------------------------------------------------------------------

OWNS:
- `src/gpuwrf/physics/rrtmg_sw.py`
- `src/gpuwrf/physics/rrtmg_lw.py`
- `src/gpuwrf/physics/rrtmg_constants.py`
- `src/gpuwrf/physics/rrtmg_tables.py`
- `src/gpuwrf/physics/noah_mp.py` (land surface, if/when the land driver evolves `t_skin`)
- `src/gpuwrf/validation/tier1_rrtmg.py`
- `src/gpuwrf/validation/tier2_rrtmg.py`
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py`
- adapter bodies of `rrtmg_adapter`, `rrtmg_radiation_diagnostics`, `_rrtmg_*`, `_surface_radiation_properties`, and the diurnal helpers `_grid_lat_lon`/`_compute_coszen` in `physics_couplers.py` (via manager merge — shared file; `_compute_coszen`/`_grid_lat_lon` are shared helpers, so B3 coordinates on them)

READS: theta/p, cloud species, dz, t_skin, lu_index, lat/lon + model time.
WRITES: `theta` (heating). Emits SWDOWN, GLW (+ SWUP/GLW_up, coszen) diagnostics.

------------------------------------------------------------------------------
## B4 — static fields + lateral boundaries + IO
------------------------------------------------------------------------------

OWNS:
- `src/gpuwrf/coupling/boundary_apply.py`
- `src/gpuwrf/io/boundary_replay.py`
- `src/gpuwrf/io/land_state.py`
- `src/gpuwrf/io/gen2_accessor.py`
- `src/gpuwrf/io/gen2_wrfout_loader.py`
- `src/gpuwrf/io/data_inventory.py`
- `src/gpuwrf/io/wrfout_writer.py`
- `src/gpuwrf/io/validation.py`
- `src/gpuwrf/integration/d02_replay.py`, `integration/daily_pipeline.py`

READS/WRITES: the `*_bdy` leaves + relaxation-zone interior fields; prescribed
static fields at init (`xland,lakemask,mavail,roughness_m,lu_index,t_skin,
soil_moisture`, terrain/eta). Emits PSFC, TSK diagnostics.

Coordination: `boundary_apply.py` is called from SHARED-CORE
`operational_mode.py:1453`; B4 owns the boundary-application body but the call
site/guard is manager-owned.

------------------------------------------------------------------------------
## Collision-free guarantee
------------------------------------------------------------------------------

- Owned `physics/*`, `io/*`, and per-scheme `validation/tier{1,2}_*` files are
  disjoint across lanes — safe for fully parallel worktrees.
- The ONE genuinely shared editable file is `physics_couplers.py` (each lane's
  adapter body). The manager serializes those small adapter-body diffs at merge.
  Lanes are instructed to keep adapters thin and push logic into owned files to
  minimize that contention.
- `validation/tier2_coupled.py`, `tier3_coupled.py`, `tier4_*` (whole-pipeline)
  are SHARED-CORE recomposition tests — manager-owned.
