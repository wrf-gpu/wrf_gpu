# Phase B Coupler / State Interface Contract (FROZEN — Gate-1)

Status: FROZEN at Gate-1 (`worker/opus/gate1`, from HEAD `f7b9fcb`).
Owner: manager (shared-core changes require manager merge — see `file_ownership.md`).
Consumers: physics lanes B1 (Thompson), B2 (surface + MYNN), B3 (RRTMG + land/diurnal driver), B4 (static + lateral boundaries).

This document freezes HOW each physics scheme injects tendencies/updates into the
operational forecast loop, WHICH `State` fields each scheme reads and writes, and
WHICH surface/radiation diagnostics each scheme must expose. It is grounded in the
ACTUAL code on this branch; every claim cites `file:line`.

------------------------------------------------------------------------------
## 1. Where physics couples: the operational RK1 physics bundle
------------------------------------------------------------------------------

The single operational entry is `run_forecast_operational`
(`src/gpuwrf/runtime/operational_mode.py:1542`). It compiles one `jax.lax.scan`
over timesteps; each timestep is `_physics_boundary_step`
(`operational_mode.py:1475`), which calls
`_physics_boundary_step_with_limiter_diagnostics` (`operational_mode.py:1422`).

The per-step operator order (the binding cadence) is, from
`_physics_boundary_step_with_limiter_diagnostics`:

1. `_rk_scan_step` — dry dynamical core, RK3 + acoustic substeps (`operational_mode.py:1431`, body at `:1263`). **CLOSED, operational-ready, fp64, GPT-confirmed. Physics lanes do NOT touch this.**
2. Dynamics guards — `_limit_guarded_dynamics_state_with_diagnostics` + `_valid_mixing_ratio` on moisture (`:1434-1443`). Skipped when `namelist.disable_guards`.
3. **Physics block** (only when `namelist.run_physics`, `:1444-1450`), applied as a strictly-ordered chain of `State -> State` adapters:
   - `thompson_adapter(next_state, dt_s)` — B1 (`:1446`; gated additionally on `not disable_guards`).
   - `surface_adapter(next_state, dt_s)` — B2 surface layer (`:1447`).
   - `mynn_adapter(next_state, dt_s, grid)` — B2 PBL (`:1448`).
   - `rrtmg_adapter(next_state, dt_s, grid)` — B3, **only on radiation-cadence steps** (`run_radiation`, `:1449-1450`).
4. **Boundary block** (only when `namelist.run_boundary`, `:1451-1470`) — B4: `apply_lateral_boundaries(next_state, lead_seconds, dt_s, boundary_config)` (`:1453`) + a finite/positive guard pass.
5. `_enforce_operational_precision` (`:1471`) canonicalises every field back to its `DEFAULT_DTYPES` dtype (or fp64 when `namelist.force_fp64`).

### 1.1 Coupling convention (FROZEN)

The operational physics coupling is the **WRF RK1-physics-bundle / process-split**
convention, NOT a tendency-accumulation convention:

- Each physics adapter is a pure `State -> State` function applied **sequentially**
  inside one timestep, after the dycore and dynamics guards. The adapter advances
  its fields by `dt_s` internally (forward in time over the full physics timestep),
  and returns a new `State`. The next adapter sees the previous adapter's output.
- Adapters are column-batched: they `moveaxis` the State `(z, y, x)` layout to
  trailing-`z` columns, call the M5 column kernel, and `moveaxis` back
  (`physics_couplers.py:226-235`, `_to_columns`/`_from_columns`).
- Radiation runs at a coarser cadence (`namelist.radiation_cadence_steps`,
  default 60, `operational_mode.py:88`); the scan is split into non-radiation and
  radiation segments by `run_forecast_operational` (`:1568-1599`).

**Frozen contract for lanes:** a physics lane delivers (or extends) a single
`*_adapter(state, dt, [grid]) -> State` function in `physics_couplers.py`. The
adapter MUST:
- read only fields it declares in §3,
- write only fields it declares in §3 (via `state.replace(...)`),
- cast its written fields back to the frozen storage dtype with
  `_field_dtype(name)` (`physics_couplers.py:291`), exactly as the existing
  adapters do (`physics_couplers.py:602-609`, `:666-674`),
- be a no-op (return `state` unchanged) when its scheme is physically inactive on
  every column (so the diagnostic harness classifies it INACTIVE_PHYSICAL, not
  MISSING — see `savepoint_schema.md` §5 and the harness fix below).

**Lane-touch boundary:** lanes own their adapter body and their `physics/*`
implementation files (see `file_ownership.md`). The adapter *call site and order*
in `_physics_boundary_step_with_limiter_diagnostics` is SHARED-CORE: changing the
order, adding/removing an adapter, or changing the `run_physics`/`run_radiation`
gating requires a manager merge. Lanes MUST NOT change the call order.

### 1.2 Tendency vs in-place semantics (the one place lanes can get this wrong)

Two adapters already demonstrate the two legal patterns:

- **In-place state advance** (Thompson, surface, MYNN): the kernel integrates the
  process over `dt` and the adapter writes the advanced field
  (`thompson_adapter` -> `_state_from_thompson_output`, `physics_couplers.py:506`).
- **Side-channel tendency** (Thompson water budget, RRTMG diagnostics): exposed via
  `return_tendencies=True` / dedicated diagnostic functions
  (`ThompsonTendencySideChannel`, `physics_couplers.py:197`,
  `thompson_adapter_with_tendencies` `:568`; `rrtmg_radiation_diagnostics` `:677`).

The dycore tendency buffers (`Tendencies`, `state.py:291`) and `namelist.tendencies`
are reserved for the LARGE-STEP DRY dynamics path
(`_augment_large_step_tendencies`, `operational_mode.py:1066`). **Physics lanes do
NOT write into `Tendencies` / `namelist.tendencies`.** Physics couples by advancing
`State` fields in its adapter. This is frozen: it keeps the dycore tendency space
(coupled, mass-weighted) cleanly separated from the process-split physics path.

------------------------------------------------------------------------------
## 2. State field registry (FROZEN reference)
------------------------------------------------------------------------------

`State` is the SoA pytree in `src/gpuwrf/contracts/state.py:349`. Field order,
shapes, dtypes, units and stagger are frozen by:
- `STATE_FIELD_ORDER` — `src/gpuwrf/contracts/precision.py:20`,
- `PRECISION_MATRIX` (dtype + fp32-gated flag) — `precision.py:77`,
- `_state_field_shapes` — `state.py:35`,
- units/stagger docstring — `state.py:351-380`.

Grid: `nz, ny, nx` from `GridSpec` (`grid.py:317`). C-grid (Arakawa-C) staggering:
`u (nz,ny,nx+1)`, `v (nz,ny+1,nx)`, `w/ph (nz+1,ny,nx)`, mass fields `(nz,ny,nx)`,
surface fields `(ny,nx)`.

`State.replace` (`state.py:566`) auto-casts updates back to the current field dtype
unless `_cast=False`; it also keeps the `(total, legacy, perturbation)` triples
synchronized for `p`, `ph`, `mu` (`state.py:590-606`). **Lanes must update via
`state.replace(...)` and must not break those triples.** A lane that changes a
perturbation field must let `replace` recompute the total, or update both.

------------------------------------------------------------------------------
## 3. Per-scheme READ / WRITE field contract (FROZEN)
------------------------------------------------------------------------------

Derived directly from the current adapters in `physics_couplers.py`. "READ" =
fields the kernel consumes as input; "WRITE" = fields the adapter returns changed.
Lanes may not widen WRITE without a manager merge (collision risk with other lanes
running in the same chain). Lanes may add READs of fields owned by an upstream lane
in the chain (that is the legal coupling path), but must declare them here.

### B1 — Thompson microphysics  (`thompson_adapter`, physics_couplers.py:553)
- READ: `theta`, `p` (-> T via `_temperature_from_theta`), `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr` (`_thompson_column_from_state`, `:486`).
- WRITE: `theta` (latent heat), `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr` (`_state_from_thompson_output`, `:506`).
- Side channel: `ThompsonTendencySideChannel` (`qv,qc,qr,qi,qs,qg` tendencies + `column_water_tendency` + `precip_out_tendency`) via `return_tendencies=True` (`:523`).
- NOT YET WRITTEN but in-scope for B1 to add: `qg`-number `Ng`, `Ns`; precip accumulators `rain_acc`/`snow_acc`/`graupel_acc`/`ice_acc` (`state.py:420-423`, currently untouched by the adapter — B1 must wire these and declare them here).

### B2 — Surface layer  (`surface_adapter`, physics_couplers.py:646)
- READ: `u`,`v` (-> mass-point speed), `theta`, `qv`, `p`, geopotential `ph` (-> `dz`), `t_skin`, `soil_moisture`, `xland`, `lakemask`, `mavail`, `roughness_m`, `ustar` (`_SurfaceColumnState`, `:179`, build at `:650-664`).
- WRITE: `ustar`, `theta_flux`, `qv_flux`, `tau_u`, `tau_v`, `rhosfc`, `fltv` (`:666-674`).

### B2 — MYNN PBL  (`mynn_adapter`, physics_couplers.py:574)
- READ: `u`,`v`,`w` (mass-point), `theta`, `qv`, `qke`, `p`, `ph` (-> dz), and the surface-flux handles written by `surface_adapter`: `theta_flux`, `qv_flux`, `tau_u`, `tau_v`, `rhosfc` (`:577-596`, bottom-BC `_apply_surface_flux_bottom_bc` `:612`).
- WRITE: `u`, `v`, `w`, `theta`, `qv`, `qke` (`:602-609`).
- **Cross-lane dependency (frozen):** MYNN READS the surface-flux fields surface_layer WROTE earlier in the same chain. Both are B2 — no cross-lane collision. The order surface→MYNN is SHARED-CORE and frozen.

### B3 — RRTMG radiation  (`rrtmg_adapter`, physics_couplers.py:703)
- READ: `theta`,`p` (->T), `qv`,`qc`,`qi`,`qs`,`qg` (cloud), `ph` (->dz), `t_skin`, `lu_index` (albedo/emissivity LUT `_surface_radiation_properties` `:382`), grid lat/lon + model time (`_compute_coszen` `:328`).
- WRITE: `theta` (SW+LW heating-rate tendency, `:710-711`).
- Diagnostics (no State write): `RRTMGRadiationDiagnostics` (`:214`) via `rrtmg_radiation_diagnostics` (`:677`): `swdown`, `swup`, `glw`, `glw_up`, `coszen`, `surface_albedo`, `surface_emissivity`.

### B4 — Static fields + lateral boundaries
- Static loaders populate the prescribed fields (`xland`, `lakemask`, `mavail`, `roughness_m`, `lu_index`, `t_skin`, `soil_moisture`, terrain/eta in `GridSpec`) at INIT, not in the timestep loop. These are READ by B2/B3.
- Lateral boundaries: `apply_lateral_boundaries(state, lead_seconds, dt, boundary_config)` (`coupling/boundary_apply.py`, called `operational_mode.py:1453`).
  - READ: `*_bdy` leaves (`u_bdy,v_bdy,w_bdy,theta_bdy,qv_bdy,ph_bdy,mu_bdy,p_bdy,pb_bdy,phb_bdy,mub_bdy`, `state.py:424-434`) + the corresponding interior fields.
  - WRITE: `u`,`v`,`w`,`theta`,`qv`,`p`/`p_total`/`p_perturbation`,`ph`/`ph_total`/`ph_perturbation`,`mu`/`mu_total`/`mu_perturbation` in the relaxation zone (guarded at `operational_mode.py:1457-1470`).

------------------------------------------------------------------------------
## 4. Diagnostic outputs to expose (FROZEN — M9 operational divergence-map set)
------------------------------------------------------------------------------

The M9 operational gate compares these surface fields hour-by-hour vs WRF
(`proofs/m9/divergence_map.json`). Each lane that owns a source field MUST expose a
host-readable diagnostic with the name/units/stagger/shape below. Surface fields are
mass-point 2-D `(ny, nx)`. These are diagnostics — they live in a side-channel
(like `RRTMGRadiationDiagnostics`), NOT new prognostic `State` leaves, unless a leaf
already exists.

| Diag  | Long name                       | Units   | Stagger | Shape   | Owner | Source                                              |
|-------|---------------------------------|---------|---------|---------|-------|-----------------------------------------------------|
| SWDOWN| downward SW flux at surface     | W m^-2  | mass    | (ny,nx) | B3    | `RRTMGRadiationDiagnostics.swdown` (couplers:696)   |
| GLW   | downward LW flux at surface     | W m^-2  | mass    | (ny,nx) | B3    | `RRTMGRadiationDiagnostics.glw` (couplers:698)      |
| HFX   | upward sfc sensible heat flux   | W m^-2  | mass    | (ny,nx) | B2    | from `theta_flux`*`rhosfc`*cp (surface_adapter)     |
| LH    | upward sfc latent heat flux     | W m^-2  | mass    | (ny,nx) | B2    | from `qv_flux`*`rhosfc`*Lv (surface_adapter)        |
| PBLH  | PBL height                      | m       | mass    | (ny,nx) | B2    | MYNN diagnostic (mynn_adapter — to expose)          |
| TSK   | skin temperature                | K       | mass    | (ny,nx) | B4/B2 | `State.t_skin` (replay/land driver)                 |
| T2    | 2 m temperature                 | K       | mass    | (ny,nx) | B2    | surface-layer diagnostic (to expose)                |
| U10   | 10 m zonal wind                 | m s^-1  | mass    | (ny,nx) | B2    | surface-layer diagnostic (to expose)                |
| V10   | 10 m meridional wind            | m s^-1  | mass    | (ny,nx) | B2    | surface-layer diagnostic (to expose)                |
| PSFC  | surface pressure                | Pa      | mass    | (ny,nx) | B4    | from `mu_total`+`pb` column-bottom (dycore/IO)      |

Notes:
- `HFX`/`LH` derive from the kinematic fluxes already written by `surface_adapter`
  (`theta_flux` K m s^-1, `qv_flux` kg kg^-1 m s^-1, `rhosfc` kg m^-3,
  `state.py:367-369`). B2 must add the host-side W m^-2 conversion + expose them.
- `PBLH`, `T2`, `U10`, `V10` are NOT yet emitted; B2 owns adding them as
  side-channel diagnostics (do NOT add prognostic State leaves for them).
- `TSK` is prescribed/data-replayed today (`State.t_skin`, bitwise-matched per
  `divergence_map.json`); a future land driver (B3 scope) may evolve it.
- `coszen` (`RRTMGRadiationDiagnostics.coszen`) is exposed for the diurnal driver.

------------------------------------------------------------------------------
## 5. Precision boundary at the coupling interface (FROZEN)
------------------------------------------------------------------------------

`_enforce_operational_precision` (`operational_mode.py:299`) runs once at the END of
every timestep and canonicalises each field to `DEFAULT_DTYPES.dtype_for(field)`
(ADR-007 `PRECISION_MATRIX`, `precision.py:77`). Implications for lanes:

- Inside a kernel, compute in whatever precision is correct (the column kernels
  generally upcast to fp64 for accumulation); on the way OUT of the adapter, cast
  written fields to `_field_dtype(name)` (couplers do this already).
- fp32-gated fields (`u,v,theta,qv,qc..qg,Ni..Ng,qke,xland,lakemask,mavail`,
  `*_bdy`) are `FP32_GATED`; mass/pressure/geopotential/surface-stability/accum
  fields are FP64-locked (`precision.py:79-137`).
- Idealized dycore cases set `force_fp64=True` (`operational_mode.py:110`,
  enforced both at the public entry `:1559` and per-step `:1471`). Physics lanes
  must tolerate either precision regime; do not hard-pin a dtype.

------------------------------------------------------------------------------
## 6. Open interface decisions (manager: confirm before launching lanes)
------------------------------------------------------------------------------

1. **Diagnostic side-channel transport.** SWDOWN/GLW/HFX/LH/PBLH/T2/U10/V10 are
   needed as host-readable per-step (or per-output-cadence) fields, but the
   operational scan (`run_forecast_operational`) returns only the final `State`
   (no per-step side outputs). Two options: (a) add a `diagnostics` pytree to the
   scan carry (like `OperationalCarry` + the limiter-diagnostics scan variant
   `operational_mode.py:1604`), emitted at output cadence; or (b) recompute the
   surface diagnostics from the saved `State` snapshots at I/O time (cheaper, no
   carry change, but PBLH/T2/U10/V10 need the surface-layer/PBL internal state at
   that step). **Recommendation: (a) — a `DiagnosticsCarry` analogous to the
   existing limiter-diagnostics scan.** This is a SHARED-CORE change the manager
   should make, not a lane, since it touches the scan signature.

2. **`Tendencies` width.** `Tendencies` (`state.py:291`) has only dry-dynamics
   leaves (`u,v,w,theta,qv,p,ph,mu`). Confirmed FROZEN as dry-only; physics does
   NOT use it. If any lane truly needs a coupled physics tendency into the dycore
   (it should not, given the process-split convention), that is an ADR, not a lane
   change.

3. **Precip accumulators (`rain_acc` etc.).** Present in `State` but untouched by
   `thompson_adapter`. B1 must wire them; confirm they are simple per-step
   accumulators (`+=`) so they do not need to be in the carry's "save" family.

4. **MYNN writes `u,v,w`.** MYNN currently rewrites the C-grid winds via
   mass-point round-trip (`_mass_to_u_face` etc., periodic assumption,
   couplers:256-274). For real (non-periodic) cases B2 must confirm the face
   reconstruction is correct at domain edges (interacts with B4 boundaries). Flag
   for B2/B4 coordination.
