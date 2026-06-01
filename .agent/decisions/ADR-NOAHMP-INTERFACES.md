# ADR-NOAHMP-INTERFACES — Prognostic Noah-MP Land-Surface Port: Frozen Interfaces + Parallel-Sprint Plan

- Date: 2026-06-01
- Author: Opus 4.8 MAX (worker: `noahmp-iface`, branched off `worker/opus/final-verdict` tip `135b6dc`)
- Status: PROPOSED — interface freeze + dispatch plan. **Design + stubs only; no component bodies.**
- Supersedes (operationally): ADR-013 (prescribed Noah-MP subset, Opt-A) and ADR-014 (prescribed-land state extension) for the **land path only**. The prescribed path remains a fallback and the ocean/water path is unchanged.
- Scope authority: v0.2.0 P0-3 (`V0.2.0-PLAN.md`), the active-option scope-bound (`NOAH-MP-SCOPING.md`), and the HFX root-cause proof (`proofs/v010_validation/hfx_overflux_root_cause.json`).

---

## 0. Why this work exists (the falsifiable target)

The v0.1.0 daytime-land HFX over-flux (+107..+328 W/m², +1.5 K midday T2 warm bias, LH 18×) is
**architectural, not a surface-layer bug** (root-cause proof, verdict
`ROOT_CAUSE_IS_ARCHITECTURAL_NOT_A_SURFACE_LAYER_BUG`). With the MYNN heat-exchange coefficient
matched to WRF Noah-MP `CH` within 5% (CHS 0.0430 vs CH 0.0452), the over-flux is **exactly the
gradient ratio** `(TSK_radiative − thx)/(TAH_canopy − thx) = 19.77/14.15 = 1.40`. The GPU land path
computes `HFX = ρ·cpm·CHS·(θ(TSK_radiative) − thx)` from the **radiative** skin temperature, while WRF
Noah-MP sets `HFX = FSH` where `FSH = ρ·cpm·CH·(TAH − SFCTMP)` from the **canopy-air** temperature TAH
(5.5 K cooler than radiative TSK at midday over dry sparse-veg land).
WRF driver mapping confirmed at `module_sf_noahmpdrv.F`:
- `:1223  HFX = FSH`
- `:1224  GRDFLX = SSOIL`
- `:1205  QFX = ECAN + ESOIL + ETRAN`  / `:1206  LH = FCEV + FGEV + FCTR`
- `:1222  TSK = TRAD` (radiative skin)
- `:1231  ALBEDO = SALB`, `:1243  EMISS = EMISSI`, `:1279-1280  Z0 = ZNT = Z0WRF`

The principal chose the **clean/standalone** fix (Option B in the root-cause proof): port the Noah-MP
**canopy/ground surface-energy balance** so HFX/LH/TSK/GRDFLX/albedo/emiss/ZNT are produced by a
prognostic land model — not by a bulk recompute and not by hourly snapping. The canopy energy balance
**is** the over-flux fix; it is the priority sprint.

This ADR freezes the interfaces so multiple component agents can port in parallel without colliding,
and specifies the dispatch plan, file ownership, oracle gates, and no-regression gates.

---

## 1. Scope (frozen, from NOAH-MP-SCOPING.md + verified against pristine WRF)

LAND-ONLY prognostic Noah-MP. Active-option scope-bound (WRF `iopt_*`, verified in
`module_sf_noahmpdrv.F:108-123`):

| option | value | meaning | consequence for the port |
|---|---|---|---|
| `dveg`     | 4 | "off" table-LAI + calculated FVEG (`= SHDFAC`) | LAI/SAI from `PHENOLOGY` table; **no dynamic vegetation/carbon** |
| `opt_crs`  | 1 | Ball-Berry stomatal resistance | canopy resistance for transpiration |
| `opt_btr`  | 1 | Noah soil-moisture stomatal factor | |
| `opt_run`  | 3 | **Schaake96** surface/subsurface runoff | **no groundwater (SIMGM/SIMTOP) — CUT** |
| `opt_sfc`  | 1 | Monin-Obukhov drag (CH/CM) | **Noah-MP does NOT own CH/CM here — sfclay supplies them** (see §4) |
| `opt_frz`  | 1 | NY06 supercooled liquid | within soil thermo/water |
| `opt_inf`  | 1 | NY06 frozen-soil permeability | within Schaake water |
| `opt_rad`  | (3 → gap = 1−FVEG) | radiation transfer | within canopy energy balance |
| `opt_alb`  | 2 | CLASS snow albedo | |
| `opt_snf`  | 1 | Jordan91 rain/snow partition | within phenology/precip-heat |
| `opt_tbot` | 2 | Noah deep-soil lower BC | within soil thermo |
| `opt_stc`  | 1 | semi-implicit snow/soil temperature | within soil thermo |

**Fixed dimensions (verified in `module_sf_noahmpdrv.F`):**
- `NSOIL = 4` soil layers (`DZS`, `ZSOIL(1..4)`; `:689-691`).
- `NSNOW = 3` snow layers, `ISNOW ∈ {-2,-1,0}` active-layer count (`:628 NSNOW = 3`).
- Single-layer canopy "big-leaf" (TV/TAH/EAH), not multi-layer.

**CUT as dead on the Canary domain (do NOT port):** CARBON / CARBON_CROP (NEE/GPP/NPP, dynamic veg),
GROUNDWATER (`module_sf_noahmp_groundwater.F`), glacier (`module_sf_noahmp_glacier.F`), crop, urban,
lake, irrigation, WRF_HYDRO. The driver guards (`ICE`, `IST`, `CROPTYPE`, `IRRFRA`) are pinned to the
land-soil branch (`ICE=0`, `IST=1`, `CROPTYPE=0`, `IRRFRA=0`).

---

## 2. Frozen data interfaces

Two new frozen pytree dataclasses live in **`src/gpuwrf/contracts/noahmp_state.py`** (new file; does
NOT widen the prognostic `State` carry — see §2.3 rationale). All fields are device-resident JAX arrays,
fp64 at construction, registered pytree node classes mirroring `State`'s `replace`/`bytes`/`tree_flatten`
discipline.

### 2.1 `NoahMPLandState` (prognostic land carry)

Shapes: `surface_2d = (ny, nx)`, `soil = (NSOIL=4, ny, nx)`, `snow = (NSNOW=3, ny, nx)`.
All temperatures K, moisture m³/m³ (volumetric) or kg/m², geopotential-free.

| field | shape | units | WRF name | notes |
|---|---|---|---|---|
| `tslb`     | soil | K | STC(1:4) | soil temperature (semi-implicit prognostic) |
| `smois`    | soil | m³/m³ | SMC(1:4) | total soil moisture |
| `sh2o`     | soil | m³/m³ | SH2O(1:4) | liquid soil moisture (≤ smois) |
| `smcwtd`   | surface | m³/m³ | SMCWTD | deep-layer soil moisture below model bottom (Schaake uses; no groundwater) |
| `isnow`    | surface | int32 | ISNOW | active snow-layer count (−2,−1,0) |
| `tsno`     | snow | K | STC(−2:0) | snow-layer temperature (active layers only) |
| `snice`    | snow | kg/m² | SNICE | snow-layer ice mass |
| `snliq`    | snow | kg/m² | SNLIQ | snow-layer liquid mass |
| `zsnso`    | (NSNOW+NSOIL, ny, nx) | m | ZSNSO | snow+soil layer interface depths (<0) |
| `snowh`    | surface | m | SNOWH | bulk snow depth |
| `sneqv`    | surface | kg/m² | SNEQV | bulk snow water equivalent |
| `sneqvo`   | surface | kg/m² | SNEQVO | prior-step SWE (albedo aging) |
| `tauss`    | surface | – | TAUSS | non-dimensional snow age |
| `albold`   | surface | – | ALBOLD | prior-step snow albedo (BATS/CLASS aging) |
| `tv`       | surface | K | TV | **canopy (vegetation) temperature** |
| `tg`       | surface | K | TG | **ground (under-canopy + bare) temperature** |
| `tah`      | surface | K | TAH | **canopy-air temperature — the HFX-fix variable** |
| `eah`      | surface | Pa | EAH | canopy-air vapor pressure |
| `canliq`   | surface | kg/m² | CANLIQ | intercepted canopy liquid |
| `canice`   | surface | kg/m² | CANICE | intercepted canopy ice |
| `fwet`     | surface | – | FWET | wetted canopy fraction |
| `lai`      | surface | m²/m² | LAI | leaf area index (table-driven, dveg=4) |
| `sai`      | surface | m²/m² | SAI | stem area index (table-driven) |
| `cm`       | surface | – | CM | momentum drag coeff (carried; **supplied by sfclay**, §4) |
| `ch`       | surface | – | CH | heat drag coeff (carried; **supplied by sfclay**, §4) |
| `t_skin`   | surface | K | TSK=TRAD | radiative skin temperature (diagnostic carry, mirrors `State.t_skin`) |
| `qsfc`     | surface | kg/kg | QSFC | surface mixing ratio |
| `znt`      | surface | m | ZNT=Z0WRF | combined roughness (Noah-MP-owned over land) |
| `emiss`    | surface | – | EMISSI | surface emissivity |
| `albedo`   | surface | – | SALB | broadband surface albedo |
| `sfcrunoff`| surface | m (accum) | SFCRUNOFF | accumulated surface runoff |
| `udrunoff` | surface | m (accum) | UDRUNOFF | accumulated subsurface runoff |

Static category fields (`ivgtyp`, `isltyp`, `xland`, `landmask`, `lakemask`, `lu_index`, `tbot`)
live in `NoahMPStatic` (§2.4), **not** in the prognostic carry, so they are not re-written each step.

### 2.2 `NoahMPFluxes` (per-step outputs consumed by the coupler)

| field | shape | units | WRF source | maps to |
|---|---|---|---|---|
| `hfx`     | surface | W/m² | FSH (`drv:1223`) | PBL bottom-BC sensible flux (land) |
| `lh`      | surface | W/m² | FCEV+FGEV+FCTR (`drv:1206`) | latent flux (land) |
| `qfx`     | surface | kg/m²/s | ECAN+ESOIL+ETRAN (`drv:1205`) | PBL bottom-BC moisture flux (land) |
| `grdflx`  | surface | W/m² | SSOIL (`drv:1224`) | ground heat flux diagnostic |
| `tsk`     | surface | K | TRAD (`drv:1222`) | radiative skin temperature |
| `qsfc`    | surface | kg/kg | QSFC | surface mixing ratio |
| `znt`     | surface | m | Z0WRF (`drv:1279`) | roughness length (land) |
| `emiss`   | surface | – | EMISSI (`drv:1243`) | longwave emissivity |
| `albedo`  | surface | – | SALB (`drv:1231`) | shortwave albedo |
| `chs`     | surface | – | CHV/CHB blend | heat exchange coeff (diagnostic; for parity check vs sfclay) |

`NoahMPFluxes` is the **only** object the coupling adapter (§4) reads back into the model state and the
PBL bottom boundary. It is a `NamedTuple` (like `SurfaceFluxes`) so it traces as a flat pytree.

### 2.3 Rationale: separate pytree, NOT a widened `State`

The prognostic dycore `State` carries 3-D atmosphere fields through the jitted timestep scan at acoustic
cadence. Noah-MP land state advances at the **physics (long) timestep** only, is 2-D/soil/snow, and is
never touched by the dycore. Widening `State.__slots__` with 30+ land fields would:
1. break the **checkpoint exact-match** (`checkpoint.py:102-108`, `state_field_order`) for every
   existing v0.1.0 restart fixture;
2. force every dycore/advection/microphysics test that constructs `State.zeros` to carry dead land
   leaves through the acoustic scan;
3. couple land schema churn to the frozen dycore carry.

Therefore land state is a **sibling pytree** (`NoahMPLandState`) threaded alongside `State` by the
operational driver, exactly as `BaseState`/`BoundaryState` are siblings today. The coupler (§4) writes
the small handful of land fields the PBL/diagnostics need (`t_skin`, `qsfc`, `roughness_m`) back into
`State.replace(...)` so the existing PBL/writer path is untouched. **`State.__slots__` is NOT modified.**

### 2.4 `NoahMPStatic` (read-only per-run inputs)

`ivgtyp` (int32 veg category), `isltyp` (int32 soil category), `xland`, `landmask`, `lakemask`,
`lu_index`, `tbot` (deep-soil lower BC temperature, K), `dzs` (soil-layer thicknesses, m, len 4),
`zsoil` (interface depths, len 4), `lat`, `dx_m`, plus the Noah-MP parameter tables
(`NoahMPParameters`, §2.5). Frozen pytree; constructed once per run from `wrfinput`.

### 2.5 `NoahMPParameters`

Per-category lookup tables loaded once from the WRF `MPTABLE.TBL`/`SOILPARM.TBL`/`GENPARM.TBL` equivalents
(vegetation: `RHOL/RHOS/TAUL/TAUS/XL/RGL/RSMIN/HS/Z0MVT/HVT/HVB/SAIM/LAIM/SLA/...`; soil: `BB/SATPSI/SATDK/SATDW/MAXSMC/REFSMC/WLTSMC/QTZ/...`; general: `CSOIL/ZBOT/CZIL/...`). Frozen; indexed by `ivgtyp`/`isltyp`. **This table-loader is a prerequisite sprint (Sprint 0, §6).**

---

## 3. Frozen driver + component signatures

All in **`src/gpuwrf/physics/noahmp/`** (new package). Each function is a **clean boundary** another agent
implements and oracle-tests independently. Stubs raise `NotImplementedError`. Signatures below are FROZEN.

### 3.1 Top-level driver — `noahmp_driver.py`

```python
def noah_mp_step(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    dt: float,
) -> tuple[NoahMPLandState, NoahMPFluxes]:
    """One Noah-MP physics-timestep over all land columns (vectorized, jit-friendly).

    Orchestrates, in WRF NOAHMP_SFLX order (module_sf_noahmplsm.F:900-1079):
      phenology_table -> precip_heat -> energy_canopy -> soil_thermo (called INSIDE
      energy as TSNOSOI) -> snow (water/SNOWWATER) -> water_hydro (Schaake).
    LAND-ONLY: ocean/lake columns are masked out upstream by the coupler (§4) and
    are never passed here. No host transfer; pure functional pytree-in/pytree-out."""
```

`NoahMPForcing` (frozen NamedTuple): `sfctmp` (lowest-level air T, K), `sfcprs`/`psfc` (Pa), `uu`,`vv`
(m/s), `q2`/`qair` (kg/kg), `qc`, `soldn` (downward SW, W/m²), `lwdn` (downward LW, W/m²),
`prcp` partition (`prcpconv`,`prcpnonc`,`prcpsnow`,`prcpgrpl`,`prcphail`), `cosz`, `julian`, `yearlen`,
`zlvl` (reference height m). These come from the atmosphere lowest level + radiation + microphysics +
clock; assembled by the coupler.

### 3.2 Components (the parallel sprint boundaries)

```python
# energy.py  — THE HFX-FIX COMPONENT (priority). Ports NOAHMP_SFLX::ENERGY
# (module_sf_noahmplsm.F:1741-2396): two-stream radiation, VEGE_FLUX (canopy/TV/TAH
# energy balance, opt_crs=1 Ball-Berry CANRES) + BARE_FLUX, FVEG-weighted tile sum,
# emits FSH/FCEV/FGEV/FCTR/SSOIL/TRAD/TV/TG/TAH/EAH/ALBEDO/EMISSI/Z0WRF. Calls
# soil_thermo internally (TSNOSOI/THERMOPROP) for STC update.
def noahmp_energy_canopy(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    rad: NoahMPRadInputs,        # SAV/SAG/parsun/parsha from two-stream (in-module)
    dt: float,
) -> tuple[NoahMPLandState, NoahMPEnergyFluxes]:
    ...

# soil_thermo.py — semi-implicit snow/soil temperature (opt_stc=1, opt_tbot=2).
# Ports THERMOPROP (module_sf_noahmplsm.F:2400-2510) + TSNOSOI (:5258-5371): tridiagonal
# semi-implicit STC solve over the NSNOW+NSOIL column. Pure thermal; no water movement.
def noahmp_soil_thermo(
    stc: jax.Array,              # (NSNOW+NSOIL, ny, nx) snow+soil temperature
    df: jax.Array, hcpct: jax.Array,  # thermal conductivity / heat capacity per layer
    ssoil: jax.Array, tbot: jax.Array,
    zsnso: jax.Array, dzsnso: jax.Array, isnow: jax.Array,
    dt: float,
) -> jax.Array:                  # updated stc
    ...

# snow.py — snow water/compaction/aging (NSNOW=3, opt_alb=2 CLASS, opt_snf=1 Jordan91).
# Ports SNOWWATER (module_sf_noahmplsm.F:6398-6535): snowfall add, compaction, layer
# combine/divide, sublimation, melt routing; updates ISNOW/SNICE/SNLIQ/SNOWH/SNEQV/ZSNSO.
def noahmp_snow(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    qsnow: jax.Array, imelt: jax.Array, qmelt: jax.Array,
    dt: float,
) -> NoahMPLandState:            # snow fields updated
    ...

# water_hydro.py — Schaake96 soil hydrology + runoff (opt_run=3, opt_inf=1, opt_frz=1).
# Ports WATER (module_sf_noahmplsm.F:5954-6261) restricted to the Schaake branch:
# canopy interception, infiltration, SOILWATER (:7234-7556) Richards-like soil-moisture
# tridiagonal, Schaake surface+subsurface runoff, supercooled-liquid frozen-soil.
# NO groundwater (SIMGM/SIMTOP CUT). Updates SMC/SH2O/SMCWTD/SFCRUNOFF/UDRUNOFF/CANLIQ/CANICE.
def noahmp_water_hydro(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    et_fluxes: NoahMPEtFluxes,   # ECAN/ETRAN/EDIR/QSEVA from energy (transpiration sink)
    dt: float,
) -> NoahMPLandState:            # soil/canopy water fields updated
    ...

# phenology.py — table phenology (dveg=4: table LAI/SAI + FVEG=SHDFAC).
# Ports PHENOLOGY (module_sf_noahmplsm.F:1255-1358): interpolate monthly LAI/SAI tables
# by julian day & category, compute ELAI/ESAI (snow-buried adjustment), FVEG, IGS.
def noahmp_phenology_table(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
) -> NoahMPPhenology:            # lai/sai/elai/esai/fveg/igs (no state mutation beyond lai/sai)
    ...
```

`NoahMPRadInputs`, `NoahMPEnergyFluxes`, `NoahMPEtFluxes`, `NoahMPPhenology` are small frozen NamedTuples
defined in `noahmp/types.py` (the shared, frozen-first sprint-0 artifact). Exact field lists are pinned in
the stubs.

---

## 4. Frozen coupling adapter (`src/gpuwrf/physics/noahmp_coupler.py`, new file)

The handshake into the existing surface→PBL→dycore chain. **LAND-ONLY masked; ocean/lake keep the current
prescribed-SST bulk `surface_layer_with_diagnostics` path verbatim.**

Sequence per physics step (NEW module; `operational_mode.surface_adapter` will call it later — NOT edited
here, perf-sidecar owns it):

1. **sfclay first (unchanged).** `surface_layer_with_diagnostics(state)` runs over ALL columns to produce
   `CH/CM` (heat/momentum exchange coeffs), `ustar`, `tau_u/tau_v`, and the **water/lake** HFX/LH/QFX/T2/Q2
   exactly as today. `opt_sfc=1`: sfclay OWNS CH/CM; they are fed INTO Noah-MP (`forcing`/`land_state.cm/ch`).
2. **Noah-MP over land.** `noah_mp_step(land_state, forcing, static, dt)` runs on the land mask
   (`is_land = (xland−1.5) < 0`). It OWNS land `HFX/LH/QFX/TSK/albedo/emiss/ZNT`.
3. **Masked blend (the only place land vs water flux is selected):**
   ```
   hfx   = where(is_land, noahmp.hfx,   sfclay.hfx)
   lh    = where(is_land, noahmp.lh,    sfclay.lh)
   qfx   = where(is_land, noahmp.qfx,   sfclay.qfx)   # via qv_flux
   tsk   = where(is_land, noahmp.tsk,   state.t_skin) # water TSK = prescribed SST path
   znt   = where(is_land, noahmp.znt,   sfclay.znt)
   ```
   Kinematic handles for the PBL are rebuilt from the blended HFX/QFX:
   `theta_flux = hfx/(rho*cpm)`, `qv_flux = qfx/rho`, `fltv = (1+EP1*qx)*theta_flux + EP1*thx*qv_flux`
   (identical formulae to `surface_layer.py:710-715`, just fed the blended flux).
4. **PBL bottom BC.** The blended `SurfaceFluxes` is passed as the `surface=` arg into the MYNN column
   (`mynn_pbl._surface_terms(state, surface=...)`, the FROZEN Gate-1 hand-off, `mynn_pbl.py:167-180`).
   **No change to mynn_pbl.** This is exactly the existing operational hand-off; only the land flux source
   changes.
5. **State write-back.** Coupler returns `(state', land_state')` where `state'` carries blended
   `t_skin/roughness_m` (and `qsfc` if present) via `State.replace`, and `land_state'` is the advanced
   prognostic land carry threaded to the next step.

Invariants (frozen):
- **No in-loop host/device transfer** (GPU-kernel rule). Noah-MP forcing is assembled from device arrays;
  `forcing` is a pytree.
- **Ocean/water path byte-for-byte unchanged**: every water/lake column takes the `sfclay` branch; the
  `where(is_land,...)` selection is the sole land/water switch.
- **CH/CM provenance**: `opt_sfc=1` → sfclay computes them, Noah-MP consumes them. Noah-MP does NOT
  recompute drag coefficients. (This matches the corpus namelist `sf_sfclay_physics=5` MYNN-sfc feeding
  Noah-MP's CH/CM inout.)
- **dycore untouched**; **microphysics untouched**; **radiation supplies SOLDN/LWDN/COSZ** into `forcing`.

---

## 5. Restart-schema addition + version-tag plan

`checkpoint.py` currently pickles only `State` with an exact `state_field_order` match
(`:74-76`, `:102-108`) and `FORMAT_VERSION = 1`. Plan:

1. **Bump `FORMAT_VERSION` 1 → 2.** Version 2 payload adds two optional top-level keys:
   - `noahmp_land_state`: the hostified `NoahMPLandState` field dict + `noahmp_land_field_order`
     (exact-match list, same fail-closed discipline as `state_field_order`).
   - `noahmp_format`: a small dict `{"nsoil": 4, "nsnow": 3, "scope_options": {...iopt map...}}` so a
     restart written with one option set cannot be silently resumed under another.
2. **Backward read:** `_read_payload` accepts `format_version ∈ {1, 2}`. A v1 checkpoint (no land state)
   loads with `noahmp_land_state = None`; the loader then cold-initializes Noah-MP land state from
   `wrfinput` (prescribed→prognostic spin-up handled by the driver). A v2 checkpoint round-trips the
   prognostic land carry exactly (bit-identical land state across save/load).
3. **New API:** `write_checkpoint(..., land_state=None)` and
   `read_checkpoint_with_land_state(path) -> (state, land_state|None, namelist, grid, step)`. The existing
   `read_checkpoint` / `read_checkpoint_with_runtime_state` keep working (return v1-shape; ignore land).
4. **Exact-match gate:** `noahmp_land_field_order != tuple(NoahMPLandState._FIELDS)` → `ValueError`
   (schema drift fails closed, same as State). The restart-continuity proof must show a save→load→step
   land-state round-trip that is **bit-identical** (this is the P0-5 restart gate, extended to land).
5. **Tag plan:** the Noah-MP land carry ships in the **v0.2.0** tag (P0-3). The checkpoint v2 format is
   introduced on the v0.2.0 branch only; v0.1.0 release checkpoints stay format v1 and remain readable.

---

## 6. PARALLEL-SPRINT DISPATCH PLAN

Component sprints, **disjoint file ownership**, per-component oracle gate, and parallel-vs-serial order.

### 6.0 Sprint 0 — FROZEN PREREQUISITE (this ADR + shared types + tables). **BLOCKS all others.**
- **Owner files (this worktree, merge first):**
  `.agent/decisions/ADR-NOAHMP-INTERFACES.md`,
  `src/gpuwrf/contracts/noahmp_state.py` (NoahMPLandState/NoahMPFluxes/NoahMPStatic/NoahMPParameters stubs),
  `src/gpuwrf/physics/noahmp/__init__.py`,
  `src/gpuwrf/physics/noahmp/types.py` (all shared NamedTuples — FROZEN),
  `src/gpuwrf/physics/noahmp/noahmp_driver.py` (driver stub + orchestration order),
  `src/gpuwrf/physics/noahmp/{energy,soil_thermo,snow,water_hydro,phenology}.py` (signature stubs),
  `src/gpuwrf/physics/noahmp_coupler.py` (coupling adapter stub).
- **Plus the table loader is its own thin sprint 0b** (parameter tables from MPTABLE/SOILPARM/GENPARM),
  owner file `src/gpuwrf/physics/noahmp/tables.py` — needed by energy/water/phenology before they can
  oracle-test. Can run in parallel with sprint 1 scaffolding once types are frozen.
- Gate: import + pytree round-trip + `tree_flatten` device check; every stub raises `NotImplementedError`;
  signatures match this ADR exactly.

### 6.1 The component sprints

| sprint | component | owner files (disjoint) | WRF source | oracle gate | parallel? |
|---|---|---|---|---|---|
| **S1 (PRIORITY)** | **canopy/ground energy balance — THE HFX FIX** | `physics/noahmp/energy.py` (+ its two-stream radiation helper) | `module_sf_noahmplsm.F` ENERGY:1741-2396, VEGE_FLUX:3578-4170, BARE_FLUX:4174-4479, CANRES:5141-5223 | **savepoint parity** vs pristine-WRF ENERGY: FSH/FCEV/FGEV/FCTR/SSOIL/TRAD/TV/TG/TAH/EAH/ALBEDO/EMISSI/Z0WRF over a land column dump (harness like `sfclay_mynn_full_parity.py`); land-day **HFX over-flux collapses to ≈0** vs corpus | depends on S2 (calls soil_thermo) + tables; START FIRST, integrate after S2 |
| **S2** | semi-implicit soil/snow thermo | `physics/noahmp/soil_thermo.py` | THERMOPROP:2400-2510, TSNOSOI:5258-5371 | analytic tridiagonal-solve oracle + WRF STC savepoint parity (NSNOW+NSOIL column) | **fully parallel** (pure thermal, no deps beyond types) |
| **S3** | snow water/compaction/albedo aging | `physics/noahmp/snow.py` | SNOWWATER:6398-6535, snow-albedo (opt_alb=2) | WRF snow-column savepoint parity: ISNOW/SNICE/SNLIQ/SNOWH/SNEQV/ZSNSO/ALBOLD; conservation (SWE budget closes) | **fully parallel** |
| **S4** | Schaake96 soil hydrology + runoff | `physics/noahmp/water_hydro.py` | WATER:5954-6261 (Schaake branch), SOILWATER:7234-7556 | WRF SMC/SH2O/SMCWTD/SFCRUNOFF/UDRUNOFF savepoint parity; **water-mass conservation** closes; LH 18×→1× over land | **fully parallel** (consumes ET from S1 only at integration) |
| **S5** | table phenology | `physics/noahmp/phenology.py` | PHENOLOGY:1255-1358 (dveg=4 branch) | table-interpolation oracle (LAI/SAI/ELAI/ESAI/FVEG by julian+category) vs WRF | **fully parallel** (smallest; can fold into S1's worker) |
| **S6** | driver orchestration + coupler integration | `physics/noahmp/noahmp_driver.py`, `physics/noahmp_coupler.py`, `runtime/checkpoint.py` (v2) | NOAHMP_SFLX:450-1079 order, noahmpdrv flux mapping | end-to-end: integrated d02/d03 land-day HFX/LH/QFX/TSK/T2/Q2/PBLH diurnal vs WRF ≥24 h; restart bit-identical | **SERIAL — last**, after S1-S5 merge |

### 6.2 Parallel-vs-serial summary
- **Fully parallel from sprint-0 merge:** S2, S3, S5 (no inter-component data deps; each oracle-tests
  against its own WRF savepoint slice).
- **Start-parallel, integrate-serial:** S1 (energy) and S4 (water) — S1 calls soil_thermo (S2) and consumes
  phenology (S5); S4 consumes S1's ET. Each can be **authored and unit-oracle-tested against WRF savepoints
  in parallel** (savepoints provide the upstream inputs as fixtures, so authoring does not block), but
  **integration** of S1↔S2 and S1↔S4 serializes after the dependency merges.
- **Strictly serial last:** S6 driver+coupler+checkpoint, which wires the merged components and runs the
  end-to-end gates.
- **Priority:** S1 first (the over-flux fix is the whole point); dispatch S2 and S5 alongside it so S1 can
  integrate the moment they land; S3/S4 in the same wave.

### 6.3 Per-component oracle-parity gate (binding)
Each component closes ONLY with a **pristine-WRF savepoint parity** proof object under `proofs/noahmp/`:
- Dump WRF Noah-MP intermediate fields at the component boundary from `/home/enric/src/wrf_pristine`
  (instrumented `module_sf_noahmplsm.F` print or the existing savepoint harness extended to Noah-MP).
- Harness pattern = `proofs/v010_validation/sfclay_mynn_full_parity.py` (load WRF inputs as fixtures, run
  the JAX component, assert field-wise RMSE/bias within the gated tolerance over a real land column set).
- Component PASS requires the oracle table to PASS on the **operational fp64 path**, not a manual upcast.

### 6.4 NO-REGRESSION gates (every merge into v0.2.0 must hold all)
1. **Idealized bit-identical:** all existing idealized dycore/physics fixtures bit-identical (Noah-MP is
   off / land mask empty on idealized cases → zero change).
2. **Ocean/water unchanged:** water/lake HFX/LH/QFX/T2/Q2/UST byte-for-byte vs pre-Noah-MP (the
   `where(is_land,...)` selection must not touch water columns).
3. **Winds no-regression:** U10/V10 vs corpus no worse than v0.1.0 (Coriolis-corrected baseline); the
   surface-layer momentum path (ustar/tau) is unchanged (sfclay still owns it).
4. **dycore untouched:** dycore/advection/acoustic test suite unchanged; `State.__slots__` unchanged.
5. **d02/d03 T2:** integrated d02 (3 km) and d03 (1 km) 24 h T2 RMSE ≤ v0.1.0 gate AND the daytime-land
   HFX over-flux collapses (the success criterion of the whole port). LH/QFX/Q2/PBLH no-regression.
6. **Conservation/restart/precision/transfer audits** re-run (P0-5/P0-7): water-mass + energy closure,
   restart land round-trip bit-identical, no in-timestep host transfer.

---

## 7. Consequences
- **Positive:** the daytime-land HFX/LH over-flux is fixed at the WRF-faithful source (canopy energy
  balance), not by a clamp or hourly snap. Land becomes prognostic (true land-memory for ≥24 h diurnal
  cycle). Ocean path and dycore are provably untouched. Components fan out to ≥4 parallel agents with
  disjoint files.
- **Risk (GPT #7):** Noah-MP is the highest overrun-risk P0 after native init. Mitigations baked in:
  sharply scoped option set (§1, carbon/groundwater/glacier/crop CUT), per-component oracle gates so a
  wrong component is caught at its boundary not at integration, and S6 serialized last so coupling bugs
  surface against a fully-validated component set.
- **Open:** WRF Noah-MP savepoint instrumentation must be added to the pristine tree (sprint-0b deliverable
  for the gate harness); two-stream radiation lives inside S1 (energy) and is the largest sub-component —
  flagged for possible split if S1 overruns.
