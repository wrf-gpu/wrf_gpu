# L2 d01 boundary forcing source audit

**Sprint**: `2026-05-27-m7-l2-nest-scout` AC3
**Scope**: how does Gen2 CPU WRF construct L2 d01 lateral forcing today, what schema does it write, and can our GPU port consume it without new code?

## Bottom line

- **Gen2 produces `wrfbdy_d01` once per day** via WRF's `real.exe` after `ungrib → metgrid` from AIFS GRIB data. All 28 days in `/mnt/data/canairy_meteo/runs/wrf_l2/` carry a `wrfbdy_d01` (verified — see `l2_day_inventory.json`).
- **Our GPU port can decode `wrfbdy_d01` already** (`gpuwrf.io.boundary_replay.decode_wrfbdy`, used today as a validation oracle), and it can apply lateral boundaries from the canonical `(N_times, 4 sides, z, max_side)` packed leaf via `apply_lateral_boundaries` (`gpuwrf.coupling.boundary_apply`).
- **What is missing**: a thin builder that wires `decode_wrfbdy(wrfbdy_d01)` → `(N_times, 4, z, max_side)` leaves with `update_cadence_s = 21600.0` for the 6 h AIFS cadence, plus an `OperationalCarry` initialised from `wrfinput_d01` instead of a d02 wrfout. The dycore, halo, RK, acoustic, boundary-apply, physics couplers, and surface adapter are all domain-agnostic and need no change.

So the work is bounded: **one new code path (~1 small worker sprint)**, no dycore changes. The gap is real but well-scoped.

## How Gen2 builds L2 d01 wrfbdy today

Traced from `~/src/canairy_meteo/Gen2/scripts/run_pipeline_l2.sh` and friends. Five steps per day:

1. `prepare_aifs_pure_forcing.py` — pull AIFS forecast/analysis NetCDF for the target day.
2. `run_ungrib_case.py` — convert to GRIB-like intermediate files for WPS.
3. `run_wps_case.py` — geogrid + metgrid → `met_em.d01.*.nc`, `met_em.d02.*.nc` (and unused d03/d04 from the L3 namelist template) at 6 h cadence over 72 h (= 13 files per domain).
4. `run_wrf_l1.py` calls `real.exe` → `wrfinput_d01`, `wrfinput_d02`, `wrfbdy_d01`.
5. `mpirun ... wrf.exe` integrates 72 h, producing `wrfout_d01_*` and `wrfout_d02_*`.

In Gen2's post-processing, the raw `wrfout_d01_*` are then **stripped** into `thin_gridded_d01_v1.nc` (~12 MB) and the originals deleted on 24/28 of the L2 days surveyed in AC1. The wrfbdy and wrfinput are **kept on disk** for all 28 days.

## `wrfbdy_d01` schema (verified by direct header-read on the 20260429 case)

```
Time           = 12               (12 × 6 h = 72 h)
DateStrLen     = 19
bdy_width      = 5                (spec_zone=1, relax_zone=4 in the L2 namelist)
south_north    = 59               west_east       = 93
south_north_stag = 60             west_east_stag  = 94
bottom_top     = 44               bottom_top_stag = 45

Per prognostic var V ∈ {U, V, T, QVAPOR, PH, MU, QCLOUD, QRAIN, QICE, QSNOW,
                        QGRAUP, QNICE, QNRAIN, qke_adv, PC, HT_SHAD}:
  V_BXS, V_BXE    — boundary value at western/eastern edge (Time, bdy_width, [bottom_top|+stag], south_north[_stag])
  V_BYS, V_BYE    — boundary value at southern/northern edge (Time, bdy_width, [bottom_top|+stag], west_east[_stag])
  V_BTXS, V_BTXE  — boundary TENDENCY (per-second) at western/eastern edge
  V_BTYS, V_BTYE  — boundary TENDENCY at southern/northern edge
```

Twenty distinct variables in the L2 wrfbdy_d01 — matches the L2 namelist's Thompson MP + MYNN PBL + Noah-MP physics suite.

WRF's lateral forcing rule between time slots is `value(t) = base(time_index) + (t - time_at_index) × tendency(time_index)`. Linear in t. Equivalent to linear interpolation between successive `base` snapshots at the AIFS 6 h cadence.

## GPU side: what exists, what doesn't

### Existing — decoder and side-pack reader (✓)

- `gpuwrf.io.boundary_replay.decode_wrfbdy(path, variables=…, time_index=…)` — `src/gpuwrf/io/boundary_replay.py:168`. Returns a dict keyed by variable, with per-side `{boundary, tendency, units, shape}`. Decoded shape preserves WRF's `(bdy_width, z?, side_index)` layout per side.
- `gpuwrf.io.boundary_replay.wrfbdy_path_for_run(run, domain="d01")` — `boundary_replay.py:213`.
- `gpuwrf.io.boundary_replay.compare_boundary_tendency_to_wrfbdy(...)` — `boundary_replay.py:261`. Validation oracle, not a forcing producer.

### Existing — boundary application (✓)

- `gpuwrf.coupling.boundary_apply.apply_lateral_boundaries(state, lead_seconds, dt_s, config)` — `src/gpuwrf/coupling/boundary_apply.py:31`. Reads from `state.u_bdy`/`v_bdy`/`theta_bdy`/`qv_bdy`/`ph_bdy`/`mu_bdy` with shape `(N_times, 4_sides, z, max_side)`, interpolates linearly at `update_cadence_s`, applies the WRF specified+relaxation pattern. Defaults: `spec_bdy_width=5, spec_zone=1, relax_zone=4, update_cadence_s=3600.0`.
- The L2 namelist exactly matches `spec_bdy_width=5, spec_zone=1, relax_zone=4`. The only diff is cadence: AIFS-driven L2 d01 wrfbdy is 6 h (21600 s), not 1 h (3600 s) as the L3-style d02 replay uses.

### Existing — a *single-step* wrfbdy_d01 → state.bdy packer (½ ✓)

- `gpuwrf.validation.tier3_coupled._pack_wrfbdy_outer_leaf(decoded, var, z_len, max_side, cadence_s)` — `src/gpuwrf/validation/tier3_coupled.py:445`. Already packs *two* time samples (`base[0]` and `base[0] + cadence_s × tendency[0]`) into the `(2, 4, z, max_side)` shape `apply_lateral_boundaries` expects.
- Used in `wrfbdy_boundary_oracle_probe` (`tier3_coupled.py:459`) as a validation oracle.
- **Gap**: it only emits two time samples (n=2 dummy 1-second window for tendency validation). For a 72 h forecast we need either all 12 time samples (or 13 with the implicit endpoint via `base + cadence_s × tendency`), with `update_cadence_s=21600.0`.

### Missing — `build_d01_replay_case` (✗)

There is no `gpuwrf.integration.d02_replay.build_d01_replay_case` analogue. `build_replay_case` is hard-wired to:

- `Gen2Run.grid(domain)` (which requires at least one wrfout to header-read attrs — `gen2_accessor.py:361`),
- `load_history_boundary_leaves(run, grid, domain="d02")` (which requires ≥2 hourly wrfouts — `d02_replay.py:274-277`).

Both presume the parent domain has a written wrfout history. For an AIFS-driven d01 backfill where we want the GPU to *be the forecaster* (not replay CPU WRF), neither precondition holds.

## What the new code path needs to do (one ~half-day sprint)

```python
# Pseudocode for a new src/gpuwrf/integration/d01_replay.py
def build_d01_replay_case_from_wrfbdy(run_dir, *, domain="d01"):
    run = Gen2Run(run_dir)
    # 1. Grid (cannot use run.grid(domain), need to read from wrfinput instead)
    grid = grid_spec_from_wrfinput(run.wrfinput_file(domain))   # <-- new helper
    # 2. Metrics (already wrfinput-driven — works as-is)
    metrics = load_wrfinput_metrics(run.wrfinput_file(domain))
    # 3. IC from wrfinput (already exists for d02 — generalize to d01)
    state = load_state_from_wrfinput(run.wrfinput_file(domain), grid)
    # 4. Boundary leaves from wrfbdy_d01 (the new bit)
    decoded = decode_wrfbdy(run.path / f"wrfbdy_{domain}")
    boundary_leaves = pack_wrfbdy_all_times(decoded, grid)       # <-- new helper
    state = state.replace(**boundary_leaves)
    # 5. Land prescribed state (already exists)
    land = load_prescribed_land_state(run, domain=domain, time=0)
    state = state.replace(t_skin=land.t_skin, ...)
    # 6. Namelist with AIFS cadence
    boundary_cfg = BoundaryConfig(spec_bdy_width=5, spec_zone=1, relax_zone=4,
                                  update_cadence_s=21600.0)      # <-- 6 h, not 1 h
    return ReplayCase(state=state, grid=grid, metrics=metrics, boundary_config=boundary_cfg, ...)
```

Required new helpers:

- `grid_spec_from_wrfinput(path)` — read header attrs (`DX/DY/MAP_PROJ/CEN_LAT/CEN_LON/TRUELAT1/2/STAND_LON`), `ZNU/ZNW`, `e_we/e_sn/e_vert` from the wrfinput file directly. Currently `Gen2Run.grid(domain)` reads the same attrs but from `history_files[0]`; refactor (~25 lines).
- `load_state_from_wrfinput(wrfinput_path, grid)` — copy the existing `build_replay_case` body, but `_load(run, domain, var, time_index=0)` is replaced with `read_var_from_wrfinput(path, var)`. Same fields: `U/V/W/T+P0_THETA_OFFSET_K/QVAPOR/PB+P/PHB+PH/MUB+MU/QCLOUD/QRAIN/QICE/QSNOW/QGRAUP/QNICE/QNRAIN/QKE`. Wrfinput contains all of these at `Time=0`.
- `pack_wrfbdy_all_times(decoded, grid)` — extend `_pack_wrfbdy_outer_leaf` to emit `N=time` snapshots. For 12 wrfbdy times → 13 packed samples by using `base[i] + cadence_s × tendency[i]` between successive bases; verify against `base[i+1]` for sanity. Output the six leaves `u_bdy/v_bdy/theta_bdy/qv_bdy/ph_bdy/mu_bdy` with shape `(13, 4, z, max_side)`.

The dycore, the WRF small-step scratch carry, the physics adapters, the halo, the surface adapter, the RRTMG/Thompson/MYNN couplers — none need touching for d01-vs-d02. The only ripple effect is in callers that hard-code `domain="d02"` (`scripts/m7_daily_pipeline.py`, etc.); they need a domain knob.

## Schema and field-name compatibility table

| Variable in wrfbdy_d01 | GPU State field | Layout transform needed |
|---|---|---|
| `U_B*S/E` (Time, 5, 44, 60) and (Time, 5, 44, 94) | `u_bdy` (N, 4, 44, max(94,60)) | Pack W/E onto axis-1=0/1; S/N onto axis-1=2/3; pad to max_side |
| `V_B*S/E` (Time, 5, 44, 59) and (Time, 5, 44, 94) | `v_bdy` | same |
| `T_B*S/E` (Time, 5, 44, …) | `theta_bdy` | add P0_THETA_OFFSET_K=300.0 (mirrors d02 replay path) |
| `QVAPOR_B*S/E` | `qv_bdy` | clip ≥ 0 in apply_lateral_boundaries (already done) |
| `PH_B*S/E` (Time, 5, 45, …) | `ph_bdy` | add `PHB` from wrfinput (geopotential base; mirrors d02 replay's `add_phb`) |
| `MU_B*S/E` (Time, 5, 1, …) | `mu_bdy` | add `MUB` from wrfinput (mass base) |
| `QCLOUD/QRAIN/QICE/QSNOW/QGRAUP/QNICE/QNRAIN_B*S/E` | not currently in `state.*_bdy` | **Decision needed** — see Q1 |

### Q1 — moisture-species boundary forcing at d01

The current `state` has only `qv_bdy`. The L2 wrfbdy carries hydrometeor boundary values too (`QCLOUD_B*S/E`, etc.). At d01 9 km, this matters for downstream cloud advection from AIFS through the lateral edges. Two options:

- **Simpler**: extend `State` with `qc_bdy/qr_bdy/qi_bdy/qs_bdy/qg_bdy/qni_bdy/qnr_bdy` and corresponding apply calls. ~30 lines added to `state.py` + boundary_apply.py.
- **Cheaper**: zero hydrometeor inflow at the boundary; rely on within-domain microphysics to spin up clouds. Acceptable for short leads (24 h) but degrades T+12..72 h cloud verification.

Recommended: ship "cheaper" for the first L2 backfill iteration, log the gap, follow up with the extension when verification shows it matters.

## Risk summary for the d01 ingestion path

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| State.{u,v,theta,qv,ph,mu}_bdy shape conflict when nx≠ny (max_side padding) | low | low | already exercised by `_pack_wrfbdy_outer_leaf`; check d01 max_side=94 |
| `update_cadence_s=21600.0` propagates to `apply_lateral_boundaries` correctly | low | medium | unit-test against synthetic linear forcing; lock the cadence in the namelist |
| Top-lid / Rayleigh damping / w-damping interactions over the 9 km parent | medium | medium | enable `w_damping=1`, `damp_opt=3`, `zdamp=5000`, `dampcoef=0.2` (matches WRF L2 namelist), run a 6 h spin-up test before claiming 72 h |
| FP32 storage in `*_bdy` leaves vs wrfbdy's FP32 — fine, but verify `base + cadence×tendency` does not overflow | low | low | guarded by `_pack_wrfbdy_outer_leaf`'s float64 internal compute |
| Top-of-domain BC: wrfbdy does not contain a top boundary (rigid lid in WRF), our `apply_lateral_boundaries` does not need one | low | low | confirm M5 dycore uses top_lid=True; check `OperationalNamelist.top_lid` |
| Hydrometeor inflow zeroed at d01 boundary (Q1 above) | high | low–medium | accept for v0 backfill; document; followup sprint |

## Recommendation

For "publish + backfills today": **`BACKFILL_NEEDS_NEW_CODE` on the d01 9 km parent.** Specifically: 1 sprint adding `gpuwrf.integration.d01_replay.build_d01_replay_case_from_wrfbdy` plus the three small helpers above (`grid_spec_from_wrfinput`, `load_state_from_wrfinput`, `pack_wrfbdy_all_times`). Estimated implementation: ~1 worker sprint (half-day or less). No dycore/physics changes.

Once d01 ingestion is shipped, the L2 backfill flow is: AIFS → wps/real (CPU, kept) → GPU d01 (new) → GPU d02 with hourly d01 wrfout as boundary (existing L3 path, repointed to L2 directories). See `nest_backfill_design.md` for the full pipeline.
