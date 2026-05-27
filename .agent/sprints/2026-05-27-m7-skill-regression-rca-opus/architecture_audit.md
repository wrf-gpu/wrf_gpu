# M7 Skill Regression — Architecture Audit (Opus angle, AC1-AC5)

**Sprint**: `2026-05-27-m7-skill-regression-rca-opus`
**Role**: tester (opus 4.7); the worker did not produce a `worker-report.md` before
this report was written, and no `worker/opus/m7-skill-regression-rca-opus` branch
exists; the opus angle was carried by the tester alone.
**Mode**: read-only diagnostic of `src/gpuwrf/**` and supporting predecessor
artifacts. Zero code modified.

## Reference artifacts read

- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/verdict.md` — GPU
  +266% T2 RMSE, +390% U10 RMSE, +243% V10 RMSE vs CPU on 73 AEMET stations.
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json`
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json`
  — operational namelist: `dt_s=10`, `acoustic_substeps=10`, `rk_order=3`,
  `radiation_cadence_steps=999999`, `run_physics=true`, `run_boundary=true`.
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/bounds_check_l2_d02.json`
  — every wrfout finite but with identical `theta_lower_30_max_k=355.36 K`
  across 24 consecutive hourly snapshots (zero drift in the upper-30 mass
  layers).
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- `src/gpuwrf/coupling/boundary_apply.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/coupling/driver.py` (`_surface_diagnostics_for_output`)
- `src/gpuwrf/io/boundary_replay.py`
- `src/gpuwrf/io/land_state.py`
- `src/gpuwrf/physics/surface_layer.py`
- `src/gpuwrf/dynamics/hyperdiffusion.py`

No reference WRF Fortran source was readable from this worktree
(`/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/` is not exposed inside
the audit shell). Where WRF semantics are quoted, they are taken from the
WRF user-guide conventions baked into the GPU code's own comments
(`module_big_step_utilities_em.F:6506-6511`, `solve_em.F:1472-1483`,
`solve_em.F:2409-2717`, `solve_em.F:3065`) and the
`lbc_fcx_gcx`/`spec_bdy_width`/`spec_zone`/`relax_zone` namelist surface.

---

## AC1 — Boundary-forcing path audit

### What the GPU pipeline actually does

1. `gpuwrf.integration.d02_replay.load_history_boundary_leaves`
   (lines 265–333) and `load_nested_parent_boundary_leaves` (lines 443–568)
   build the boundary leaves consumed by `apply_lateral_boundaries`.
2. Per side, only the outermost row/column is sampled from the parent
   wrfout: `d02_replay.py:218-228`
   ```python
   def _field_sides_3d(field: np.ndarray) -> dict[str, np.ndarray]:
       return {
           "W": field[:, :, 0],
           "E": field[:, :, -1],
           "S": field[:, 0, :],
           "N": field[:, -1, :],
       }
   ```
   The packed boundary tensor has shape `[time, side, z, padded_side_len]`.
   There is **no `bdy_width` dimension**; the boundary holds the parent's
   single outermost column / row, not the WRF `spec_bdy_width=5` strip.
3. Verified at runtime by `pipeline_run_20260521.json`:
   `mu_bdy.shape = [9, 4, 1, 160]`, `theta_bdy.shape = [9, 4, 44, 160]`,
   `u_bdy.shape = [9, 4, 44, 160]`. Dimension 2 is `nz`, not `bdy_width`.
4. The runtime applies a 4-row relaxation zone, but the *target* at every
   offset is the same outermost-row value: `boundary_apply.py:61-77`
   builds `forcing = interpolate_boundary_leaf(boundary, ...)` (shape
   `[side, z, side_len]`) and then `_apply_specified` / `_apply_relax` both
   call `_side_values(forcing, side, ...)` which returns that single
   outermost slice for every relax offset (`_apply_relax` lines 99–126
   pull the same `target = _side_values(...)` for offsets 1, 2, 3).
5. Time interpolation between two snapshot anchors is linear at hourly
   cadence (`interpolate_boundary_leaf`, lines 48-58, `update_cadence_s=3600.0`).
   This is mathematically equivalent to WRF's tendency-based update *if*
   the snapshots are the same two anchor times — fine.

### Divergence from WRF semantics

- **Critical**: WRF's `wrfbdy_d02` stores `(bdy_width, z, j, side)` arrays
  (e.g. `bxs[i, k, j, m]` for `m = 1..spec_bdy_width`, default 5). The
  outermost column is `m=0` and the four interior relax rows have their
  own parent-derived target values. The GPU pipeline keeps only `m=0` and
  forces every relax-zone row toward that single outer value. The
  decoded WRF `_BT*` tendencies that the GPU code knows about
  (`boundary_replay.decode_wrfbdy:168-210`, `WRFBDY_BASE_SUFFIXES`,
  `WRFBDY_TENDENCY_SUFFIXES`) are decoded for offline comparison
  (`compare_boundary_tendency_to_wrfbdy:261-306`) but **never consumed**
  by the integration path. `d02_replay._field_sides_3d` is what feeds
  `apply_lateral_boundaries`, and it discards width > 1.
- WRF `lbc_fcx_gcx` weights are reproduced in
  `_wrf_relax_weights:149-159` and look correct in isolation
  (`fcx = 0.1/dt * linear * sponge`, `gcx = fcx/50`,
  `linear = max(0, (spec_zone+relax_zone - loop_1based) / (relax_zone-1))`).
  The weight curve is fine; the *targets it nudges toward* are wrong.

### Expected skill impact

Lateral forcing collapses the parent's interior shear/temperature profile
to a single outer-edge value applied across the relax zone. This is
particularly bad for wind on a 3 km nest where the dominant skill source
*is* the parent-domain large-scale flow: the only correct value is the
outermost cell, the next three relax rows are pulled toward that same
outer value, and the interior must adjust through advection. With the
defect in §AC2 below (theta/mu reset every step) the interior can't
advect the boundary signal inward effectively, so U10/V10 RMSE is more
than 4× CPU.

---

## AC2 — Physics + dycore coupling order audit

### What the GPU pipeline actually does

`operational_mode._physics_boundary_step` (lines 537–594) is the per-step
sequence used by `run_forecast_operational` (the entry point chosen by
`daily_pipeline._default_forecast_fn`, lines 229–232):

1. Save `physical_origin = carry.state`.
2. `_rk_scan_step` → 3-stage RK with WRF-shaped PGF and acoustic
   substepping (lines 407–437). Updates `u, v, w, theta, ph, mu, p`.
3. **Guard branch** (default `disable_guards=False`, locked off by
   `tests/test_m6_guard_disabled_debug.py:393-405`) replaces theta and mu
   with `physical_origin.theta` / `physical_origin.mu` /
   `physical_origin.mu_total` / `physical_origin.mu_perturbation`
   (lines 552–563).
4. `thompson_adapter` (microphysics) — only runs in the guard-on branch
   (line 565 condition `if not bool(namelist.disable_guards)`).
5. `mynn_adapter` (PBL).
6. `surface_adapter` — computes `theta_flux, qv_flux, tau_u, tau_v,
   ustar, rhosfc, fltv` from `state.t_skin` and stores them back into
   `State`, but **does not apply** any of them as a tendency to
   `theta, qv, u, v`. The `dt` argument is explicitly discarded
   (`physics_couplers.py:284-312`, `del dt` on line 287).
7. Conditional `rrtmg_adapter` (radiation).
8. `apply_lateral_boundaries`.
9. NaN-safe re-projection.

### Divergence from WRF semantics

- **Catastrophic**: In WRF `solve_em.F` the prognostic theta evolves
  through advection (`rk_addtend_dry`) plus the PBL/MP/radiation
  source terms in the RHS of the RK update. Here the dycore's theta
  update is *thrown away every timestep* (`next_state.replace(theta=
  physical_origin.theta, …)` on line 553). Mu (dry-air column mass) is
  similarly reset. After the reset, theta is only modified by
  `mynn_adapter` (a PBL diffusion + surface-flux contribution that runs
  on **zero surface fluxes**, see AC3) and `rrtmg_adapter` (which is
  effectively off, AC4). The boundary's prescribed theta is written into
  the outermost row, but with no theta advection that signal cannot
  propagate inward.
- `mynn_adapter` (`physics_couplers.py:250-281`) passes three `zeros`
  arrays as the surface-flux bottom-BC inputs to `MynnPBLColumnState`
  (lines 266–269). MYNN therefore runs with **no surface heat, moisture,
  or momentum fluxes**, no matter what `surface_adapter` computed.
- `surface_adapter` is called *after* `mynn_adapter` (line 568 in
  operational_mode.py runs `mynn_adapter` before `surface_adapter`), so
  even if MYNN read `state.theta_flux` instead of the literal `zeros`,
  it would always be stale by one step. The ordering is the inverse of
  WRF, where surface fluxes feed the same-step PBL closure.
- `thompson_adapter` runs only in the guard-on branch (line 564–566
  read: `if bool(namelist.run_physics): if not bool(namelist.disable_guards):
  next_state = thompson_adapter(...)`). Because production locks
  `disable_guards=False`, microphysics actually does run — but it runs
  *after* the dycore reset, so the only theta entering microphysics is
  the pre-step theta plus the (zero-flux) MYNN update, not the
  RK3-advected theta the kernel was designed for.

### Expected skill impact

Heat advection by winds is absent. Boundary inflow of cold/warm air
cannot propagate inward. Diurnal evolution is missing because there is
no SW heating (AC4) and no upward sensible heat flux from `t_skin`
(this AC + AC3). Both T2 (+5.46 K bias on 24 valid times) and 3-D theta
in the upper layers (identical `theta_max_k = 492.67 K` across 24
consecutive bounds-check files in `bounds_check_l2_d02.json`) are
consistent with theta being effectively frozen by the dycore reset.

The wind fields *do* evolve through RK3 (u/v are not in the reset
list), but the PGF that drives them is recomputed against a never-
updated `p_perturbation` / `mu_perturbation`, so the wind tendencies
are computed from a stale pressure field after the first acoustic
substep. This is consistent with the +390% U10 / +243% V10 RMSE — the
wind dynamics still respond to the lateral forcing but on the wrong
thermodynamic baseline.

---

## AC3 — Surface layer / SST / land state audit

### What the GPU pipeline actually does

- `io.land_state.load_prescribed_land_state` (lines 61–107) reads `TSK,
  SST, SMOIS, SH2O, TSLB, XLAND, IVGTYP, ISLTYP, LU_INDEX, MAVAIL, ZNT`
  from `wrfinput_d02` at index 0 only — the `time` parameter is
  explicitly discarded on line 64 (`del time`).
- `d02_replay.build_replay_case` (lines 571–679) calls
  `load_prescribed_land_state(run, domain=domain, time=0)` once and
  stuffs `land.t_skin, land.soil_moisture[0], land.xland,
  land.lakemask, land.mavail, land.roughness_m` into the initial
  `State` (lines 636–641). These fields are not updated during the
  integration.
- `physics_couplers.surface_adapter` writes back `ustar, theta_flux,
  qv_flux, tau_u, tau_v, rhosfc, fltv` (lines 304–312) but never
  modifies `t_skin, soil_moisture, soil_temperature`, and never feeds
  the fluxes back into the prognostic atmospheric variables.
- `coupling.driver._surface_diagnostics_for_output` (lines 1185–1204)
  recomputes T2/U10/V10 at output time using
  `surface_layer_with_diagnostics`, which interpolates between
  `theta_ground = t_skin / exner_sfc` and the level-1 theta. T2 is
  formally `th2 * (p_sfc/P0)^(Rd/Cp)` and `th2 = theta_ground + dtg *
  psit2/psit` where `dtg = theta0 - theta_ground` (verified
  `physics/surface_layer.py:258-261`).

### Divergence from WRF semantics

- WRF runs a prognostic LSM (Noah, Noah-MP, or RUC) that integrates the
  surface energy balance and updates `TSK` every step. The GPU pipeline
  uses ADR-023 "Option A prescribed Noah-MP", an explicitly
  non-prognostic shim (`invoked_schemes` entry at
  `d02_replay.py:1369-1372`: "Bounded prescribed Noah-MP state,
  non-prognostic Option A").
- Over a 24-hour forecast, the truth surface temperature swings by
  10–20 K diurnally over land. The GPU `t_skin` is the IC value at
  `2026-05-21T18:00 UTC` (range 282.86–306.80 K per the
  `pipeline_run_20260521.json` snapshot) and stays at that range for
  every output hour. T2 therefore inherits the IC's surface
  temperature, modulated only by the MYNN/sfclay stability functions
  whose level-1 theta input is itself stuck (AC2). This is the
  dominant mechanism for the +5.46 K T2 bias.
- The skin/SST drift has a second, smaller effect: the surface bulk
  Richardson number used by sfclay is derived from a constant
  ground potential temperature and a near-frozen level-1 theta. The
  stability regime selection (stable / unstable / damped) at every
  point is essentially frozen at the IC's diurnal phase, so the
  diagnosed 10 m winds inherit a fixed stability bias.

### Expected skill impact

Direct first-order contribution to the T2 RMSE. Secondary contribution
to U10/V10 through the stability-dependent `psix10/psix` ratio.

---

## AC4 — Microphysics + radiation cadence audit

### What the GPU pipeline actually does

- `DailyPipelineConfig.radiation_cadence_steps: int = 999999`
  (`integration/daily_pipeline.py:76`). The 20260521 run wrote
  `radiation_cadence_steps: 999999` into its proof.
- `run_forecast_operational` (lines 615–666) only emits a radiation
  step when `((step + cadence - 1) // cadence) * cadence <= steps`.
  For `dt=10 s`, 24 h ⇒ 8640 steps and cadence 999999 means
  `next_radiation = 999999` is never `<= 8640`, so the radiation
  branch is never taken.
- Unlike `d02_replay.run_replay_scan` (which calls one tail
  radiation step via `final_radiation=True`, lines 1121–1132), the
  operational pipeline never calls `rrtmg_adapter` at all over the
  24-hour forecast.

### Divergence from WRF semantics

WRF defaults to `radt = 30 min` (RRTMG every 30 model minutes). With
radiation off, there is no SW heating during the day, no LW cooling
at the surface, no atmospheric heating/cooling rates, and therefore
no diurnal cycle in the thermodynamic forcing. Combined with AC3
(`t_skin` frozen), the surface energy balance is **doubly broken**:
neither the radiative source term nor the surface temperature
response carries information about local time.

### Expected skill impact

Independent confirmation of the T2 bias. The +5.46 K offset is the
right sign and rough magnitude for a 6 h afternoon segment of the
24-h verification window (19 UTC … 18 UTC next day for Iberia, mid-
day local). Whether the dominant contributor is "skin/SST frozen"
(AC3) or "RRTMG off" (this AC) is partially confounded — both need
to be fixed before the residual can be re-attributed.

---

## AC5 — Wind damping / diffusion audit

### What the GPU pipeline actually does

- `dynamics/hyperdiffusion.py` defines a sixth-order
  `HyperdiffusionConfig(enabled=False, coefficient=0.0,
  monotonic_guard=False)` skeleton with `apply_horizontal_hyperdiffusion`.
- `grep` shows the only consumer is the file itself; nothing in
  `operational_mode.py`, `_physics_boundary_step`,
  `_rk_scan_step`, or `_acoustic_scan` ever invokes it.
- The Rayleigh sponge is wired through `AcousticConfig.rayleigh`
  (`dynamics/acoustic_wrf.py:74, 990`) but `ReplayConfig` defaults
  `rayleigh_coefficient: float = 0.0` (`d02_replay.py:175`) and
  `OperationalNamelist` never exposes a non-zero Rayleigh
  coefficient. MYNN's vertical diffusion is the only diffusion path
  active on `u, v, w, theta, qv`.
- The `smdiv` (small-divergence damping) path is referenced
  (`SmdivConfig, apply_smdiv_pressure` import in
  `dynamics/acoustic_wrf.py:17`) but I did not trace its activation
  to the operational path; it is not the dominant suspect.

### Divergence from WRF semantics

L2 d02 WRF baselines typically run with `diff_6th_opt=2,
diff_6th_factor=0.12` and a Rayleigh damping layer in the top model
levels. Both are absent here. Without them, grid-scale noise can
accumulate in u/v through the RK3 advection, especially near the
boundary inflow region where the relax-zone defect (AC1) injects a
spatial discontinuity at the spec/relax interface.

### Expected skill impact

Secondary. Plausibly contributes 0.5–1 m/s of additional U10/V10 RMSE
through unfiltered 2Δx noise propagated by the advection, but is far
smaller than the AC1+AC2 mechanism.

---

## Cross-cut interactions

- **AC2 ↔ AC1**: even a correct multi-width `bdy_width` strip cannot
  improve interior skill if theta is reset every step — the boundary
  signal can only travel inward via advection.
- **AC2 ↔ AC3**: even a correct prognostic skin/SST cannot improve T2
  if level-1 theta is dycore-reset and the MYNN bottom-BC fluxes are
  hard-coded to zero.
- **AC3 ↔ AC4**: with both skin/SST frozen *and* radiation off, the
  surface energy budget has no degrees of freedom. Either fix alone
  is insufficient.

## AC7 — Cross-check with codex sibling

At the time of this audit:

```
$ ls /tmp/wrf_gpu2_rcaopus/.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/
role-prompts
sprint-contract.md
$ ls /tmp/wrf_gpu2_rcacodex/.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/
role-prompts
sprint-contract.md
$ git log --all --oneline | grep -i rca
e1fc63a [Skill regression RCA dispatch] parallel opus + codex root-cause investigation
```

No codex artifacts (`hour_by_hour_deviation.json`,
`first_hour_diff.json`, `boundary_vs_interior.json`,
`physics_on_off_bracket.json`) exist on disk yet. AC7 is therefore
**deferred** to a follow-up read once the codex sibling reports. The
opus suspects in `top_3_suspects.md` are written to be falsifiable by
the codex bisection — see that file for the testable predictions.
