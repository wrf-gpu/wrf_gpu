# M7 Skill Regression — Top 3 Architectural Suspects (Opus angle, AC6)

**Sprint**: `2026-05-27-m7-skill-regression-rca-opus`
**Source evidence**: `architecture_audit.md` (this sprint),
`pipeline_run_20260521.json`, `gpu_vs_cpu_skill_diff.json`,
`bounds_check_l2_d02.json`.

Suspects are ranked by **expected skill-impact magnitude × architectural
confidence**. Each carries a falsifiable prediction the codex sibling's
empirical bisection (AC3 first-hour diff, AC4 boundary-vs-interior split,
AC5 physics-on/off bracket) can confirm or reject.

---

## Suspect #1 — Dycore theta/mu state is reset every timestep

- **Confidence**: HIGH (direct code read; logic is a single guard branch).
- **Files**: `src/gpuwrf/runtime/operational_mode.py:537-594` (the
  `_physics_boundary_step` guard branch lines 548-563).
- **Mechanism**: After `_rk_scan_step` advances `u, v, w, theta, ph, mu,
  p` through RK3 + acoustic substeps, the default guard branch
  (`disable_guards=False`, locked off by
  `tests/test_m6_guard_disabled_debug.py:393-405`) overwrites the
  next-step `theta`, `mu`, `mu_total`, `mu_perturbation` with the
  pre-step values. The dycore's heat- and mass-advection updates are
  discarded every step. Only `mynn_adapter` (PBL diffusion on zero
  surface fluxes — see Suspect #2) and `rrtmg_adapter` (off — see
  Suspect #3) can subsequently modify theta. Mu is only set by
  acoustic substepping, which is then thrown away.
- **Expected skill impact**: dominant for all three failing fields.
  - T2 +5.46 K bias: level-1 theta cannot evolve except through MYNN
    diffusion of a frozen profile.
  - U10/V10: PGF is recomputed from a never-updated pressure split,
    biasing winds throughout the column.
- **Independent corroboration on disk**: `bounds_check_l2_d02.json`
  reports `theta_lower_30_max_k = 355.36 K` and `theta_max_k = 492.67 K`
  identical to 7+ decimal places across all 24 hourly snapshots — a
  signature consistent with theta being state-reset every step rather
  than advected.
- **Codex falsifier**: AC3 first-hour diff should show interior 3-D
  theta nearly identical to the IC (mean(GPU - CPU) ≈ mean(IC - CPU at
  t+1)) and only minor differences in the boundary zone. AC4
  boundary-vs-interior should be interior-dominated for theta but
  near-zero for u, v at lead = 1 h (winds still advect correctly the
  first hour; the bias accumulates).
- **Suggested fix-sprint scope**:
  1. ADR for an alternative stabilization that does *not* throw away
     the prognostic theta/mu update.
  2. Inline the WRF microphysics moisture-saturation/positive-definite
     limiter inside the RK3 update so the original motivation for the
     guard branch (theta/mu blow-up) is no longer needed.
  3. Re-run M7 24 h pipeline and confirm skill drops back to within
     ±20 % of CPU.
- **Dependence on codex**: Codex AC3+AC4 confirm interior-driven theta
  bias dominates; if codex instead finds first-hour theta close to
  CPU and divergence accumulates linearly only via the boundary, this
  suspect demotes from HIGH to MEDIUM (the boundary defect, Suspect
  #4 below, would re-rank).

---

## Suspect #2 — Surface fluxes are computed but never applied to the atmosphere

- **Confidence**: HIGH.
- **Files**:
  - `src/gpuwrf/coupling/physics_couplers.py:250-281` (`mynn_adapter`)
    — line 255 `zeros = jnp.zeros_like(theta_columns)` and lines
    266-269 pass three `zeros` arrays into `MynnPBLColumnState` as the
    surface heat/moisture/momentum-flux bottom-BC inputs.
  - `src/gpuwrf/coupling/physics_couplers.py:284-312` (`surface_adapter`)
    — line 287 `del dt`; the function stores `theta_flux, qv_flux,
    tau_u, tau_v, ustar, rhosfc, fltv` back into `State` but never
    advances `theta, qv, u, v`.
  - `src/gpuwrf/runtime/operational_mode.py:564-570` orders
    `mynn_adapter` **before** `surface_adapter`, so even if MYNN read
    `state.theta_flux` it would be one step stale.
- **Mechanism**: WRF couples sfclay → PBL within a single big step:
  sfclay computes surface fluxes, those fluxes are the bottom BC of
  the PBL closure that mixes the column. Here, MYNN sees zero
  surface fluxes, and the values surface_adapter computes are
  written into `State` only for reuse next step (also zero-consumed)
  and for the wrfout-time `_surface_diagnostics_for_output`
  recomputation.
- **Expected skill impact**: dominant for T2 (no upward sensible-heat
  transport from the warm/cool surface to the atmosphere), strong for
  10 m wind (no momentum stress from the surface into the PBL).
- **Codex falsifier**: AC5 physics-on/off bracket — if `run_physics=
  False` (dycore-only) and the AC5 `radiation_cadence_steps=999999`
  GPU runs produce **nearly identical** T2 RMSE within the first
  hour, that strongly implies the MYNN+surface coupling is not
  contributing useful physics. If the two brackets diverge, Suspect
  #2 is partially exonerated.
- **Suggested fix-sprint scope**:
  1. Reorder: `surface_adapter` runs first; its fluxes feed
     `mynn_adapter`'s bottom-BC inputs (replace the three `zeros`
     arrays with `state.theta_flux`, `state.qv_flux`, and a tau-derived
     momentum-flux pair).
  2. Confirm MYNN PBL kernel expects the same units/sign convention
     as `physics/surface_layer.py` outputs.
  3. Add a savepoint that asserts column-integrated theta increases
     between the surface flux and the post-PBL theta over a
     prescribed 1 h column run with `theta_flux = +200 W/m² / (rho
     cp)`.
- **Dependence on codex**: orthogonal to Suspect #1. Both should
  be fixed together; either alone is insufficient.

---

## Suspect #3 — Surface boundary frozen at IC + RRTMG never called

- **Confidence**: HIGH; mechanism is two independently confirmed defects
  whose combined effect must be untangled by fixing them in order.
- **Files**:
  - `src/gpuwrf/io/land_state.py:61-107` (`load_prescribed_land_state`,
    line 64 `del time`) — `TSK, SST, SMOIS, SH2O, TSLB` loaded once at
    index 0 and never refreshed.
  - `src/gpuwrf/integration/daily_pipeline.py:76`
    (`radiation_cadence_steps: int = 999999`).
  - `src/gpuwrf/runtime/operational_mode.py:615-666`
    (`run_forecast_operational`) — radiation branch never fires for
    cadence > step count.
  - `src/gpuwrf/coupling/physics_couplers.py:315-365` (`rrtmg_adapter`)
    — would have applied a heating-rate tendency to theta if called.
- **Mechanism**: T2/U10/V10 are diagnostics that depend on `t_skin` and
  on level-1 atmospheric state. With `t_skin` frozen at the IC's
  19:00 UTC value across the 24 h forecast, the diagnostic surface
  cannot track diurnal warming. With RRTMG never invoked, there is
  no atmospheric SW heating during the day nor LW cooling at night,
  so even if the dycore advected theta correctly (Suspect #1) the
  atmosphere would not feel the missing radiative source.
- **Expected skill impact**: first-order on T2 RMSE, second-order on
  wind through stability-dependent diagnostic-wind reduction
  (`u10 = u0 * psix10 / psix`, surface_layer.py:256).
- **Codex falsifier**: AC2 spatial deviation map of T2 at hour 12
  should show the deviation peak over land (where the missing
  diurnal cycle hurts most) and small deviations over sea (where SST
  changes little anyway). If the T2 deviation is uniform sea+land,
  the radiation/skin axis demotes and Suspect #1+#2 dominate.
- **Suggested fix-sprint scope**:
  1. Step 1 — Re-enable RRTMG at WRF-default `radt=30 min` cadence:
     change `DailyPipelineConfig.radiation_cadence_steps` default to
     `60 * 30 / dt_s = 180` steps. Re-run pipeline; measure residual
     T2 bias.
  2. Step 2 — Add a minimal `t_skin` update from the surface energy
     balance closure (or move to ADR-023 Option B prognostic
     Noah-MP). Treat as a separate sprint after Suspect #1 is fixed,
     because skin/SST cannot evolve correctly if level-1 theta is
     frozen by the dycore reset.
- **Dependence on codex**: codex AC2 spatial map distinguishes
  land vs sea contribution to T2 RMSE and confirms whether the
  radiation-off and skin-frozen mechanisms are land-dominated.

---

## Honorable mention (Suspect #4) — Lateral boundary width-1 strip

Documented in `architecture_audit.md` §AC1. The relax zone is nudged
toward the *outermost* parent cell at every offset, not the four
distinct interior parent values WRF expects. This is a real
architectural defect, but its expected skill cost is masked by
Suspect #1: with theta unable to advect inward at all, fixing the
boundary's interior profile changes nothing. After Suspect #1 is
fixed, this becomes the next probable contributor and should be
re-ranked against the codex AC4 boundary-vs-interior split.

---

## Recommended fix-sprint sequencing

1. **Sprint A** — Remove the dycore theta/mu reset
   (Suspect #1). Add inline positivity/saturation limiters inside the
   RK3 update so the guard branch is no longer needed. Acceptance:
   24 h pipeline produces theta_lower_30_max_k that varies hour to
   hour by at least 5 K and `bounds_check` still passes (no
   non-finite states).
2. **Sprint B** — Wire surface_adapter outputs into MYNN bottom BC and
   re-order so surface runs before PBL (Suspect #2). Acceptance:
   `surface_layer` returned `theta_flux` matches the column-integrated
   theta delta from the post-PBL state on a 1 h column oracle
   within 5 %.
3. **Sprint C** — Re-enable RRTMG at WRF default cadence
   (`radiation_cadence_steps = 180` for `dt=10`). Acceptance:
   diurnal cycle is visible in `T2` at the AEMET-verified land
   stations.
4. **Sprint D** — Make `t_skin/SST/SMOIS` prognostic or refresh from
   Gen2 every hour (Suspect #3 follow-up).
5. **Sprint E** — Fix the lateral-boundary `bdy_width` strip (Suspect
   #4). Decode the existing wrfbdy `bxs/bys/btxs/btys` arrays directly
   (`io/boundary_replay.decode_wrfbdy` already exists), feed all five
   widths into the relax zone.
6. **Sprint F** — Re-measure AEMET RMSE against CPU and assert within
   ±20 % on T2/U10/V10. Publish.

Each of A–E can be independently validated; A and B should be merged
together because they exercise the same RK step composition and a
single rebaseline of M6B parity is cheaper than two.
