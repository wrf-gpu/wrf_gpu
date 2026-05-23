# Manager's Strategic Plan — M6 Close

**Author**: Manager (Claude Opus 4.7)
**Date**: 2026-05-23
**Status**: DRAFT — for Codex critique. The Codex critic should return its own version with rationale; manager will synthesize.

## Premise: "what is the correct solution + what is the easiest way to it"

After 10 sprints overnight, the M6.x dycore reached a better-understood-but-not-solved state. Three things we now know:

1. The conservative MPAS-recurrence path (ADR-023, currently on main) is *operational* — finite through 600s, R7 oracle green, hydrostatic-rest green, MPAS-slice RMSE 1.69% — but it carries a `_mu_continuity_increment` tanh limiter to keep finite (saturates at ~86 kPa).
2. The ADR-021 WRF-shape carry-expansion prototype (on branch, not merged) has correct architecture (cited WRF source lines) but its "PASS" of warm-bubble was a clamp to exactly 9.0 m/s. Architecture good, stabilization unphysical.
3. The warm-bubble `[5, 10]` amplitude target itself was unsourced. Gate is now operator-sanity per ADR-024 PROPOSED; the **real M6 close gate per `MILESTONES.md` + `VALIDATION_STRATEGY.md` + `ADR-007` is Tier-3 short-run convergence + initial Tier-4 RMSE vs Gen2 backfill**.

The cheapest correct path is therefore **measure the real gate first, fix what it tells us is wrong**, rather than chase a synthetic gate (warm-bubble amplitude) that may be the wrong question.

## Sprint sequence (target: 2-3 weeks, 5 sprints)

### S1 — d02 1h boundary replay + diagnostic sidecars (5-8h, codex)

Re-dispatch the d02 boundary-replay sprint that was halted earlier. The scaffolding is on main (`scripts/m6_d02_boundary_replay_1h.py`, `src/gpuwrf/integration/d02_replay.py`, `tests/test_m6x_d02_boundary_replay.py`). Run on the current unified ADR-023 path (post-wiring-fix, post-gate-redesign). Build the diagnostic sidecars as part of this sprint:

- `scripts/diagnostic_field_rmse_timeline.py` — for each output time, spatial RMSE of every prognostic + diagnostic field vs Gen2 wrfout
- `scripts/diagnostic_spatial_divergence_map.py` — 2-D maps showing WHERE in the domain divergence accumulates (boundary zone? steep terrain? convective area?)
- `scripts/diagnostic_conservation_tracker.py` — total mass, total energy, total KE, total enstrophy as time series
- `scripts/diagnostic_bound_violation_tracer.py` — first (column, step) where each field exits a physical bound

**Measurable goals (S1)**:
- 1h forecast completes without nonfinite (no Tier-4 yet, just finiteness gate)
- Field-by-field RMSE table at t=15min, 30min, 1h vs Gen2 wrfout (anchored to Gen2 backfill on `/mnt/data/canairy_meteo/runs/`)
- Conservation tracker shows total mass drift < 0.1% over 1h
- Bound-violation tracer identifies first violation (if any) with `(field, column_i, column_j, level_k, step)` precision

### S2 — MPAS damping derivation (4-6h, codex, parallel with S1)

The `_mu_continuity_increment` tanh limiter is undocumented stabilization. Replace it with a derived form from MPAS source. The Opus gate-strategy critic cited the legitimate MPAS post-solve Rayleigh form at `mpas_atm_time_integration.F:2184-2193` (the `dss` block). Mine that for the cited damping coefficient form. Also strip the inherited `0.38` and `1.35` magic numbers (the operator-sanity gate already flags these as warnings) by deriving the correct per-column omega-to-w metric from MPAS `mu_d * dz/deta / g`.

**Measurable goals (S2)**:
- `_mu_continuity_increment` removed; replaced with MPAS-line-cited `dss`-style damping
- `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE` constant removed; replaced with derived expression
- `MPAS_OMEGA_TO_W_METRIC` constant removed; replaced with per-column-per-level computation
- All prior tests (R7, slice oracle, c2 horizontal, production-grade, path-unification, wiring fix, operator-sanity) still PASS
- Operator-sanity verdict on warm-bubble harness shifts from `FAIL_PHYSICAL_BOUNDS` to either `PASS_OPERATOR_SANITY` or a strictly smaller bound violation

### S3 — 24h Tier-4 RMSE (6-10h, codex, after S1+S2)

Extend the d02 1h replay to 24h. Add a Gen2 ensemble-spread comparator for the operational rejection criterion per `ADR-007`. Mining inspiration from `.agent/references/cpu-wrf-baseline.md` for the exact Gen2 run to compare against.

**Measurable goals (S3)**:
- 24h forecast completes finite
- Spatial-mean RMSE on `T2`, `U10`, `V10` at 6h, 12h, 24h checkpoints captured
- Comparison to Gen2 ensemble spread / Gen2 model-vs-model day-over-day variance (the operational-noise floor)
- Verdict: M6 close gate PASS if RMSE < 2× Gen2 day-over-day spread on each field at 24h; otherwise diagnostic divergence-map identifies the specific failure mode

### S4 — M6 Tier-3 short-run convergence (4-6h, codex, parallel with S3)

The M6 milestone-plan scout already drafted Tier-3 spec at `2026-05-21-m6-milestone-plan-scout/m6-milestone-plan.md`. Implement the timestep-sensitivity envelope on an idealized case (`em_hill2d_x` or equivalent dry case from WRF) — vary `dt` by factors of 2, measure trajectory deviation; pass if growth is sub-quadratic per `VALIDATION_STRATEGY.md` Tier-3.

**Measurable goals (S4)**:
- Tier-3 envelope reported with sub-quadratic dt-doubling growth
- Independent of S3's Gen2 result — Tier-3 is a per-operator property, not an operational property

### S5 — M6 closeout (manager, 1-2h)

If S3 RMSE PASS and S4 Tier-3 PASS:
- Write `.agent/decisions/MILESTONE-M6-CLOSEOUT.md`
- Promote ADR-023 from PROPOSED → ACCEPTED
- Promote ADR-024 from PROPOSED → ACCEPTED
- Identify operator weaknesses for M7 follow-up (e.g., GPU memory at 1km)

If either fails, the diagnostic sidecars tell us exactly which field, which lead-time, which region — the next sprint is then a targeted fix, not another architecture pivot.

## Inspiration mining (cross-cutting across S1-S4)

For every operator change in this plan, the worker must cite a source line from one of:

- **WRF source** `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/` — especially `module_small_step_em.F:619-1597` (canonical small-step) and `module_em.F` (large-step)
- **MPAS source** `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F` — especially the `dss` Rayleigh block (`:2184-2193`), tridiagonal coefficient builder (`:1589-1656`), forward/back-sub (`:2172-2208`)
- **Pace `fv3core`** (public GitHub `Pace@6a46e69`) — `dyn_core.py` (AcousticDynamics), `del2cubed.py` (hyperdiffusion), `ray_fast.py` (Rayleigh damping), `fillz.py` (negative-tracer adjustment)
- **ICON4Py** (public GitHub `ICON4Py@3934f68`) — `solve_nonhydro.py` (NonHydrostaticConfig + divergence damping), `vertically_implicit_dycore_solver.py` (Thomas pattern)
- **NeuralGCM/Dinosaur** for JAX IMEX style — `dinosaur/time_integration.py` (IMEX abstractions)

No magic numbers in the production path without a cited source.

## Risk register (delta on top of `RISK_REGISTER.md`)

| Risk | Probability | Mitigation |
|---|---|---|
| S1 d02 replay reveals architectural failure mode that requires ADR-021 carry expansion after all | Medium | Diagnostic sidecars from S1 will identify exactly which field/process fails; the ADR-021 prototype is on `worker/gpt/m6x-adr021-wrf-smallstep-prototype` and can be rebased + clamp-stripped if needed |
| S2 MPAS-cited damping replacement fails the operator-sanity gate | Medium | The damping is a smaller-blast-radius change than carry expansion; iterate within S2 |
| S3 24h forecast goes nonfinite before t=24h | Medium | S1's finiteness gate at 1h is the early-warning; if 1h is unstable, halt before S3 dispatch |
| S3 Tier-4 RMSE exceeds 2× Gen2 spread | Medium-high | Diagnostic divergence map identifies whether the failure is in dycore, surface coupling, microphysics, or boundary forcing — drives targeted follow-up |
| GPU OOM at d02 over 24h | Medium | M6-S6 already OOM'd; ADR-023's small carry should help but verify in S1 first |

## What this plan DOESN'T do

- Does not pursue ADR-021 carry expansion as the default path. The current ADR-023 unified state is "operational but instrumented"; pivot only if diagnostics specifically demand it.
- Does not invent new stabilizers. Every stabilization term comes from cited source.
- Does not extend the harness with RK3 big-step coupling (Opus's §9.2 suggestion). The harness becomes a diagnostic; M6 close gate is Tier-4 RMSE, which already exercises the coupled stack.
- Does not block on warm-bubble amplitude. ADR-024 closed that as a gate.

## Questions for the Codex critic

1. Is the **measure-first-fix-after** premise (Path D in the Opus diagnostic's framing) the cheapest correct path, or is there a cheaper variant?
2. Should the diagnostic sidecars in S1 be built **before** the d02 replay run or **alongside** it? (S1 currently has them in one sprint.)
3. Should the MPAS-cited damping derivation in S2 happen **before** S1, **alongside** S1, or **after** S1 (depending on what S1's diagnostics show)?
4. Is the "2× Gen2 day-over-day spread" rejection threshold reasonable, or does the project need a more rigorous Tier-4 acceptance criterion? Cite from `ADR-007` or `VALIDATION_STRATEGY.md`.
5. Are there other diagnostic sidecars worth building beyond the 4 listed? Especially anything that would clarify the dycore failure modes faster.
6. Is the 5-sprint sequence in the right order? Specifically: should Tier-3 (S4) come before Tier-4 (S3)?
