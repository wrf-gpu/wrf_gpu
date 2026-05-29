# F7H Worker Report — geopotential/pressure restoring coupling

**Status: `F7H_PARTIAL`** (AC1 PASS, AC5 PASS; AC2/AC3 FAIL; AC4 DEFERRED with
analytic fallback)

## Objective
Fix the linear-in-t `w` runaway that detonates the idealized cases (Skamarock
warm bubble, Straka density current) on the WRF-faithful dry dycore. Empirically
confirm/refute the "frozen `ph_work` geopotential carry" hypothesis first, then
settle the buoyancy magnitude against WRF ground truth.

## Files changed
- `src/gpuwrf/runtime/operational_mode.py` — **two WRF-faithful fixes**:
  - `_acoustic_core_state_from_prep`: build the once-per-stage `pg_buoy_w`
    `rw_tend` from the FULL-perturbation `grid%p` (`diagnose_pressure_al_alt` =
    JAX `calc_p_rho_phi`) instead of the `calc_p_rho_wrf(prep)` work-delta
    pressure. Matches WRF `module_em.F:1362`.
  - NEW `_refresh_grid_p_from_finished` + call in `_carry_from_finished_stage`:
    recompute `grid%p` from the finished physical `ph'`/θ at each RK-stage
    boundary, as WRF closes each RK step (`solve_em.F:6180`, `:7542`).
  - import: added `BaseState`.
- `scripts/f7h_ph_carry_trace.py`, `f7h_rwtend_source.py`, `f7h_full_p_compare.py`,
  `f7h_buoyancy_sign.py`, `f7h_wfield_structure.py` — Phase-1 instrumentation.
- `proofs/f7h/*` — all deliverables (below).
- **Reverted** an exploratory theta-coupled-work carry change in
  `src/gpuwrf/dynamics/core/acoustic.py` (broke bare-core m4 + detonated earlier).

## Commands run (all `PYTHONPATH=src taskset -c 0-3 …`, cuda:0, fp64)
- `python -u scripts/f7d_substep_tracer.py --steps N --stride S` (before/after)
- `python -u scripts/f7h_ph_carry_trace.py` → `proofs/f7h/ph_carry_trace.json`
- `python -u scripts/f7h_rwtend_source.py` → `proofs/f7h/rwtend_source.json`
- `python -u scripts/f7h_full_p_compare.py` → `proofs/f7h/full_p_compare.json`
- `python -u scripts/f7h_buoyancy_sign.py`, `f7h_wfield_structure.py`
- `run_warm_bubble_case` + `run_density_current_case` → diagnostics+verdicts+plots
- `pytest tests/test_m4_acoustic.py test_m4_dycore_step.py test_m4_tier2_invariants.py` → 10 passed
- `scripts/f7g_ac1_signed_metric_roundtrip.py` (PASS), `f7g_ac2_pg_buoy_ratio.py` (pre-existing FAIL)

## Proof objects produced (`proofs/f7h/`)
- `ph_carry_trace.json` — per-substep `w/ph_work/t_2ave/p/al/rw_tend` (Phase 1).
- `rwtend_source.json` — decomposition proving `rw_tend` ∝ growing `c1f·mu'`.
- `full_p_compare.json` — full grid%p vs work-delta p (interior balance).
- `skamarock_warm_bubble.json` + `skamarock_bubble_verdict.md` (FAIL) + plots.
- `straka_density_current.json` + `straka_density_current_verdict.md` (FAIL) + plots.
- `ph_coupling_fix.md` — what was broken + WRF file:line + before/after trace.
- `wrf_vs_jax_warmbubble.json` — AC4 deferral + WRF-source analytic fallback.
- `regression_recheck.json` — AC5 no-regression evidence.

## Acceptance gates
- **AC1 (ph/p restoring evolves, not frozen): PASS.** The frozen-131.83 was a
  max-statistic artefact; `ph_work` and (post-fix) the diagnostic `p'` evolve with
  `w`. Restoring `p'` lifted from O(1) → O(100–400) Pa.
- **AC2 (warm bubble): FAIL.** `max|w|@100s` cut ~10× (44.7→4.3 m/s) and the
  thermal now RISES (centroid +173 m by 160 s), but a residual vertical 2Δz mode
  detonates the run ~180 s (< 500 s gate).
- **AC3 (Straka): FAIL.** Same residual mode; non-finite before 900 s.
- **AC4 (WRF ground truth): DEFERRED.** WRF em_quarter_ss `ideal.exe` build
  deferred (canonical Gen2 tree is em_real-only; building would change the pinned
  sha; clean copy+compile is outside the empirical time-box). Delivered the
  WRF-source line audit + analytic discrete-balance fallback that localized the
  fix; the binary remains the definitive arbiter for the RESIDUAL 2Δz mode.
- **AC5 (no regression): PASS.** 10/10 m4 acoustic+dycore+tier-2 invariant tests
  pass; AC1 round-trip PASS; no clamps/caps/sanitizers/xfails added or weakened.

## Root cause (evidence-first, triangulated with parallel GPT bug-hunt)
The linear-`w` runaway is a **pressure-restoring inconsistency**, NOT a frozen ph
carry and NOT the 9.4× buoyancy yardstick:
- `pg_buoy_w` and the carried `state.p_perturbation` used the small-step
  WORK-DELTA pressure (O(1–10 Pa); built from `prep.ph_work`≈0, `prep.mu_work`≈0),
  not the WRF `calc_p_rho_phi` diagnostic `grid%p` (O(1e3–1e4 Pa) once `ph'`
  evolves). The `rdn·Δp'` restoring PGF was therefore ~10–1000× too small and
  could not balance the `−c1f·mu'` column weight, which grows unbounded as `mu'`
  ramps 0→1.5 Pa from the acoustic mass divergence → positive-feedback runaway.

## Unresolved risks / remaining gap
- A residual **vertical 2Δz acoustic mode at the bubble-center column**
  (`w@center(k)` ±9 between adjacent levels) still grows and detonates ~180 s.
  This is a coefficient-level question in the implicit-`w`/`ph` solve
  (`c2a` / `epssm` off-centering / `calc_coef_w` tridiagonal) that the deferred
  WRF em_quarter_ss savepoint comparison is the definitive arbiter for.
- Secondary (reverted) lead: the bare-core ↔ operational theta-coupled-work
  re-coupling drift (~6.5e-2/substep). Activating the direct coupled carry helps
  theta transport but needs a clean bare-core/operational split to avoid the m4
  regression.

## Next decision needed
Manager call: (a) authorize a WRF em_quarter_ss ideal build in a separate tree
(definitive AC4 + seeds M9 per-operator parity) to localize the residual 2Δz
mode, and/or (b) a follow-up sprint to (i) land the theta-coupled-work carry with
a proper bare-core/operational split and (ii) audit the implicit-w `c2a`/`epssm`
coefficients against WRF savepoints. The two committed fixes are correct and
regression-safe and should be kept regardless.

F7H_PARTIAL
