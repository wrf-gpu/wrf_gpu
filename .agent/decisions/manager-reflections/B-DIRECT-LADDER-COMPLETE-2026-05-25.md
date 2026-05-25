# B-Direct Ladder Complete — Manager Reflection

**Date:** 2026-05-25 (overnight run, ~04:05 local)
**Trigger:** M6B6 close with `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED` at worst delta = 0.0 (bitwise!) on column / patch16 / golden tiers for 10 timesteps with physics + lateral boundary on.

## What was achieved

The B-direct savepoint-first methodology, dispatched on 2026-05-24 night following external consultation, **completed in ~20 hours of wall-time** with 7 successive PASS verdicts:

| Rung | Sprint | Worst delta vs WRF | Duration |
|---|---|---|---|
| 1 | M6B0-R (calc_coef_w) | 0.0 (post-DEFECT-ANALYSIS fix) | 23 min + 14 min defect fix |
| 2 | M6B1 (advance_mu_t) | PASS, per-field ULP | 17 min |
| 3 | M6B2 (Thomas tridiag) | PASS, per-field ULP | 16 min |
| 4 | M6B3 (scratch state) | PASS, per-field ULP | 18 min |
| 5 | M6B4 (acoustic loop composition) | PASS, per-substep | 24 min |
| 6 | M6B5 (full dycore step ×10, physics-off) | **1.11e-16 (FP64 ULP)** | 33 min |
| 7 | **M6B6 (coupled step ×10, physics+boundary on)** | **0.0 (bitwise)** | 44 min |

Plus 4 supporting sprints (hygiene cleanup, JAX top-row fix, reproducer audit, ladder cumulative audit) and 4 research scouts (env audit, plan critic, E-scout, PCR scout, speed-vs-bitwise critic).

## Why this matters

The catastrophic baseline from 2026-05-24 morning was T2 RMSE = 136.9 K (218× Gen2 noise floor) on the unified ADR-023 operator. The bug-hunt returned `NO-BUG-LOCALIZED`. The consultation diagnosed *missing operator-by-operator instrumentation*, not WRF un-portability.

The B-direct savepoint harness was the response. With per-operator parity validated bottom-up against a WRF-shaped oracle, and composition proven at acoustic-substep → acoustic-loop → dycore-step → coupled-step granularity, **the JAX dycore now matches CPU WRF bitwise on the validated harness path**.

## What this does NOT yet prove

- **Operational-mode speed**: the harness uses Python-extracted WRF references plus a wrapper shim, not a relinked WRF emitting from inside the timestep loop. Operational mode (validation-mode-distinct per PROJECT_PLAN §14.5.1) is still un-built. M6-perf-design (next sprint) builds it and proves wall-clock < 28-rank CPU WRF.
- **Full forecast quality**: parity is on 10-step short replays. M6b (1h Canary honest) and M6c (24h Gen2 consistency) are the operational acceptance gates.
- **Bitwise = correct for production**: per Critic Amendment #1, every new field/boundary/dtype is classified validation-only or operational-approved-with-Tier-4-evidence. Many are Undecided pending M6-perf-design's per-operator ablation.

## What's next (sprint chain, all pre-drafted)

1. **M6-perf-design (in dispatch now)**: build operational-mode entry point; run Stage 1.5 PCR-vs-Thomas-vs-hybrid solver bakeoff (per PCR scout's `BAKEOFF` recommendation); produce ADR-026 with per-operator carry/precision/fusion/solver decisions; demonstrate ≥1.2× wall-clock speedup vs 28-rank CPU WRF + zero H2D/D2H in timestep loop + Tier-4 envelope pass.
2. **M6b honest 1h Canary** (pre-drafted): sanitizer-off, operational-mode 1h forecast, Tier-4 envelope assertion across 3+ Gen2 run-IDs.
3. **M6c Gen2 24h consistency** (TBD contract): AceCAST-style probabilistic envelope across the 17-pair sample.
4. **M6 CLOSE** when M6c passes.

Then M7 (Canary operational v0) can migrate from CPU WRF to GPU operational mode.

## Sprint cost

~17 codex sprints + 4 opus tester sprints to bring M6 from BLOCKER memo (2026-05-24 morning) to ladder-complete (2026-05-25 04:05). Operational `wrf.exe` SHA `1ec3815...` unchanged throughout.

## Discipline preserved

- Operational `wrf.exe` immutable (pre/post sha256 verified every build)
- Validation-mode/operational-mode separation explicit (§14.5.1 invariants)
- Critic Amendment #1 classification enforced on every parity sprint from M6B2 onward (M6B1 backfilled during hygiene)
- Patches at RC=0 dry-run (hygiene cleanup repaired prior malformedness)
- SCHEMA_VERSION bumped with backward-compat
- Every claim cites file:line; no spec-gaming
- CPU 4 cores honored; 28 cores reserved for CPU WRF

## Reflection on the methodology

The consultation's framing ("use WRF as the numerical compiler — emit savepoints and rebuild operator-by-operator") was correct in shape but conservative in pace. Codex high-reasoning agents executed each parity rung in 15-45 minutes, including build + extraction + comparison + tests. The pre-draft pattern (writing N+1 sprint contract while sprint N runs) compressed wall-time further.

The discipline that mattered most was **Amendment #1 classification**: every new field defaulted to Validation-only or Undecided. Without that, the scratch families from M6B3 would have shipped to operational mode as carry expansion, capping speed (per the principal directive that wrong-by-design = wrong for this project).

## What I'd flag to the principal

When the principal returns:
1. Ladder is complete; M6 close is now a function of M6-perf-design + M6b + M6c rather than dycore correctness.
2. The catastrophic 136 K RMSE finding from 2026-05-24 morning had a real root cause: the JAX `calc_coef_w` implementation used a MPAS-family meter-space coefficient builder (`build_epssm_column_coefficients(theta, dz_m)`) while WRF uses eta-coordinate + hybrid-mass coefficients. The B-direct harness exposed this surgically at M6B0-R Stage 5; the DEFECT-ANALYSIS sprint applied a localized fix with WRF source citation; parity went from 259× tolerance violation to bitwise zero.
3. Two latent follow-up sprints queued but not urgent: Fortran hook ABI completion (would tighten oracle from Python-reproduction to relinked-WRF-in-timestep) and operational `acoustic_wrf.py:_calc_coef_w` runtime wire-in (the validated `calc_coef_w_wrf_coefficients` helper is currently only invoked by the harness, not by the operational dycore).
4. ADR-023 stays SUPERSEDED-PROVISIONAL; ADR-025 stays PROPOSED until M6-perf-design folds back the operational-mode ablation evidence.
