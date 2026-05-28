# Project Reset — Path to Real WRF v4 Skill Parity (DRAFT, pre-critique)

**Status**: DRAFT — pending codex critic + codex blinded plan + merge
**Created**: 2026-05-28
**Approved at top level by principal**: 2026-05-28 (skill gate, timeline, publish freeze all confirmed)

## Why this reset exists

`v0.0.1` shipped 2026-05-28 with **bitwise dycore savepoint parity at 100 coupled steps vs unmodified WRF v4** and a **22.26× apples-to-apples speedup vs 28-rank CPU WRF on the same workstation**. The same release also documented **operational skill regression**: T2 RMSE +161–378 %, U10 RMSE +214–370 %, V10 RMSE +177–353 % vs CPU WRF on a 5-day Canary case. The principal's reading: that is not a usable GPU port, and the artifact should not be published in that state. This document defines what "usable" means, where we stand against it today, and the milestone path from here to there.

## Binding goal (top of every sprint contract)

> A JAX-native GPU port of WRF v4 that delivers Canary L2/L3 forecasts whose **24–72 h RMSE on T2, U10, V10 is not statistically distinguishable from CPU WRF v4** (paired t-test p > 0.05 on a ≥ 15-case ensemble across seasons), while preserving **≥ 10× speedup vs 28-rank CPU WRF** on the same workstation.

This gate binds: artifact not re-released, paper not re-published, no milestone closes until this is proven.

## Position assessment — ~34 % complete

| Block | Weight | Done | Contribution |
|---|---:|---:|---:|
| JAX/XLA + whole-state device residency (foundation) | 5 % | 100 % | 5.0 |
| Dycore @ 100 coupled steps savepoint parity | 10 % | 100 % | 10.0 |
| Physics couplers savepoint-verified vs WRF | 20 % | 30 % | 6.0 |
| Operational composition (surface flux, theta limiter, BC, guards) | 15 % | 30 % | 4.5 |
| Land surface prognostic (Noah-MP) | 15 % | 10 % | 1.5 |
| Multi-IC validation corpus + stat-equivalence methodology | 15 % | 20 % | 3.0 |
| Statistical-equivalence reached (final gate) | 15 % | 10 % | 1.5 |
| Performance preserved under correctness fixes | 5 % | 50 % | 2.5 |
| **Total** | **100 %** | | **~34 %** |

We have built the rails (foundation + dycore). The engine that runs along them (couplers, fluxes, land surface, guards) is roughly two-thirds remaining work.

## Milestone roadmap M8 → M15

| # | Milestone | Definable proof | Weeks | Δ% | Risk |
|---|---|---|---:|---:|---|
| **M8** | Operational-mode savepoint parity audit | `divergence_map.json` listing first bitwise-divergence operator in `_physics_boundary_step`, step, magnitude, side-by-side WRF Fortran trace | 3-4 | +5 | Low |
| **M9** | Surface flux + MYNN bottom-BC parity | `surface_layer.F` outputs reproduce bitwise; T2 RMSE ≤ 5.0 K on pinned 5-day Canary | 2-3 | +15 | Medium (T2 driver) |
| **M10** | Theta limiter replacement (clip → positive-definite) | No diurnal saturation in 24 h trace; T2 RMSE ≤ 3.5 K | 1-2 | +5 | Low |
| **M11** | Prognostic Noah-MP on GPU | Replaces hourly data replay; bitwise match vs WRF Noah-MP at 24 h | 4-6 | +20 | High (largest new code) |
| **M12** | Microphysics admissibility removal + Thompson trust | No NaN over 24 h; T2 RMSE ≤ 2.5 K | 1-2 | +3 | Low |
| **M13** | Multi-IC validation corpus (15–30 Canary L2 + L3) | RMSE table per case; paired diffs vs CPU WRF | 2 | +10 | Low |
| **M14** | Statistical equivalence achieved | Paired t-test p > 0.05 on T2/U10/V10 RMSE vs CPU WRF | 2 | +5 | Medium |
| **M15** | Perf regression-proof + v0.1.0 release | ≥ 10× speedup preserved; arXiv preprint; tag v0.1.0 | 2 | +5 | Low |

Total: **17–23 weeks**, target delivery **2026-09-30 → 2026-10-31**. Δ% sums to +68, putting us at ~100 % at M15 close (with 2 % buffer for surprises).

## "Constantly improve without breaking" — the invariant ladder

Each milestone close MUST satisfy:

- **INV-1**: ADR-027 (D2H-zero in timestep loop) holds.
- **INV-2**: B6 savepoint parity @ 100 coupled steps stays bitwise.
- **INV-3**: From M8 onward, B6 extends to 1 000+ steps; once verified, never regresses.
- **INV-4 (contracting RMSE band)**: T2/U10/V10 RMSE on the **pinned 5-day Canary case** must **decrease or hold equal** vs previous milestone. Regression = sprint rejected.
- **INV-5 (perf floor)**: ≥ 10× speedup vs 28-rank CPU WRF on canonical case. Drop below = profiler-driven re-tune before merge.
- **INV-6 (no test relaxation)**: no test deleted, no tolerance widened, no `xfail` added without explicit ADR + reviewer approval.

Every sprint produces a `proof.json` with INV-1..6 measured values. The merge gate blocks if any invariant trips. This is the structural answer to "constantly improve without breaking" — the invariants are a **one-way ratchet** that contracts the RMSE band sprint-by-sprint while protecting everything already won.

## Sprint sizing + multi-AI verification

- **Sprint length**: 1-2 weeks of frontrunner work. No micro-sprints. Each sprint closes on a numeric target (e.g. "T2 RMSE ≤ 5.0 K") + a proof object + an invariant audit.
- **Frontrunner**: Codex GPT-5.5 xhigh — writes code + proof object.
- **Mandatory second AI (non-negotiable)**: Opus 4.7 tester rebuilds the test independently against the worker's branch; Opus 4.7 reviewer reads the diff and files objections. **No sprint merges with only codex sign-off.**
- **Debugger**: Opus 4.7 debugger spawned only when the worker's proof object fails an invariant.
- **Tiebreaker**: Gemini agy engaged at milestone closes (full-council vote) and when codex+opus deadlock mid-milestone.

## Publish repo + paper handling

- `/home/enric/src/wrf_gpu/` (public GitHub repo) and v0.0.1 paper + tag **stay frozen** until M15.
- No new pushes to `wrf-gpu/wrf_gpu` until v0.1.0 (which closes M15 with statistical equivalence proven).
- Public README gets one sentence: *"v0.0.1 is a foundation preview; the operational-skill closure is the M8-M15 work tracked in the wrf_gpu2 development repo."*
- All work happens in `wrf_gpu2/` until M15.

## Principal-confirmed decisions (2026-05-28)

- **Skill gate (M14)**: paired t-test p > 0.05 on T2/U10/V10 RMSE across 15–30 Canary L2 + L3 cases.
- **Timeline**: Sept–Oct 2026 full scope, no scope cuts on Noah-MP.
- **Publish freeze**: v0.0.1 paper + tag frozen until M15; small README disclaimer pending after final plan adoption.

## What happens after this draft is critiqued

This draft is dispatched in parallel to:
1. **Codex critic** — reads this draft, attacks it. What's missing, what's wrongly ordered, what assumption is unsafe.
2. **Codex blinded planner** — does NOT see this draft; builds an independent plan from repo state + binding goal.

After ~90 min the manager merges both into `PROJECT-RESET-PLAN-FINAL.md`, then updates `README.md`, `AGENTS.md`, `CLAUDE.md`, manager + worker skill files, and project memory. Final plan goes to the principal for sign-off; M8 sprint contract dispatches immediately after sign-off.
