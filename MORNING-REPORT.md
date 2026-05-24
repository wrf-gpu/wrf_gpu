# Morning Report — 2026-05-24 (post-consultation)

**Status:** M6 entered post-blocker execution today. The HYBRID exit-rule fired as designed; the external deep-AI consultation diagnosed the root cause and prescribed **B-direct with savepoint-first discipline**. ADR-023 is now SUPERSEDED-PROVISIONAL; ADR-024 is ACCEPTED; ADR-025 (savepoint harness + B-direct port ladder) is DRAFT, to be finalized at M6B0 close. M6 has been split into M6a (savepoint parity) / M6b (honest 1h) / M6c (Gen2 24h consistency). Sprint M6B0 (WRF savepoint harness) is dispatching.

## Milestone Ledger

| Milestone | Status | Current read |
|---|---|---|
| M0 | Closed | AgentOS/bootstrap complete. |
| M1 | Closed | WRF oracle and fixture foundation complete. |
| M2 | Closed | Backend = JAX/XLA. |
| M3 | Closed | GPU state/grid skeleton. |
| M4 | Closed | Minimal dycore. |
| M5 | Closed | Physics suite (Thompson μP, MYNN PBL, RRTMG LW+SW). |
| **M6a** | **Active** | WRF small-step savepoint parity. M6B0 dispatching. |
| M6b | Pending | Honest 1h Canary d02 (sanitizer-off). Blocked on M6a (M6B6). |
| M6c | Pending | 6h/24h Gen2 statistical consistency. Blocked on M6b. |
| M7 | Prologue done | Stays on CPU WRF until M6b passes. |
| M8 | Pending | Public/forkable release not started. |

## What happened overnight (2026-05-23 → 2026-05-24)

1. **S2.2 fixed the d02 replay hang** (commit `4ee4d31`) — JAX `lax.cond` radiation predicate pathology.
2. **S3-narrow** cleaned stabilizer provenance — 28→20 experiment-backed, 8→37 source-backed.
3. **S4-prep** built Tier-3 convergence infrastructure.
4. **S2.1-redo** ran the first REAL Gen2-anchored 1h d02 baseline (commit `4b97743`):
   - T2 RMSE = **136.885 K** (218× Gen2 24h noise floor 0.628 K)
   - U10 = **106.419 m/s** (73× off), V10 = **102.232 m/s** (64× off)
   - θ_max = 550 K at step 3600 (post-sanitize, sanitizer cap)
   - **17.2 billion** sanitized nonfinite candidates per 1h run
5. **Exit-rule critic** (full-state GPT-5 critique) verdicted `DISPATCH-OPERATOR-BUG-HUNT` with 9 grep targets.
6. **S3-hunt operator bug-hunt** ran 7 single-suspect A/B toggles under sanitizer-bypass. Verdict: **`NO-BUG-LOCALIZED`** — every toggle still first-nonfinite at step 2.
7. **Third-path substrate scout** evaluated Dinosaur/ICON4Py/Pace/NeuralGCM. Verdict: `RECOMMEND-OPTION-B`.
8. **HYBRID exit-rule fired** as designed. Manager wrote `M6-DYCORE-BLOCKER-MEMO.md` with 4 bounded options.
9. **External AI consultation** (response received 2026-05-24): "Finish this as Option B, but make the savepoint harness the product for the next sprint." Skip A-probe. Keep C as fallback. Add E as optional shadow benchmark.
10. **User approved all 8 plan items.** PROJECT_PLAN.md §14, MILESTONES.md M6 split, ADR-023 supersede, ADR-024 accept, ADR-025 draft, post-consultation reflection, M6B0 sprint contract all committed.

## Diagnosis (refined)

Per consultation: not "WRF small-step is unportable" but **"the project lacks the instrumentation needed to distinguish wrong recurrence / wrong staging / missing scratch / wrong source coupling."** WRF compatibility must be validated at the acoustic-substep level, not at 1h.

The correction:
> Stop trying to make a WRF-like dycore stable from the outside. Rebuild the WRF small-step from the inside, under savepoint parity, then optimize.

## Committed sequencing (B-direct)

```
M6B0 ── M6B1 ── M6B2 ── M6B3 ── M6B4 ── M6B5 ── M6B6 ── M6b ── M6c ── M6-perf
savepoint  coef   tridiag  scratch  acoustic  full    coupled  1h    24h   optim
harness   parity  parity   parity   parity   dycore   step    honest Gen2  (post-correctness)
```

**Sprint #1 = M6B0** (in dispatch): WRF `module_small_step_em` savepoint extractor + JAX comparator + deliberate-perturbation negative test + first coefficient parity proof.

## What's NOT scheduled

- **A-probe** (WRF scratch hybrid as first sprint) — skipped per consultation.
- **C — substrate port** — fallback only, after B proves WRF small-step unportable.
- **D — defer M6** — reserved for business continuity.
- **E — shadow GPU-WRF (AceCAST / wrf-gpu-port)** — optional; deferred unless principal authorizes.

## Validation gates (binding)

| Gate | Binds | Bar |
|---|---|---|
| G1 — Savepoint parity | M6B0–B6 | sanitizer-off; per-operator delta; no caps |
| G2 — 10-step replay | end of M6B6 | first-nonfinite null; no caps |
| G3 — 1h d02 | M6b | no sanitizer in production path; T2/U10/V10 RMSE ≤5× Gen2 floor |
| G4 — 6h/24h Gen2 | M6c | AceCAST-style probabilistic envelope |
| G5 — Performance | post-M6c | wall-clock < 28-rank CPU WRF; 0 H2D/D2H in loop |

## Time to M6 close

10–17 sprints. Wider than the prior 5–9 estimate, but each sprint produces hard per-operator parity evidence rather than forensics on billions of nonfinites.

## Out-of-band ask

The consultation recommends an **external human WRF expert review** at M6B0 close — two hours from someone who has touched WRF `module_small_step_em`, MPAS-A dycore, or ICON HEVI numerics. The review target is "are these savepoints sufficient to prove the small-step port?", not "is JAX good?". Flagged to principal for M6B0 close.

## Risk gates active

- M6B0 cannot patch WRF Fortran in ≤2 sprints → escalate; consider AceCAST alternative.
- Comparator finds >15 savepoints diverging at step 2 even with WRF carries added → trigger external WRF-expert review; reconsider C.
- No measurable speedup at M6B5 → reopen performance section before M6b.
- External WRF expert unobtainable → Codex critical-review substitute; flag gap.
