# Sprint Contract — M6.x Third-Path Substrate Scout

## Objective

Both prototyped dycore architectures cannot deliver an honest 1h coupled d02 forecast (see `.agent/decisions/blockers/M6-DYCORE-BLOCKER-MEMO.md`). Per user standing order #5 (anti-stuck rule), dispatch a research scout to evaluate Option C: porting a working JAX/Python dycore substrate as the base, instead of continuing to hand-roll the WRF-equivalent operator.

**This is a scout sprint** — pure research + recommendation, no implementation, no architecture commitment. Output: a comparison memo the manager can present to the user alongside Options A (WRF Scratch Hybrid) and B (Full WRF Small-Step Port).

## Non-Goals

- NO code edits anywhere.
- NO sub-sprint dispatch.
- NO ADR creation.
- NO promotion of any architecture.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_thirdscout` on branch `scout/codex/m6x-third-path-substrate`.

Write-only:
- `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/worker-report.md` (deliverable)
- `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/option_comparison.md` (the comparison memo)

Read-only everywhere else.

## Inputs

Required reading (all of these — the blocker memo + existing context):

- **`.agent/decisions/blockers/M6-DYCORE-BLOCKER-MEMO.md`** — the situation + Options A/B/C/D framing
- **`.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/verdict.md`** + worker-report — why neither ADR-023 nor ADR-021 works
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/findings_real.md` — the catastrophic baseline numbers
- `.agent/sprints/2026-05-24-m6x-exit-rule-critic/reviewer-report.md` — critic's framing of the exit + §6 mentions third-path options
- `.agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md` — earlier alt-methods scout (already covered Pace/Dinosaur/NeuralGCM/ICON4Py/SCREAM/MPAS-A)
- `.agent/decisions/ADR-023-conservative-column-solver.md` — what survives a pivot
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md` — gate policy
- `.agent/decisions/source_mining_operator_table.md` — operator term provenance
- `PROJECT_CONSTITUTION.md` — what cannot change (JAX primary, GPU-resident, Canary-3km-then-1km target)
- `MILESTONES.md` § M6 — close gate
- `VALIDATION_STRATEGY.md` — Tier-1/2/3/4 pyramid
- `ADR-001` (backend selection) — JAX is primary

## Acceptance Criteria

### Part 1: Three-option comparison memo

Write `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/option_comparison.md` scoring **Options A, B, and C** from the BLOCKER memo on these dimensions:

| Dim | A: WRF Scratch Hybrid | B: Full WRF Small-Step + Savepoint | C: Substrate Port |
|---|---|---|---|
| Estimated time to 1h coupled honest | ? sprints, ? weeks | ? sprints, ? weeks | ? sprints, ? weeks |
| Confidence that path closes M6 | Low / Medium / High | Low / Medium / High | Low / Medium / High |
| Reuse of M5 physics (Thompson/MYNN/RRTMG) | Full / Adapter / Rewrite | Full / Adapter / Rewrite | Full / Adapter / Rewrite |
| Reuse of S1 sidecars + source-mining | Full / Partial / Drop | Full / Partial / Drop | Full / Partial / Drop |
| Reuse of d02 replay infrastructure | Full / Adapter / Rewrite | Full / Adapter / Rewrite | Full / Adapter / Rewrite |
| Carry expansion required | minimal / moderate / full WRF | full WRF | unknown (substrate-defined) |
| ADR cost (new ADRs + supersessions) | 1-2 | 2-3 | 3-5 |
| Newton outer loop required | No | No | depends |
| External dependency added | none | none | substrate library |
| Tier-1 (analytic oracle) reuse | Full | Full | Likely needs adaptation |
| Tier-4 binding (Gen2 RMSE) feasibility | Same as today | Same as today | Same as today |

Cite specific source file:line ranges in WRF/MPAS/Pace/ICON4Py/Dinosaur for each row. If a dimension is genuinely unknown from current evidence, mark "?(needs probe)" — don't fabricate.

### Part 2: Option C deep-dive

Within `option_comparison.md`, dedicate a section to Option C (the new investigation): **which specific substrate** would you port?

Candidates per prior alt-methods scout:
- **Dinosaur** — JAX, IMEX time-integration, spectral global. `Dinosaur@59a0197:dinosaur/time_integration.py:74-114, 193-405`
- **ICON4Py** — Python/GT4Py, regional, vertically implicit, explicit damping. `ICON4Py@3934f68:model/atmosphere/dycore/solve_nonhydro.py:139-1378`
- **Pace/FV3-JAX** — GT4Py + DaCe, FV3 split-explicit. `Pace@6a46e69:fv3core/pace/fv3core/stencils/dyn_core.py:472-936`
- **NeuralGCM** — Dinosaur-based, hybrid ML. Operational at ECMWF benchmark.
- Other? (e.g., SCREAM ports, hand-roll IMEX from scratch using JAX abstractions)

For each candidate, evaluate:
- License compatibility (LGPL/MIT/Apache/none)
- WRF-compatibility delta (can Canary 3km input feed it? does it use hybrid-eta?)
- JAX-native vs DSL (GT4Py)?
- Maintenance: active? frozen? abandoned?
- Effort to "stand up" enough to ingest Gen2 d02 + produce 1h coupled forecast: optimistic and pessimistic estimates
- One key risk per candidate (e.g., Dinosaur is global spectral, would require regional adaptation)

Score the candidates and **name the strongest one as "C-primary"**.

### Part 3: Recommendation

Final paragraph of `worker-report.md`: one of:
- `RECOMMEND-OPTION-A` (WRF Scratch Hybrid)
- `RECOMMEND-OPTION-B` (Full WRF Small-Step + Savepoint Harness)
- `RECOMMEND-OPTION-C-WITH-<substrate>` (e.g., C-with-ICON4Py-pattern)
- `RECOMMEND-OPTION-D` (Defer M6, redirect to M7)
- `RECOMMEND-MIXED-A+C` or similar (with rationale)

Plus one paragraph of dissent against your recommendation (the steelman of the alternative you DIDN'T pick).

### Part 4: No regression

`pytest --collect-only 2>&1 | tail -3` — verify no test files were touched (scout doesn't run tests since no code changes).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_thirdscout
pytest --collect-only 2>&1 | tail -3 | tee .agent/sprints/2026-05-24-m6x-third-path-substrate-scout/proof_no_touch.txt
```

## Performance Metrics

N/A — scout sprint.

## Proof Object

- `option_comparison.md`
- `worker-report.md` (with recommendation + dissent)
- `proof_no_touch.txt`
- Branch `scout/codex/m6x-third-path-substrate`

Time budget: **3-6 hours**. Pure research; no code execution.

## Risks

- **Confabulation**: every claim cites `file:line` in WRF/MPAS/Pace/ICON4Py/Dinosaur source. If a project isn't locally accessible, mark as "(public commit, not locally verified)" with URL.
- **Recommendation bias**: the dissent paragraph is mandatory to counter pre-formed preferences.
- **Scope creep**: don't propose a 4th option unless it's genuinely different from A/B/C/D. The point is to evaluate, not invent.

## Handoff Requirements

When `option_comparison.md` + `worker-report.md` on disk + committed on branch `scout/codex/m6x-third-path-substrate`: `/exit`. Wrapper sends AGENT REPORT to manager pane.

## Failure modes the manager will reject

- Recommendation without source citations.
- Skipping the dissent paragraph.
- Missing any of A/B/C scoring dimensions.
- Code edits anywhere.
