# Sprint Contract — PCR vs Thomas Solver Scout (codex, M6-perf-design input)

## Objective

The B-direct savepoint ladder validates the serial Thomas forward-backward recurrence (M6B2 PASS). Operational mode (§14.5.1) allows alternative solvers (PCR / batched-Thomas) if profiler + Tier-4 envelope justifies. **M6-perf-design needs this evidence to pick the operational solver.**

This research-only scout sprint produces a comparison memo: Thomas vs Parallel Cyclic Reduction (PCR) vs batched-Thomas for the WRF vertical-implicit acoustic system on RTX 5090, covering correctness profile, GPU occupancy at d02 dimensions (159×66×44 ≈ 10500 columns), expected speedup, and Tier-4-impact estimate.

**Pure research** — no implementation, no commitment, no operator changes. Output is the comparison memo that ADR-026 will cite.

## Non-Goals

- NO code edits anywhere.
- NO operator implementation.
- NO sub-sprint dispatch.
- NO commitment to a solver — that's M6-perf-design's call.
- NO modification of operational `wrf.exe` or any state.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_pcrscout` on branch `scout/codex/m6-perf-pcr-vs-thomas-scout`.

Write-only:
- `.agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/solver_comparison.md` (deliverable)
- `.agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/worker-report.md`
- `.agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/proof_*.txt`

Read-only everywhere else.

## Inputs

1. This sprint contract
2. `PROJECT_PLAN.md §14.5 + §14.5.1 + §14.5.2` (operational-mode invariants)
3. `.agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/worker-report.md` (the Thomas implementation that's validated)
4. `src/gpuwrf/dynamics/tridiag_solve.py` (the validated Thomas reference)
5. `.agent/sprints/2026-05-25-m6-perf-design/sprint-contract.md` (the sprint this memo feeds)
6. `.agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/env_audit_memo.md` Part 3 (d02 dimensions)
7. Web research targets (cite URLs):
   - Hockney & Jesshope cyclic reduction (classical reference)
   - NVIDIA cuSPARSE / cuSOLVER tridiagonal docs
   - JAX `lax.scan` GPU performance characteristics on Blackwell
   - PCR + Thomas hybrid (used in ICON, MPAS, etc.)
   - Pace/GT4Py and ICON4Py vertical solve implementations
8. `data/fixtures/gen2_baseline/rmse_summary.csv`

## Acceptance Criteria

### Part 1 — Per-algorithm characterization (MANDATORY)

| Algorithm | Asymptotic FLOPs per column (n=44) | Memory traffic per column | Parallel depth | GPU occupancy estimate (10500 columns) | Numerical conditioning vs Thomas |
|---|---|---|---|---|---|
| **Serial Thomas (Gaussian)** | ~8n = 352 | ~6n loads + 2n stores | n = 44 | High (one warp per column with 10500 columns / 32 threads/warp ≈ 328 SMs target; RTX 5090 has 170 SMs) | Reference; well-conditioned |
| **Parallel Cyclic Reduction (PCR)** | ~n log n = 234 | ~3n log n loads | log₂ n = 6 | Excellent (parallelizes within a column) | Slightly worse for ill-conditioned tridiag; usually fine for acoustic |
| **Batched Thomas (PCR + Thomas hybrid)** | tunable | tunable | tunable | Tunable | Combines |
| **cuSPARSE `gtsv`** | external | external | external | external | Reference for benchmark only |

Cite source for each FLOP/memory estimate.

### Part 2 — Suitability for WRF vertical-implicit acoustic (MANDATORY)

For each algorithm, evaluate:
- Does WRF/MPAS/ICON-style vertical implicit acoustic require any specific solver property (positivity, monotonicity, stiff stability) that one algorithm preserves better than another?
- What is the expected Tier-4 envelope impact of switching from Thomas to PCR? Cite ICON-EXCLAIM or Pace results if public.
- Is there an existing JAX `lax.scan` implementation pattern for PCR or batched-Thomas that would be `@jit`-friendly and avoid `device_get`?

### Part 3 — Expected speedup at d02 + d01 scales (MANDATORY)

- d02 (159×66×44 = 10,500 columns): which algorithm wins, by how much (estimate)?
- d01 (50×50×44 = 2,500 columns): which algorithm wins?
- 1km target (≈ 480×200×60 = 5.76M columns): which algorithm wins?
- For each, cite the bottleneck (memory bandwidth, occupancy, kernel launch, etc.)

### Part 4 — Implementation cost (MANDATORY)

| Algorithm | Estimated LOC to implement in JAX | Testing surface (new edge cases) | Risk |
|---|---|---|---|
| Thomas via `lax.scan` (current) | 0 (already in tree) | — | None |
| PCR via `lax.scan` | ? | ? | ? |
| Batched-Thomas via `vmap` + scan | ? | ? | ? |

### Part 5 — Recommendation

ONE of:
- `KEEP-THOMAS` — current solver wins on this hardware/domain; document the speedup ceiling
- `SWITCH-TO-PCR` — provide implementation outline; estimate speedup; flag Tier-4 risk
- `BATCHED-THOMAS` — provide implementation outline
- `BAKEOFF` — propose a small implementation sprint that benchmarks all 3 with Tier-4 checks

Plus one paragraph dissent against your recommendation.

### Part 6 — No regression

`pytest --collect-only 2>&1 | tail -3` — confirm no test changes.

## Validation Commands

None — research scout.

## Performance Metrics

N/A — research only.

## Proof Object

- `solver_comparison.md` (2000–3500 words, with URL citations)
- `worker-report.md` with recommendation + dissent
- Branch `scout/codex/m6-perf-pcr-vs-thomas-scout`

Time budget: **2–4 hours**. Public research + back-of-envelope analysis only.

## Risks

- Confabulation: every FLOP/memory/speedup estimate cites a source or is marked "(estimate, not measured)".
- Spec-gaming: recommendation must respect the validation-mode/operational-mode separation; PCR is operational-mode-only unless WRF itself uses it (it doesn't).
- The scout is research — do NOT promise speedups without measured evidence.

## Handoff Requirements

When `solver_comparison.md` + `worker-report.md` committed on branch `scout/codex/m6-perf-pcr-vs-thomas-scout`: `/exit`. Manager folds into M6-perf-design sprint contract before that sprint activates.
