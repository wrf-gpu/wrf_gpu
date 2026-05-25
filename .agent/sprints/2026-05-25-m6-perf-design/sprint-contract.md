# Sprint Contract — M6-perf-design: Operational-mode Design ADR

**Status:** Pre-drafted 2026-05-24; **activates when M6B6 closes** with full coupled-step savepoint parity.

## Objective

Produce the "operational-mode design ADR" that bridges M6B6 (savepoint parity validated) → M6b (1h honest Canary d02). The ADR enumerates per-operator: which carry fields drop, which operators fuse, which fields downcast, which solver variant runs. The ADR is validated by running the same Canary d02 1h that M6B6 passed, measuring Tier-4 RMSE on T2/U10/V10, and beating 28-rank CPU WRF on wall-clock.

This sprint exists because correctness alone is insufficient: per principal directive 2026-05-24, *"solutions that just bring correct results but are massively inefficient for the GPU bomb the project purpose by design."* The savepoint harness validated *correctness*. This sprint validates *the GPU-optimized core that ships*.

## Non-Goals

- NO modifications to the savepoint harness or validation-mode build.
- NO weakening of Tier-4 RMSE envelope.
- NO commitment to bitwise WRF parity in operational mode — that's validation mode's job.
- NO host/device transfers in the operational timestep loop. Constitutional invariant.
- NO new operator semantics not validated in M6B0–M6B6.
- NO sub-sprint dispatch for solver alternatives (PCR, batched-Thomas) until the ADR proposes them.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_perfdesign` on branch `worker/gpt/m6-perf-design`.

Write-only:
- `.agent/decisions/ADR-026-operational-mode-design-DRAFT.md` (NEW)
- `src/gpuwrf/runtime/operational_mode.py` (NEW) — operational-mode entry point; selects fused kernels, dropped carry, downcast fields
- `src/gpuwrf/dynamics/*.py` — **only** to add operational-mode-only entry points; do not change validation-mode semantics
- `scripts/m6_perf_design_canary_1h.py` (NEW) — orchestrates a Canary d02 1h with operational mode and runs Tier-4 RMSE
- `tests/test_m6_operational_mode_parity_envelope.py` (NEW) — asserts operational mode is within Tier-4 envelope on golden slice
- `tests/test_m6_operational_mode_no_h2d.py` (NEW) — Nsight-trace-derived check that the operational timestep loop has zero H2D/D2H
- `.agent/sprints/2026-05-25-m6-perf-design/` — proofs + ADR + worker-report

Read-only everywhere else.

## Inputs

1. `.agent/sprints/2026-05-25-m6-perf-design/sprint-contract.md` (this)
2. `PROJECT_PLAN.md §14.5 + §14.5.1 + §14.5.2`
3. `.agent/decisions/ADR-001-backend-selection.md`
4. `.agent/decisions/ADR-007-precision-policy.md` (per-field precision authorization)
5. `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-PROPOSED.md`
6. All M6B0–M6B6 worker reports and ADR amendments
7. `data/fixtures/gen2_baseline/rmse_summary.csv` (Tier-4 envelope anchors)
8. `feedback_gpu_optimized_core_primacy.md` (memory)

## Acceptance Criteria

### Stage 1 — Per-operator operational-mode plan (MANDATORY)

In `ADR-026-operational-mode-design-DRAFT.md`, enumerate for each operator validated in M6B0–M6B6:

| Operator | Validation carry | Operational carry | Validation precision | Operational precision | Fusion target | Solver |
|---|---|---|---|---|---|---|
| calc_coef_w | … | … (strict subset) | fp64 | … | fused with `solve_w_tridiag` if Tier-4 passes | Thomas / PCR / batched |
| advance_mu_t | … | … | … | … | … | … |
| advance_w internal checkpoints (6) | … | … | … | … | … | … |
| advance_uv | … | … | … | … | … | … |
| calc_p_rho | … | … | … | … | … | … |
| sumflux | … | … | … | … | … | … |

For each row, cite the Tier-4 envelope evidence permitting the choice (downcast, drop, fuse).

### Stage 1.5 — Solver mini-bakeoff (MANDATORY, per PCR scout `BAKEOFF` recommendation)

The PCR-vs-Thomas research scout (`.agent/sprints/2026-05-25-m6-perf-pcr-vs-thomas-scout/worker-report.md`) recommended `BAKEOFF` — do NOT switch from literature alone. Before ADR-026 commits to a solver, run a small measured comparison on the validated M6B2 baseline:

- Current JAX `lax.scan` batched-Thomas (already in tree at `src/gpuwrf/dynamics/tridiag_solve.py`)
- Pure PCR (Hockney & Jesshope) with n=44 columns padded to n=45 + masks
- One fixed PCR+Thomas hybrid split (e.g., PCR for outer log₂ levels then Thomas)
- cuSPARSE / cuSolverDx `gtsv` as benchmark references **only** (not a deployable option)

**Bakeoff acceptance** (all required):
- HLO cost analysis + Nsight Systems trace showing zero H2D/D2H per algorithm
- Kernel/loop count comparison
- `block_until_ready()` wall-clock per algorithm at d02 (10500 columns) on RTX 5090
- Residuals vs validated Thomas on real Canary `calc_coef_w` coefficients (Tier-1 ULP-scale)
- Tier-4 RMSE check on Canary 1h golden slice
- NO operational solver promotion without all 5 above present.

Output: `proof_solver_bakeoff.json` + `proof_solver_bakeoff_nsight.qdrep` + a row in ADR-026's Stage 1 table picking the winner with cited evidence.

### Stage 2 — Operational-mode build (MANDATORY)

Implement `src/gpuwrf/runtime/operational_mode.py`:
- Single entry point `run_forecast_operational(state, namelist, hours)`
- Wraps the dycore in a single `@jit` with `lax.scan` for RK + acoustic loops
- No `device_get`, no host callbacks, no Python diagnostics
- Honors ADR-007 per-field precision authorization
- Uses the carry subset chosen in Stage 1
- Uses the solver picked in Stage 1.5

### Stage 3 — Tier-4 envelope check on golden slice (MANDATORY)

Run operational-mode on the M6B0-R golden slice for 1h:
- Per-field RMSE vs CPU WRF reference (Gen2 d02 of same IC)
- Acceptance: within 5× Gen2 noise floor on T2 (≤3 K), U10 (≤7.5 m/s), V10 (≤7.5 m/s)

If FAIL: name the operator whose operational variant breaks the envelope; route to operator-specific fix.

### Stage 4 — Wall-clock beat 28-rank CPU WRF (MANDATORY GATE)

Same 1h Canary d02:
- 28-rank CPU WRF baseline wall-clock (from `data/fixtures/gen2_baseline/`)
- Operational-mode JAX wall-clock on RTX 5090

Acceptance: **operational wall-clock < 28-rank CPU WRF** by at least 1.2× (modest first-pass speedup). Profiler artifact (Nsight Systems trace) committed.

If FAIL: identify the dominant hotspot, propose one targeted optimization sprint, re-run. Total budget: 2 perf-design sprints. If both fail, fire the §14.5.2 kill gate.

### Stage 5 — No H2D/D2H in timestep loop (MANDATORY)

Nsight Systems trace of the operational 1h run. Filter for `cudaMemcpyHtoD` and `cudaMemcpyDtoH` inside the dycore region. Expected count: **0**.

If non-zero: name the offending call; either lift it out of the loop or escalate.

### Stage 6 — ADR-026 promotion path

ADR-026 moves DRAFT → PROPOSED at sprint close. Acceptance to PROPOSED:
- Stages 1–5 all pass
- Operational-mode entry point exercised on golden slice + Canary 3km 1h
- Reviewer (codex critic) signs off in a follow-up sprint

### Stage 7 — Worker report

`worker-report.md`: per-operator decisions, Tier-4 results table, wall-clock comparison, Nsight trace summary, ADR-026 status, files changed, risks, handoff to M6b (1h honest Canary).

## Performance Metrics (BINDING)

- 1h operational wall-clock < 28-rank CPU WRF × 1/1.2 (i.e., ≥1.2× speedup)
- Zero H2D/D2H in timestep loop
- Tier-4 RMSE inside envelope on T2/U10/V10

## Kill Gates

- Tier-4 envelope fails → fix sprint targeted at the failing operator's operational variant.
- Wall-clock loses to CPU WRF → one more perf-design sprint; if that also loses, fire §14.5.2 architectural-reopening gate.
- H2D/D2H non-zero in timestep loop → constitutional violation, must be lifted out before any other claim.

## Risks

- Aggressive carry pruning may re-introduce nonfinites that validation mode caught. Mitigation: run operational mode under sanitizer-tracing in this sprint only; sanitizer-OFF for the Tier-4 acceptance.
- PCR may produce different RMSE on small-d01 cases vs Thomas. Test only on Canary 3km in this sprint; defer d01 perf to M7.
- ADR-007 may not yet have per-field justifications for the downcasts the worker wants; if so, write a small precision-justification sub-document inside ADR-026.

## Handoff Requirements

When ADR-026 PROPOSED + all proofs + worker-report.md committed: `/exit`. Manager dispatches **M6b (1h honest forecast)** running the operational mode.

## Failure modes the manager will reject

- "Faster than CPU WRF" without Nsight trace.
- "Inside Tier-4 envelope" without per-field RMSE table.
- Skipping H2D/D2H verification.
- Reintroducing host callbacks "temporarily."
- Operational mode that requires validation-mode scratch fields (must be strict subset).
