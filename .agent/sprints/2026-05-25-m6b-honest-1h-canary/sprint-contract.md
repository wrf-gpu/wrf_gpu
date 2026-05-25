# Sprint Contract — M6b: Honest 1h Canary d02 Forecast (operational-mode, sanitizer off)

**Status:** Pre-drafted 2026-05-25. **Activates when M6-perf-design closes** (which itself activates when M6B6 closes).

## Objective

After 7 parity rungs (M6B0-R through M6B6) and M6-perf-design picking the operational-mode design (ADR-026), this sprint runs the **first sanitizer-off 1h coupled Canary d02 forecast** using operational mode and validates it against the Gen2 noise-floor envelope.

This is the M6b acceptance gate per `MILESTONES.md M6` revision. Pass = M6b closes.

## Acceptance (binding)

- **No nonfinite** at any step (sanitizer off in production path).
- Theta physically bounded (e.g., 200 K < θ < 400 K throughout domain).
- Wind maxima plausible (|u|, |v| < 100 m/s; |w| < 50 m/s).
- **T2, U10, V10 RMSE vs Gen2** ≤ **5× Gen2 noise floor**: T2 ≤ 3 K, U10 ≤ 7.5 m/s, V10 ≤ 7.5 m/s on Canary d02 24h pair sample.
- Interior error not dominated by single boundary or terrain artifact (spatial diagnostic).
- **Operational-mode used**, not validation mode. Confirmed by absence of savepoint emission, fused kernels, Tier-4-justified precision.
- Wall-clock < 28-rank CPU WRF (operational speedup gate, per §14.5.2).
- Zero H2D/D2H in timestep loop (Nsight Systems trace).

## Non-Goals

- NO bitwise WRF parity in operational mode (M6B0-R through M6B6 already did parity at savepoint level).
- NO sanitizer.
- NO carry expansion beyond what ADR-026 (M6-perf-design output) authorizes.
- NO operator-level changes (locked from M6B0-R/B1/B2/B3/B4/B5/B6).
- NO physics or solver changes outside ADR-026's authorizations.
- NO remote push.
- NO modifications to operational `wrf.exe`.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b_honest` on branch `worker/gpt/m6b-honest-1h-canary`.

Write-only:
- `scripts/m6b_canary_1h_honest.py` (NEW) — orchestrator
- `tests/test_m6b_honest_acceptance.py` (NEW) — Tier-4 envelope assertion
- `tests/test_m6b_operational_no_h2d.py` (NEW or reuse) — Nsight-trace check
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary/` — proofs + worker-report

Read-only operational-mode infrastructure (from M6-perf-design / ADR-026): `src/gpuwrf/runtime/operational_mode.py`.

## Inputs (mandatory)

1. `.agent/sprints/2026-05-25-m6b-honest-1h-canary/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-25-m6-perf-design/worker-report.md` (the operational-mode ADR-026 deliverable)
3. `src/gpuwrf/runtime/operational_mode.py` (the M6-perf-design build)
4. `.agent/decisions/ADR-026-operational-mode-design.md` (or its current status)
5. `data/fixtures/gen2_baseline/rmse_summary.csv` (Gen2 noise floor)
6. Gen2 d02 fixtures: `/mnt/data/canairy_meteo/runs/wrf_l3/` (multiple run-IDs for the 17-pair sample)
7. M6B6 worker-report (last parity rung)

## Acceptance Criteria

### Stage 1 — Operational-mode 1h run

Run `operational_mode.run_forecast_operational(state=canary_d02_ic, namelist=canary_d02_namelist, hours=1)` on at least 3 pinned Gen2 run-IDs from the 17-pair sample.

### Stage 2 — Finiteness + bounds audit

For each run: confirm no NaN/Inf at any step; θ ∈ [200K, 400K]; |u|, |v| ≤ 100 m/s; |w| ≤ 50 m/s. Sanitizer must NOT be active (verify via code inspection).

### Stage 3 — Tier-4 envelope vs Gen2

For each run: RMSE on T2/U10/V10 against Gen2 d02 1h output of same IC. Aggregate across the 3 runs. Assert: T2 mean ≤ 3 K, U10 mean ≤ 7.5 m/s, V10 mean ≤ 7.5 m/s.

### Stage 4 — Spatial-divergence audit

Plot or quantify the spatial distribution of RMSE. Acceptance: interior RMSE within 1.5× of boundary-row RMSE (no single boundary/terrain artifact dominates).

### Stage 5 — Performance + H2D/D2H

Nsight Systems trace of the operational 1h run. Wall-clock metric vs 28-rank CPU WRF reference (from `data/fixtures/gen2_baseline/` if present, or measured). H2D/D2H count inside timestep loop = 0.

### Stage 6 — No regression

Standard pytest suite + M6b new tests pass.

### Stage 7 — Worker report

`worker-report.md` with: per-run RMSE table, finiteness audit, performance comparison, Nsight trace summary, files changed, **M6 close recommendation**: `CLOSE-M6` if all gates pass, `BLOCKER` with named cause otherwise.

## Performance Metrics (binding)

- Wall-clock < 28-rank CPU WRF on Canary 3km d02 1h
- Zero H2D/D2H in timestep loop
- Per-field RMSE inside envelope

## Kill Gates

- Any field nonfinite at any step → STOP (operational-mode failure; route to operational-mode-fix sprint).
- RMSE outside envelope → STOP, route to operator-specific fix sprint.
- Wall-clock loses to CPU WRF → §14.5.2 escalation (one more perf-design sprint; if still slower, architectural re-open).
- Non-zero H2D/D2H → constitutional violation, must be lifted.

## Risks

- The 17-pair Gen2 sample may have outliers; aggregate cautiously, document distribution.
- Operational mode may surface new defects that validation mode's per-operator parity didn't catch (composition vs Tier-4 envelope is a different test).

## Handoff Requirements

When all proofs + worker-report committed: `/exit`. If `CLOSE-M6`: manager dispatches `M6c` (24h Gen2 consistency). If `BLOCKER`: manager dispatches the named-cause fix sprint.
