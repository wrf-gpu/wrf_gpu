# Sprint Contract — M6b Honest 1h Canary RETRY (with carry-expanded operational mode + corrected theta bounds)

## Objective

M6b first attempt returned BLOCKER (theta-bounds at 10s on all 3 Gen2 runs + D2H=53). Step-back identified both as fixable:
- **Carry expansion** completed (commit `worker/gpt/m6b-fix-carry-expansion`): 6 scratch families promoted to Operational-required-with-Tier-4-evidence; 10s probe PASS on all 3 runs.
- **D2H=53 was Nsight profiling artifact**, not real violation (commit `tester/opus/m6b-d2h-grep` + `tester/opus/m6b-d2h-warmed-recapture`).

This sprint RE-RUNS the original M6b honest 1h Canary forecast with the carry-expanded operational mode and **corrected per-level theta bounds**:

- Tropospheric levels (lower 30 eta levels per d02 vertical grid): 200K ≤ θ ≤ 400K
- Stratospheric levels (upper 14 eta levels for d02 nz=44): 250K ≤ θ ≤ 700K (loose physical sanity)
- Hard rules across full column: no NaN, no Inf, |u|/|v| ≤ 100 m/s, |w| ≤ 50 m/s

Acceptance: pass all bounds, Tier-4 RMSE (T2 ≤ 3K, U10/V10 ≤ 7.5 m/s) on 3+ Gen2 run-IDs, zero D2H in warmed-Nsight window (inherited from sister sprint), spatial-divergence audit.

## Non-Goals

- NO code edits to operational_mode.py (carry expansion is from prior sprint; locked here).
- NO validation-mode modifications.
- NO operational `wrf.exe` modifications.
- NO sanitizer in operational path.
- NO multi-week 24h work (that's M6c).
- NO new physics or solver.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b_retry` on branch `worker/gpt/m6b-honest-1h-canary-RETRY`.

Write-only:
- `scripts/m6b_canary_1h_honest_v2.py` (NEW or replace) — uses carry-expanded operational mode + per-level theta bounds
- `tests/test_m6b_honest_v2_acceptance.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/` — proofs + worker-report

## Inputs

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-honest-1h-canary/worker-report.md` (the original BLOCKER for comparison)
3. `.agent/sprints/2026-05-25-m6b-fix-carry-expansion/worker-report.md` (carry expansion + theta-bound caveat)
4. `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/d2h_warmed_memo.md` (the D2H GO/NO-GO)
5. `src/gpuwrf/runtime/operational_mode.py` (post carry-expansion — LOCKED here)
6. `src/gpuwrf/runtime/operational_state.py` (the carry-expanded state — LOCKED)
7. Gen2 d02: 3+ pinned run-IDs from `/mnt/data/canairy_meteo/runs/wrf_l3/`
8. `data/fixtures/gen2_baseline/rmse_summary.csv` (Gen2 noise floor)
9. `.agent/sprints/2026-05-25-m6b-honest-1h-canary/sprint-contract.md` (the original; same Tier-4 envelope numbers)

## Acceptance Criteria

### Stage 1 — Run operational 1h on 3 pinned Gen2 IDs (MANDATORY)

`scripts/m6b_canary_1h_honest_v2.py`:
- Inputs: 3 pinned Gen2 run-IDs (same as carry-fix 10s probe: 20260509, 20260521, 20260523)
- Calls `run_forecast_operational(state, namelist, hours=1)` with carry-expanded operational mode
- Captures wall-clock per run + per-step bounds audit
- Sanitizer-OFF
- Cores 0-3

### Stage 2 — Per-level bounds audit (MANDATORY)

For each timestep and each run:
- Lower 30 levels: 200K ≤ θ ≤ 400K (TROPOSPHERIC)
- Upper 14 levels: 250K ≤ θ ≤ 700K (STRATOSPHERIC — loose)
- Full column: no NaN, no Inf, |u|,|v| ≤ 100 m/s, |w| ≤ 50 m/s
- Document max abs wind per step (informational)

PASS = all bounds hold every step for all 3 runs.

### Stage 3 — Tier-4 RMSE vs Gen2 (MANDATORY)

For each of the 3 runs:
- Compare operational `wrfout` at t=1h against Gen2 `wrfout_d02_2026-MM-DD_HH:00:00` at t=1h same IC
- Spatial-mean RMSE on T2, U10, V10
- Aggregate across 3 runs

PASS = T2 mean ≤ 3K, U10 mean ≤ 7.5 m/s, V10 mean ≤ 7.5 m/s.

### Stage 4 — Spatial-divergence audit (MANDATORY)

For each run: quantify per-grid-point RMSE distribution. Check no single boundary row or terrain feature dominates total RMSE.

PASS = interior RMSE within 1.5× boundary-row RMSE.

### Stage 5 — Wall-clock comparison (informational, since CPU WRF 28-rank still aborts)

Operational JAX wall-clock per run. Compare against Gen2 timestamp-derived denominator (inherited from perf-acceptance — caveat-bound).

### Stage 6 — D2H verification inherited

Inherit from D2H warmed re-capture sister sprint. If sister returned GO, just cite the proof; if NO-GO, escalate.

### Stage 7 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py tests/test_m6b_d2h_warmed_*.py tests/test_m6b_honest_v2_*.py -v
```

### Stage 8 — Worker report

`worker-report.md`: per-run table (bounds + RMSE + wall-clock), Tier-4 envelope status, spatial-divergence audit, D2H inheritance citation, files changed, **M6 close recommendation**: `CLOSE-M6` (PASS all gates) / `BLOCKER` (named cause).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6b_retry
taskset -c 0-3 python scripts/m6b_canary_1h_honest_v2.py --runs 3 --hours 1 2>&1 | tee .agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_1h_runs.txt
pytest <full test list> -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/proof_no_regression.txt
```

## Performance Metrics

- Tier-4 RMSE inside envelope (binding)
- Wall-clock informational (CPU WRF reference is tripwire-only per perf-acceptance caveat)
- D2H inside warmed window = 0 (inherited)

## Kill Gates

- Any per-step bounds violation → STOP, classify (operational carry insufficient → Hypothesis B or further promotion); escalate to a fix sprint.
- Tier-4 RMSE outside envelope → STOP, name the offending field; route to operator-fix sprint.
- Spatial-divergence dominated by single artifact → flag but don't STOP; investigate in M6c.
- Operational sha changes → STOP.

## Risks

- 1h × 3 runs may take 30-90 min wall-clock total even at GPU-accelerated rates.
- Carry expansion may surface secondary instabilities later in the 1h that 10s probe didn't reveal — route to follow-up promotion if it does.
- Tier-4 envelope was set generously (5× Gen2 noise floor); if it's still missed, the operational carry promotion may be necessary but not sufficient.

## Handoff Requirements

When all stages PASS + worker-report committed: `/exit`. Manager dispatches **M6c Gen2 24h consistency** OR closes M6 if M6c is deferred per §14.5.2 risk-gate revision.

If BLOCKER: manager dispatches the named-cause fix sprint per critic recommendation.

## Failure modes the manager will reject

- Bounds violation masked by sanitizer.
- RMSE claim without per-field table.
- Skipping D2H sister-sprint inheritance.
- Modifying operational_mode.py or operational_state.py (locked from prior sprint).
- Adding clamps to mask divergence.
