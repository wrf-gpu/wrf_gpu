# Sprint Contract — M6b Honest 1h Canary V3 (post-reframe, post-bisect)

## Objective

After the shared-core reframe (commit `worker/gpt/m6b-reframe-shared-core` + `worker/gpt/m6b-standalone-vs-comparator-bisect`) the operational mode is:
- ✅ B6 golden validation = 0.0 bitwise
- ✅ Real-IC step-1 controlled-comparator parity = 0.0 bitwise
- ✅ Multi-step standalone parity 2/5/10 = 0.0 bitwise
- ✅ 10s standalone probe PASSES bounded
- ✅ 173 tests pass

This sprint runs the **actual M6b acceptance test**: 1h coupled Canary d02 forecast on 3 pinned Gen2 IDs with operational mode (carry-expanded, shared-core, theta-offset-fixed). Tier-4 RMSE envelope. Per-level bounds. Spatial-divergence audit.

If PASS → **M6 close pending M6c**.

## Non-Goals

- NO modifications to `dynamics/core/` or `validation_wrappers.py` (locked from prior sprints).
- NO modifications to `operational_mode.py` body beyond minor instrumentation if needed.
- NO modifications to operational `wrf.exe`.
- NO sanitizer.
- NO 24h forecast (M6c).
- NO PCR / precision changes.
- NO new physics, new solver, new operator.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b_v3` on branch `worker/gpt/m6b-honest-1h-canary-V3`.

Write-only:
- `scripts/m6b_canary_1h_honest_v3.py` (NEW) — uses the now-working operational mode
- `tests/test_m6b_honest_v3_acceptance.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/` — proofs + worker-report

Read-only:
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/dynamics/core/`
- `src/gpuwrf/dynamics/validation_wrappers.py`

## Inputs (mandatory)

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-standalone-vs-comparator-bisect/worker-report.md` (the PASS verdict + theta-offset fix + caveat about D2H)
3. `.agent/sprints/2026-05-25-m6b-reframe-shared-core/worker-report.md` (the reframe foundation)
4. `data/fixtures/gen2_baseline/rmse_summary.csv` (Gen2 noise floor anchors)
5. **Available Gen2 d02 run-IDs (wrfout-rich; verified)**:
   - `20260429_18z_l3_24h_20260524T204451Z`
   - `20260509_18z_l3_24h_20260511T190519Z`
   - `20260509_18z_l3_24h_20260512T154354Z`
   - `20260521_18z_l3_24h_20260522T072630Z`
   - `20260521_18z_l3_24h_20260522T133443Z`
   - `20260524_18z_l3_24h_20260525T074709Z` (per bisect worker)

## Acceptance Criteria

### Stage 1 — Run operational 1h on 3 pinned Gen2 IDs (MANDATORY)

Pick 3 from the available list (recommend: `20260521_18z_l3_24h_20260522T072630Z`, `20260521_18z_l3_24h_20260522T133443Z`, `20260509_18z_l3_24h_20260511T190519Z`).

For each:
- Run `operational_mode.run_forecast_operational` for 1h
- Sanitizer OFF
- Cores 0-3 + GPU

### Stage 2 — Per-level bounds audit (MANDATORY)

For each run, per step:
- Lower 30 levels: 200K ≤ θ ≤ 400K (tropospheric)
- Upper 14 levels: 250K ≤ θ ≤ 700K (stratospheric — loose)
- Full column: no NaN, no Inf, |u|,|v| ≤ 100 m/s, |w| ≤ 50 m/s

PASS = all bounds hold every step for all 3 runs.

### Stage 3 — Tier-4 RMSE vs Gen2 (MANDATORY)

For each run:
- Compare operational `wrfout_d02` at t=1h against Gen2's `wrfout_d02_YYYY-MM-DD_HH:00:00` at t=1h same IC
- Spatial-mean RMSE on T2, U10, V10
- Aggregate across 3 runs

PASS = T2 mean ≤ 3K, U10/V10 mean ≤ 7.5 m/s.

### Stage 4 — Spatial-divergence audit (MANDATORY)

Per run: interior RMSE within 1.5× boundary-row RMSE.

### Stage 5 — Wall-clock comparison (informational)

Operational JAX wall-clock per run. Compare against Gen2 timestamp-derived denominator (caveat-bound per prior perf-acceptance).

### Stage 6 — No regression (MANDATORY)

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_*.py -v
```

### Stage 7 — Worker report

`worker-report.md`: per-run bounds table + Tier-4 RMSE table + spatial-divergence audit + wall-clock + files changed + **M6 close recommendation** (`CLOSE-M6` if all PASS; `BLOCKER` with named cause otherwise).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6b_v3
taskset -c 0-3 python scripts/m6b_canary_1h_honest_v3.py --runs 3 --hours 1
pytest <full test list> -v
```

## Performance Metrics

- Tier-4 RMSE inside envelope (BINDING)
- Wall-clock informational
- D2H verification deferred to sister sprint

## Kill Gates

- Bounds violation → STOP; bisect (use `m6b_real_ic_operational_compare.py` for multi-step controlled parity)
- Tier-4 RMSE outside envelope → STOP; localize operator
- Operational sha256 changes → STOP

## Risks

- Total wall-clock: 3 × 1h runs on CPU 0-3 may take 30-90 min. Use GPU if available.
- The 10s bounded probe passed; but 1h is 360× longer — late-emerging drift possible.
- Per-step state propagation between operational timesteps was verified at 10 steps but not 360 steps.

## Handoff Requirements

When all PASS + worker-report committed: `/exit`. Manager dispatches **M6c Gen2 24h consistency** OR closes M6 per principal direction.

If BLOCKER: dispatch named-cause fix.

Time budget: **60-150 min**.
