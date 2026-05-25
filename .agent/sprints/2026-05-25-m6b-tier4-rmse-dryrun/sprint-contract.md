# Sprint Contract — M6b Tier-4 RMSE Dry-Run on 20260429 (Opus Tester)

## Objective

While codex workers hunt the V3 blockers in parallel, **verify the Tier-4 RMSE comparator end-to-end** on a known-passing IC (20260429). The 1h Canary V3 has only verified bounds compliance + spatial divergence; we still need to demonstrate the actual M6 acceptance gate: per-field RMSE for U10/V10/T2 ≤ 3K / 7.5 m/s / 7.5 m/s (5× Gen2 noise floor) vs Gen2 wrfout.

If the RMSE comparator pipeline doesn't even work yet on a passing IC, then no matter how V3 localization resolves we're still blocked. This is a dry-run sprint to prove the comparator works, computes correct RMSE, and produces a publishable proof object.

## Non-Goals

- NO modification to operational_mode.py / dynamics/core/ / validation_wrappers.
- NO change to bound thresholds.
- NO new physics or new validation tier.
- NO production claim on the 20260429 RMSE — this is a **dry-run** of the comparator infrastructure.
- NO remote push.

## File Ownership

Worktree **already created** at `/tmp/wrf_gpu2_tier4` on branch `tester/opus/m6b-tier4-rmse-dryrun`.
Your FIRST command: `cd /tmp/wrf_gpu2_tier4`.

Write-only:
- `scripts/m6b_tier4_rmse_dryrun.py` (NEW) — drives the dry-run
- `tests/test_m6b_tier4_rmse_dryrun.py` (NEW) — smoke test for the comparator
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/` — proof JSONs + tester-report.md

Read-only:
- `src/gpuwrf/validation/tier4_probtest.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `data/fixtures/gen2_baseline/rmse_summary.csv`

## Inputs

1. This sprint contract.
2. Gen2 wrfout at `/mnt/data/canairy_meteo/runs/wrf_l3/20260429_18z_l3_24h_20260524T204451Z/wrfout_d02_*`.
3. `data/fixtures/gen2_baseline/rmse_summary.csv` — noise floor anchors.
4. The 13 `scripts/diagnostic_*.py` helpers (especially `diagnostic_field_rmse_timeline.py` and `diagnostic_gen2_rmse_baseline.py`).

## Acceptance Criteria

### Stage 1 — Run operational 1h on 20260429

Run the operational forecast for 1h on `20260429_18z_l3_24h_20260524T204451Z`. Confirm:
- All bounds checks pass (this IC was known-passing in M6b retry).
- Output produces a wrfout-equivalent state at t=1h (or in-memory equivalent).

Write `proof_1h_run.json` with timing and finiteness check.

### Stage 2 — Compute Tier-4 RMSE

For each of T2, U10, V10:
- Read operational output at t=1h
- Read Gen2 wrfout at `wrfout_d02_2026-04-29_19:00:00`
- Compute spatial-mean RMSE across the d02 domain
- Compute spatial-ratio (max(|local_rmse|) / mean(|local_rmse|)) as a heterogeneity check

Compare each value to the envelope:
- T2 ≤ 3.0 K
- U10 ≤ 7.5 m/s
- V10 ≤ 7.5 m/s
- spatial-ratio ≤ 1.5

Write `proof_tier4_rmse.json` with all 8 numbers + PASS/FAIL per metric.

### Stage 3 — Sanity-check vs Gen2 noise floor

Confirm operational RMSE is at least ≥ Gen2 internal noise floor (from rmse_summary.csv). If operational RMSE is *below* the noise floor, the comparator is broken (suspiciously good). If operational RMSE is in [noise_floor, 5×noise_floor] → PASS healthy. If > 5×noise_floor → operational has real error.

Write `proof_noise_floor_compare.json`.

### Stage 4 — Tester report

Write `tester-report.md` with literal `Decision:` token, one of:
- `Decision: Tier-4 comparator GREEN on 20260429 — ready to gate M6 V3 ICs`
- `Decision: Tier-4 comparator BROKEN — comparator pipeline produces invalid RMSE`
- `Decision: Tier-4 comparator GREEN but operational FAILS RMSE — operational drifts even on passing-bounds IC`

Include: edge cases tested, pipeline gaps, recommendations for hardening the comparator before M6 close gate.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_tier4
export OMP_NUM_THREADS=4
export PYTHONPATH="src"
taskset -c 0-3 python scripts/m6b_tier4_rmse_dryrun.py --run-id 20260429_18z_l3_24h_20260524T204451Z --output .agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/
taskset -c 0-3 python -m pytest tests/test_m6b_tier4_rmse_dryrun.py -v
git add -A && git commit -m "[tier4 RMSE dryrun] $(date -u +%FT%TZ)"
```

## Handoff

Tester-report.md with `Decision:` token, all 3 proof paths, edge cases tested, comparator hardening recommendations.
