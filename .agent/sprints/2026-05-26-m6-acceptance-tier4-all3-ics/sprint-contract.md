# Sprint Contract — M6 Acceptance Tier-4 RMSE on 3 ICs

## Objective

Four fix sprints landed today (acoustic-V workaround, vertical-implicit coftz, operational-theta-fix, microphysics-coupling guards). Multi-step CPU parity 2/5/10 = 0.0 bitwise, B6 preserved, 1h theta bounds pass on 20260509 + 20260521.

**Now run the actual M6 acceptance gate**: Tier-4 RMSE for T2/U10/V10 vs Gen2 wrfout at t=1h on all 3 V3 ICs (20260429 + 20260509 + 20260521). If RMSE within envelope (T2 ≤ 3K, U10/V10 ≤ 7.5 m/s — 5× Gen2 noise floor) AND bounds pass AND parity preserved → **M6 close**.

This sprint also closes the loop on M6: writes the M6-CLOSEOUT.md decision memo with all proofs collected.

## Non-Goals

- NO new bug fixes. If RMSE fails, REPORT — don't fix.
- NO modification to source. Only `.agent/sprints/...` + the M6 closeout memo.
- NO remote push.

## File Ownership

Worktree at `/tmp/wrf_gpu2_m6acc` on branch `worker/gpt/m6-acceptance-tier4-all3-ics`.
FIRST: `cd /tmp/wrf_gpu2_m6acc`.

Write-only:
- `scripts/m6_acceptance_tier4_all3.py` (NEW) — driver
- `.agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/` — proofs + worker-report.md
- `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` (NEW — only if all gates pass)

Read-only:
- Everything else.

## Inputs

1. All 4 fix sprint reports from today.
2. `data/fixtures/gen2_baseline/rmse_summary.csv`.
3. Gen2 wrfout for the 3 IDs in `/mnt/data/canairy_meteo/runs/wrf_l3/`.
4. `scripts/m6b_tier4_rmse_dryrun.py` — the prior dryrun gives the comparator pipeline.

## Acceptance Criteria

### Stage 1 — Bounds + parity preserved across all 3 ICs

For each IC in {20260429_18z_l3_24h_20260524T204451Z, 20260509_18z_l3_24h_20260511T190519Z, 20260521_18z_l3_24h_20260522T072630Z}:

- 1h Canary: theta in [200,700]K all 360 steps, |u|,|v| ≤ 100 m/s, |w| ≤ 50 m/s.
- B6 preserved: `python scripts/m6b6_coupled_step_compare.py --tier all` → 0.0 bitwise.
- Multi-step parity: `python scripts/m6b_real_ic_operational_compare.py --steps 10` → 0.0 bitwise.

Write `proof_bounds_parity.json` with per-IC results.

### Stage 2 — Tier-4 RMSE

For each IC at t=1h:
- Run operational forecast (use the already-compiled JIT from Stage 1 if possible).
- Extract operational wrfout-equivalent state at t=1h.
- Read Gen2 wrfout at the matching hour.
- Compute spatial-mean RMSE on T2, U10, V10.
- Compute spatial heterogeneity ratio max|err|/mean|err| (informational; per Tier-4 tester report a tight ratio of 1.5 is unrealistic for real forecasts).

Envelope:
- T2 RMSE ≤ 3.0 K
- U10 RMSE ≤ 7.5 m/s
- V10 RMSE ≤ 7.5 m/s

Aggregate: pass iff per-IC AND mean-across-3-ICs both pass.

Write `proof_tier4_rmse_all3.json`.

### Stage 3 — M6 closeout decision

Write `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` with:

- **Status**: `M6-CLOSED` if Stage 1+2 GREEN; `M6-BLOCKED-<reason>` otherwise.
- **Acceptance evidence**: links to all 4 fix sprint reports + Stage 1/2 proofs.
- **Outstanding caveats**: list (e.g., V workaround is suppression not root-cause; microphysics is guard not root-cause; boundary/dynamics audit pending).
- **Recommended next steps for M7**: profiling gate, GPU optimization, perf comparison vs 28-rank CPU WRF.

### Stage 4 — Worker report

`worker-report.md` with `Summary:`, all proofs, final status. >=400 bytes.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6acc
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

taskset -c 0-3 python scripts/m6_acceptance_tier4_all3.py --output .agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/

taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10
taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/v3_521/
taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/v3_509/

git add -A && git commit -m "[M6 acceptance] $(date -u +%FT%TZ)"
```

## Handoff

Per universal spec.
