# Sprint Contract — M7 Honest Speedup + GPU-vs-CPU Skill Diff

**Sprint ID**: `2026-05-27-m7-honest-speedup-skill-diff`
**Created**: 2026-05-27 (user direction: publish-ready validation before backfills)
**Status**: READY — top priority
**Predecessor**: `.agent/decisions/MILESTONE-M7-CLOSEOUT.md` (156.82× preliminary, flagged as potentially inflated due to nest-domain accounting)

## Objective

The preliminary 156.82× speedup claim compared **GPU d02-only** wall-clock against a **CPU baseline timing denominator** that may have aggregated all 5 nested domains in the Gen2 nightly run. Before publishing, two things must be honest:

1. **Apples-to-apples speedup**: isolate the CPU d02-only timing from the existing Gen2 `namelist.output` per-step records (no fresh CPU runs — they're on disk), and recompute the speedup ratio.
2. **GPU vs CPU skill diff**: run the existing forecast-vs-obs scaffold on BOTH the GPU forecast (already exists at `/tmp/m7_pipeline_runs/20260521/`) AND the corresponding Gen2 CPU wrfout for 20260521. Report side-by-side BIAS/RMSE/MAE on T2/U10/V10 against AEMET stations.

Both deliverables go into a single closeout-update memo. **If the honest speedup is below 4× OR if GPU skill is materially worse than CPU on T2/U10/V10**, the M7 closeout must be amended.

## Acceptance

- **AC1 — CPU per-domain timing extraction**: parse `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_*/namelist.output`. WRF's `Timing for main` lines are per-step per-domain. Extract: total integration steps × wall-time per step, grouped by domain (d01-d05). Compute "d02-only" cumulative wall-clock as the sum of `Timing for main: time ... on domain 2:` lines. Emit `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/cpu_per_domain_wall_clock.json` with: domain, step count, total wall, mean per-step.

- **AC2 — Honest speedup table**: produce `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` with at minimum the following comparisons (against GPU d02 24h end-to-end = 324.78 s from `pipeline_run_20260521.json`):
  - vs CPU full-nest 5-domain 24h wall
  - vs CPU d02-only 24h cumulative wall
  - vs CPU d02 + d01 24h wall (the minimum subset that makes physical sense — d02 needs d01 BCs)
  - vs CPU d01-only 24h wall (for completeness)
  Each row: GPU wall, CPU wall, ratio, what's-being-compared, fairness verdict.

- **AC3 — Side-by-side AEMET skill diff**: use `gpuwrf.validation.forecast_vs_obs` on TWO sources for 20260521:
  - GPU wrfouts at `/tmp/m7_pipeline_runs/20260521/wrfout_d02_*` (the 24 hourly NetCDFs already produced)
  - CPU reference wrfouts at `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_*/wrfout_d02_*`
  For each: per-station BIAS, RMSE, MAE on T2, U10, V10 across all 24 hours. Aggregate metrics. Emit `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json`.

- **AC4 — Verdict memo**: write `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/verdict.md` answering:
  - **Honest speedup**: what's the apples-to-apples number?
  - **Skill**: is GPU's BIAS/RMSE/MAE within ±20% of CPU's on T2/U10/V10? (Pre-declared tolerance; rationale: a fair port should match operational CPU skill within typical noise.)
  - **Publication-ready?**: explicit YES / NEEDS-CAVEAT / NO with the precise number to publish.

- **AC5 — Tests**: add `tests/test_m7_honest_speedup.py` with unit tests for the per-domain timing parser (synthetic `Timing for main` lines, expected aggregates).

- **AC6 — Worker report** with verdict.

## Files Worker May Modify

- `scripts/m7_cpu_per_domain_timing.py` (NEW)
- `scripts/m7_gpu_vs_cpu_skill_diff.py` (NEW)
- `tests/test_m7_honest_speedup.py` (NEW)
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/**`

## Files Worker Must Not Modify

- `src/gpuwrf/**` — measurement and analysis only
- `src/gpuwrf/validation/forecast_vs_obs.py` — frozen, just use it
- governance files
- `.agent/decisions/MILESTONE-M7-CLOSEOUT.md` — do not amend; this sprint produces an UPDATE memo instead
- `/mnt/data/canairy_meteo/**` — read-only

## Hard Rules

1. **No fresh CPU WRF runs.** Use existing namelist.output timings. The nightly is on cores 4-31; do not interfere.
2. **CPU pinning**: `taskset -c 0-3` for the analysis scripts.
3. **No remote push.** Local commit on `worker/gpt/m7-honest-speedup-skill-diff`.
4. **Read-only on Gen2 data**.
5. **Be ruthless with the speedup denominator**: if Gen2's `namelist.output` doesn't cleanly break out d02-only, document the precision and prefer the conservative bound. Better a defensible 30× than an overstated 156×.

## Dependencies

- M7 perf-measurement closeout commit `b7d9fe7`
- GPU forecast files at `/tmp/m7_pipeline_runs/20260521/` (still present from pipeline integration sprint)
- Gen2 reference at `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_*/`

## Proof Objects

- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/cpu_per_domain_wall_clock.json` (AC1)
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` (AC2)
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json` (AC3)
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/verdict.md` (AC4 — the publication-readiness call)
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/worker-report.md`

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-4 h
- Branch: `worker/gpt/m7-honest-speedup-skill-diff`
- Worktree: `/tmp/wrf_gpu2_honest`
- GPU usage: NONE for AC1/AC2/AC4; minimal for AC3 (forecast-vs-obs scaffold is host-side pandas/xarray on existing wrfouts, no fresh forecast). Safe to run concurrently with the L2 d02 replay sprint.
