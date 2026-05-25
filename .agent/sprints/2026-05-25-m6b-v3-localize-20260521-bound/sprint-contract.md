# Sprint Contract — M6b V3 Localize 20260521 Wind Bound (post-reboot restart)

## Objective

A previous V3 localization sprint died with /tmp wipe on reboot. This sprint **restarts the 20260521-only branch**: pinpoint whether the v=103.7 m/s bound violation at step 46 of the 1h Canary on Gen2 ID `20260521_18z_l3_24h_20260522T072630Z` is **PHYSICAL** (real weather, our 100 m/s bound is too tight) or **MATH** (operator-level defect surfaced only after step 45).

Decide between:
- **BOUND-REVISION** — relax v bound to 120 m/s, document with WRF reference showing wind near that magnitude.
- **NAMED-FIX** — identify the specific operator/coupling defect causing acceleration past step 45.

## Non-Goals

- NO modifications to `dynamics/core/` (locked at 0.0 bitwise B6).
- NO modifications to `operational_mode.py` body.
- NO 24h forecast.
- NO sanitizer.
- NO remote push.

## File Ownership

Worktree **already created** at `/tmp/wrf_gpu2_loc_521` on branch `worker/gpt/m6b-v3-localize-20260521-bound`.
Your FIRST command must be `cd /tmp/wrf_gpu2_loc_521` — do everything else from there.

Write-only:
- `scripts/m6b_v3_localize_521.py` (NEW) — drives the targeted localizer
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/` — proof JSONs, localization_memo.md, worker-report.md

Read-only:
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/dynamics/core/`
- `src/gpuwrf/dynamics/validation_wrappers.py`
- `scripts/m6b_canary_1h_honest_v3.py` (your reference for how V3 was driven)
- `scripts/diagnostic_*.py` (your tool arsenal)

## Inputs

1. This sprint contract.
2. `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/` — V3 outcomes that produced this blocker.
3. Gen2 wrfout truth at `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/wrfout_d02_2026-05-21_*` (12 hourly files spanning t=18:00 to t=05:00 next day).
4. The 13 `scripts/diagnostic_*.py` helpers.

## Acceptance Criteria

### Stage 1 — Re-run V3 on 20260521 only (1h)

Set `PINNED_RUN_IDS = ("20260521_18z_l3_24h_20260522T072630Z",)` and run the 1h forecast. Capture the per-step max(|u|, |v|, |w|) timeline (this should be a NEW small helper, not the full V3 driver).

PASS/FAIL doesn't matter; we EXPECT the v=103.7 violation. Write `proof_step46_violation.json` with:
- step at which bound first fires
- max |v| value and (k, j, i) location
- field snapshots (theta, mu, u, v, w, ww) at that cell at step 45 and 46

### Stage 2 — Physical reality check via Gen2 wrfout

For the same (lat, lon, level, time) of the step-46 cell, look up the **WRF reference value** in the Gen2 wrfout for that hour. Report:
- WRF reference max |V| at that horizontal level + nearby cells
- WRF reference vertical max wind anywhere on the domain at that hour

If WRF reference itself shows |V| > 90 m/s at or near that cell → **PHYSICAL**, bound too tight, recommend BOUND-REVISION to ~120 m/s.

If WRF reference is well below 90 m/s at that location → **NAMED-FIX path**, proceed to Stage 3.

Write `proof_wrf_reference_compare.json`.

### Stage 3 — Operator term budget at step 46 (only if Stage 2 says NAMED-FIX)

Run `diagnostic_operator_term_budget_tracer.py` at step 45 → 46 at the bad cell. Decompose dv/dt into pressure-gradient, advection, Coriolis, vertical advection, RK acoustic. Identify which term dominates. Cross-check against WRF `module_small_step_em.F` operator at the same time-step.

Run `diagnostic_first_bad_step_tracer.py` with finer time resolution between step 40 and 46 to localize the earliest detectable divergence vs WRF reference (not just bound violation, but actual delta growth).

Write `proof_operator_decomposition.json` and `proof_first_divergent_step.json`.

### Stage 4 — Localization memo (mandatory deliverable)

Write `localization_memo.md` with sections:
- **Verdict**: one of `BOUND-REVISION` | `NAMED-FIX:<operator>` | `INSUFFICIENT-EVIDENCE`
- **Evidence summary**: 5-8 bullets citing the 3 proof JSONs.
- **Recommended next sprint**: exact follow-up sprint name + 1-line scope.
- **Risks / caveats**: physical realism caveats, GPU-vs-CPU concerns (cross-link the gpu-cpu-step2 sprint if relevant).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_loc_521
export OMP_NUM_THREADS=4
export PYTHONPATH="src"
taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/
git add -A && git commit -m "[V3 localize 20260521] $(date -u +%FT%TZ)"
```

## Handoff (worker-report.md)

Must include `Summary:`, files changed, commands run + outputs (head/tail), proof object paths, the verdict from `localization_memo.md`, risks, and explicit handoff naming the next sprint.
