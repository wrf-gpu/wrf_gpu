# Sprint Contract — M6b V3 Localize 20260509 Theta Explosion (post-reboot restart)

## Objective

A previous V3 localization sprint died with /tmp wipe on reboot. This sprint **restarts the 20260509-only branch**: pinpoint whether the theta explosion mid-run on Gen2 ID `20260509_18z_l3_24h_20260511T190519Z` is **PHYSICAL** (this IC is genuinely tough — WRF reference also strains), **MATH** (operator defect surfaces on this IC), or **IC-SPECIFIC** (this IC has corrupt or unusual boundary forcing).

## Non-Goals

- NO modifications to `dynamics/core/` (locked at 0.0 bitwise B6).
- NO modifications to `operational_mode.py` body.
- NO 24h forecast.
- NO sanitizer.
- NO remote push.

## File Ownership

Worktree **already created** at `/tmp/wrf_gpu2_loc_509` on branch `worker/gpt/m6b-v3-localize-20260509-theta`.
Your FIRST command must be `cd /tmp/wrf_gpu2_loc_509` — do everything else from there.

Write-only:
- `scripts/m6b_v3_localize_509.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/` — proofs + localization_memo.md + worker-report.md

Read-only: same as 20260521 sprint.

## Inputs

1. This sprint contract.
2. `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/` — V3 outcomes.
3. Gen2 wrfout truth at `/mnt/data/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260511T190519Z/wrfout_d02_*`.
4. The 13 `scripts/diagnostic_*.py` helpers.

## Acceptance Criteria

### Stage 1 — Re-run V3 on 20260509 only (1h)

PINNED to `20260509_18z_l3_24h_20260511T190519Z`. Capture per-step max/min theta per level + first step at which any per-level theta bound fires (lower 30 levels [200,400]K, upper 14 levels [250,700]K).

Write `proof_theta_explosion.json` with:
- first step + level + cell of bound violation
- per-step theta min/max timeline for each level
- field snapshots at the violating cell at step N-1 and N

### Stage 2 — Physical reality check via Gen2 wrfout

At the same (lat, lon, level, time) of the violation cell, look up WRF reference theta value. Report:
- WRF reference theta at that cell over time
- WRF reference horizontal extremes at that level at the violation hour
- Convective signal: look at wrfout T, QVAPOR, W at neighbors — is the model in a strong convection region?

If WRF reference itself shows theta drifting toward the violation bound → **PHYSICAL**, the bound is too tight for this regime, recommend ENVELOPE-EXTENSION for stratosphere or convective columns.

If WRF reference is benign at that cell → **NAMED-FIX or IC-SPECIFIC path**, proceed to Stage 3.

Write `proof_wrf_reference_theta.json`.

### Stage 3 — Discriminate MATH vs IC-SPECIFIC

Run:
- `diagnostic_first_bad_step_tracer.py` to find first divergent step vs WRF reference (delta > 1e-6, not just bound violation).
- `diagnostic_vertical_column_phase_space.py` on the violating column → is the vertical structure pathological from the start?
- `diagnostic_boundary_ring_error_profiler.py` at the violating step → is the explosion coming from the boundary ring (forcing) or from the interior?

If first divergence step is 1 and the boundary ring is benign → **MATH** (operator defect, surfaces only on this IC's flow regime). Run `diagnostic_operator_term_budget_tracer.py` to name the term.

If first divergence step is 1 but boundary-ring error is large → **IC-SPECIFIC** (boundary forcing problem from AIFS d02 backfill).

If first divergence step >> 1 and is gradual → **NUMERICAL-DRIFT**, suggesting precision or accumulator issue.

Write `proof_math_vs_ic.json`.

### Stage 4 — Localization memo

Write `localization_memo.md`:
- **Verdict**: `PHYSICAL` | `MATH:<operator>` | `IC-SPECIFIC` | `NUMERICAL-DRIFT` | `INSUFFICIENT-EVIDENCE`
- **Evidence**: 5-8 bullets.
- **Recommended next sprint**: exact name + 1-line scope.
- **Risks / caveats**: cross-link the GPU-vs-CPU sprint if step-2 NaN reproduces here.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_loc_509
export OMP_NUM_THREADS=4
export PYTHONPATH="src"
taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/
git add -A && git commit -m "[V3 localize 20260509] $(date -u +%FT%TZ)"
```

## Handoff

Same shape as 20260521 sprint.
