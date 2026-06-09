# Review: V0.14 Grid After Live-Nest Base Fix

Date: 2026-06-09
Reviewer: GPT-5.5 xhigh
Sprint: `.agent/sprints/2026-06-09-v014-grid-after-base-direct/sprint-contract.md`

## Findings

1. **P0 - Grid symptom is not closed.**
   Evidence: `proofs/v014/grid_after_live_nest_base.md` reports h1-h12 V10 RMSE `2.55039100124724` m/s, worst V10 lead h11 RMSE `4.277008742661733` m/s, PSFC RMSE `517.1905702423264` Pa, P RMSE `230.30713670774634` Pa, MU RMSE `266.52491970646497`, and PH RMSE `292.3872984317863`. This fails any practical grid-parity closure interpretation and supports verdict `GRID_SYMPTOM_NOT_CLOSED`.

2. **P1 - Base/static payload improved materially, but PB/MUB are not exact.**
   Evidence: C1H/C2H/C4H/DN/RDN/MAPFAC_M/XLAT/XLONG are exact in the fresh comparator, HGT max abs is `3.0517578125e-05`, PHB max abs is `0.109375`, but PB max abs remains `249.8828125` and MUB max abs remains `250.671875`. This is a major improvement versus the retained/grid-envelope artifacts, not full static/base closure.

3. **P2 - Runtime was recorded, peak VRAM was not.**
   Evidence: `proofs/v014/grid_after_live_nest_base/gpu_h12/wall_clock_l2_d02.json` records total wall `1192.2986149120043` s and forecast-only `1186.442607951998` s on `cuda:0`, but no peak VRAM field is present. The final report correctly says VRAM was not recorded.

## Acceptance Check

- Branch includes `7d11be42`: pass (`git merge-base --is-ancestor 7d11be42 HEAD` returned rc 0).
- Exactly one bounded GPU run: pass. One `scripts/run_gpu_lowprio.sh` h12 command was started; it exited 0 with `L2_D02_GREEN`.
- TOST not resumed: pass. No TOST orchestration command was run.
- No production `src/` edits: pass. Changes are proof/review scoped.
- JSON validates: pass. Comparator and synthesis JSON are parseable.
- Required comparison artifacts included: pass. `post_static_writer_grid_compare`, `grid_cell_envelope`, and `v10_grid_diagnostics` are compared in `proofs/v014/grid_after_live_nest_base.md` and `sprint_synthesis`.

## Decision

Accept as a valid not-closed proof. The correct next target is same-state dynamic localization in the h10-h12 window, centered on pressure-gradient/mass-wind coupling for PSFC, MU, P, PH, U/V, and V10.

## Evidence Paths

- `proofs/v014/grid_after_live_nest_base.json`
- `proofs/v014/grid_after_live_nest_base.md`
- `proofs/v014/grid_after_live_nest_base/gpu_h12/`
- `/mnt/data/wrf_gpu2/v014_grid_after_live_nest_base/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
