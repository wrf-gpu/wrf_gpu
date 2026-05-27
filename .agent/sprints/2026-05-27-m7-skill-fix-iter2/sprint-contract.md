# Sprint Contract — M7 Skill Fix Iter 2 (theta envelope + land state + boundary width)

**Sprint ID**: `2026-05-27-m7-skill-fix-iter2`
**Created**: 2026-05-27 (user direction: finish project + publication)
**Status**: READY — second iteration after partial fix
**Predecessor**: `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/worker-report.md` (SKILL_IMPROVED_PARTIAL)

## Objective

The first algorithmic fix sprint (A+B+C) landed `SKILL_IMPROVED_PARTIAL`. Wind metrics improved; T2 worsened because the theta guard envelope `[200K, 400K]` for the lower 30 levels saturates the diurnal warming maximum. The post-fix `theta_lower_30_max_k` varies only `0.0067 K` across 24 hours — i.e., the cap is constantly active.

Two interpretations are possible:
1. The cap is too tight and is suppressing real diurnal warming (cap should be raised, e.g. to `[200K, 450K]` or wider).
2. The cap is masking a real runaway — something downstream of the surface→PBL fix is causing theta to grow unphysically (cap is doing safety work but hides the real bug).

This sprint investigates both **and** addresses two further named defects from the RCA:
- **Land state evolution**: `t_skin`, `SST`, `SMOIS`, `SH2O`, `TSLB` are frozen at IC. Add a minimal evolution path (either a prognostic Noah-MP-style update OR a Gen2-hourly refresh) so surface temperature can track diurnal forcing.
- **Lateral boundary width**: `_field_sides_3d` currently uses only the outermost parent row. Widen to WRF's `spec_bdy_width=5` strip with the standard relax-zone Newtonian profile, using the existing `decode_wrfbdy` machinery.

## Acceptance

- **AC1 — Diagnose theta saturation**: produce `.agent/sprints/2026-05-27-m7-skill-fix-iter2/theta_saturation_diagnosis.md` answering: at which time, level, and grid cells does theta first hit the 400 K cap? Is the cap clipping every step or only some? Is the surface heat flux too large, the PBL mixing too small, or both? Document with a 1-hour instrumented forecast that records pre-cap and post-cap theta histograms each step.

- **AC2 — Theta envelope decision**: based on AC1, EITHER widen the envelope to a defensible bound (e.g. `[180K, 450K]` for lower 30 levels — justify the choice physically) OR identify and fix the runaway source. Commit the chosen path with a defensible comment in `_physics_boundary_step`.

- **AC3 — Land state evolution**: implement minimal land-state time evolution. Recommended approach (lowest risk): hourly refresh from Gen2 `wrfout_d02_*` for `t_skin`, `SST`, `SMOIS`, `SH2O`, `TSLB`. This is a data path (Gen2 hourly outputs → State land fields), not a prognostic scheme. Add the helper in `gpuwrf/io/land_state.py` and wire into `daily_pipeline.py` hourly step boundary.

- **AC4 — Boundary width fix**: replace `_field_sides_3d` outermost-row pack with `decode_wrfbdy`-based 5-row strip in `gpuwrf/integration/d02_replay.py`. Use existing `apply_lateral_boundaries` with `spec_bdy_width=5, spec_zone=1, relax_zone=4` as the WRF namelist demands.

- **AC5 — Re-run 24h pipeline + skill diff**: re-run the 20260521 case with all 3 fixes applied. Emit `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json` with GPU vs CPU AEMET BIAS/RMSE/MAE on T2/U10/V10. Target: ALL three variables within ±20% of CPU.

- **AC6 — Invariant preservation (HARD)**: 20260521 multi-step step 2 = 0.0 bitwise; B6 = 0.0 bitwise; D2H inter-kernel = 0; restart bitwise PASS. If ANY regresses, the offending change must be revised before merge.

- **AC7 — Speedup re-check**: emit `post_iter2_speedup.json`. May dip further if land-state refresh and 5-row boundary add overhead. Expected: ≥ 4-8× target preserved.

- **AC8 — Worker report** with verdict `SKILL_FIXED` (±20% met on all 3 vars) / `SKILL_IMPROVED_AGAIN` (further improvement but not ±20%) / `BLOCKED`.

## Files Worker May Modify

- `src/gpuwrf/runtime/operational_mode.py` (AC2 if envelope widening; if runaway source found, fix in physics_couplers.py instead)
- `src/gpuwrf/coupling/physics_couplers.py` (AC2 runaway fix)
- `src/gpuwrf/io/land_state.py` (AC3 hourly refresh helper)
- `src/gpuwrf/integration/daily_pipeline.py` (AC3 wiring)
- `src/gpuwrf/integration/d02_replay.py` (AC4 5-row boundary pack)
- `src/gpuwrf/coupling/boundary_apply.py` (AC4 — ONLY if the existing apply path can't consume 5-row input; minimal change)
- `tests/test_m7_skill_fix_iter2.py` (NEW)
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/**`

## Files Worker Must Not Modify

- `src/gpuwrf/dynamics/**` — dycore is correct (B6 0.0 bitwise)
- `src/gpuwrf/contracts/state.py`, `contracts/precision.py` — State schema frozen
- `src/gpuwrf/runtime/checkpoint.py` — frozen
- `src/gpuwrf/io/wrfout_writer.py` — frozen (LU_INDEX writer cleanup is M8, not this sprint)
- `src/gpuwrf/validation/**` — frozen
- governance files
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **Invariants are hard gates** (AC6).
2. **One fix at a time on first 1h probe**: when each of AC2/AC3/AC4 is implemented, run a fast 1h probe and check it doesn't regress invariants before moving to the next. Don't combine until each works.
3. **CPU pinning**: `taskset -c 0-3`.
4. **GPU**: yes.
5. **No remote push.** Local commit on `worker/gpt/m7-skill-fix-iter2` only.
6. **Honest BLOCKED**: if AC5 still shows skill outside ±20%, emit BLOCKED with field-level diagnosis. We characterize honestly in the publication.

## Proof Objects

- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/theta_saturation_diagnosis.md` (AC1)
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json` (AC5 — gate)
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_speedup.json` (AC7)
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/invariant_preservation_iter2.json` (AC6)
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/worker-report.md` (AC8)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 6-12 h (substantial — 3 coupled fixes + re-validation)
- Branch: `worker/gpt/m7-skill-fix-iter2`
- Worktree: `/tmp/wrf_gpu2_skillfix2`
- GPU usage: YES
