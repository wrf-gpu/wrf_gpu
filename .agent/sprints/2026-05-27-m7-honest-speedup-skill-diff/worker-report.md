Summary: HONEST_M7_UPDATE_READY. Implemented the measurement-only M7 honest speedup and GPU-vs-CPU skill diff sprint. The defensible d02-only speedup is 50.204847x against de-duplicated CPU d02 timing from the complete Gen2 20260521 run. The previous 156.82x denominator is not publishable. Skill gate fails: GPU aggregate BIAS/RMSE/MAE is materially worse than CPU for T2/U10/V10 and outside the pre-declared +/-20% tolerance.

## Objective

Recompute M7 speedup using per-domain Gen2 CPU timing records and run side-by-side AEMET station scoring for GPU and CPU 20260521 d02 wrfouts.

## Files Changed

- `scripts/m7_cpu_per_domain_timing.py`
- `scripts/m7_gpu_vs_cpu_skill_diff.py`
- `tests/test_m7_honest_speedup.py`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/cpu_per_domain_wall_clock.json`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/verdict.md`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/command_outputs/*`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/worker-report.md`

## Commands Run

All Python commands were pinned with `taskset -c 0-3`.

1. `python -m py_compile scripts/m7_cpu_per_domain_timing.py scripts/m7_gpu_vs_cpu_skill_diff.py`
   - stdout: empty
   - stderr: empty
2. `pytest -q tests/test_m7_honest_speedup.py`
   - stdout: `3 passed in 0.01s`
   - stderr: empty
3. `python scripts/m7_cpu_per_domain_timing.py`
   - stdout: `status=PASS`, selected CPU run `20260521_18z_l3_24h_20260522T133443Z`, d02-only speedup `50.20484702814852`
   - stderr: empty
4. `python scripts/m7_gpu_vs_cpu_skill_diff.py`
   - stdout: `status=FAIL_SKILL_DIFF`, common valid times `24`, stations scored `73`
   - stderr: empty
5. `pytest -q tests/test_m7_honest_speedup.py tests/test_m7_forecast_vs_obs.py`
   - stdout: `10 passed in 0.34s`
   - stderr: empty
6. Proof JSON validation one-liner
   - stdout: selected run `20260521_18z_l3_24h_20260522T133443Z`, d02 speedup `50.20484702814852`, skill verdict `FAIL_SKILL_DIFF`
   - stderr: empty
7. `python scripts/validate_agentos.py`
   - stdout: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`
   - stderr: empty

Full stdout/stderr is captured under `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/command_outputs/`.

## Proof Objects Produced

- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/cpu_per_domain_wall_clock.json`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/verdict.md`

## Results

CPU `namelist.output` files contain zero `Timing for main` lines. The timing artifact therefore scans the requested namelist paths plus sibling `rsl.error.0000` / `rsl.out.0000`, then de-duplicates mirrored records before computing totals. The complete 24h CPU run is `20260521_18z_l3_24h_20260522T133443Z`; the earlier `20260521_18z_l3_24h_20260522T072630Z` timing is incomplete for this purpose.

Honest speedup rows:

- d02-only: CPU `16,305.31132 s` / GPU `324.77564 s` = `50.204847x`
- d01+d02 physical subset: `102.621968x`
- d01-d05 aggregate: `138.240484x`
- d01-only context: `52.417121x`

Skill diff fails the +/-20% tolerance. Aggregate CPU vs GPU RMSE:

- T2: CPU `2.148693`, GPU `7.858779`
- U10: CPU `2.306471`, GPU `11.311116`
- V10: CPU `2.752321`, GPU `9.435313`

## Risks

- The timing source is not literally `namelist.output`; those files had no timing records. This is documented in the JSON and verdict.
- The d01-d05 aggregate row is a conservative denominator diagnostic, not an apples-to-apples wall-clock claim.
- Skill scoring uses the existing station scaffold only; it does not diagnose why GPU skill is worse.

## Handoff

Objective complete. Manager should amend the M7 closeout before publication. Publishable timing-only number, if allowed with caveat, is **50.20x d02-only**. The combined speedup-and-skill claim is **not publication-ready**.
