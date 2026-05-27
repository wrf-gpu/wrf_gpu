# Worker Report - M7 Skill Fix Iter 2

Summary: Implemented the contracted theta-envelope, hourly land-state refresh, and 5-row boundary-strip changes. Verdict: `BLOCKED`. AC6 invariants and AC7 speedup are preserved, but AC5 fails: GPU station skill is still outside +/-20% of CPU for T2/U10/V10, and most aggregate RMSE/MAE wind metrics worsened versus the predecessor partial-fix run. This is not `SKILL_FIXED` or `SKILL_IMPROVED_AGAIN`.

## Files Changed

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/io/land_state.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/coupling/boundary_apply.py`
- `tests/test_m7_skill_fix_iter2.py`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/**`

## Implementation Summary

- AC1: Added and ran `theta_saturation_probe.py`; produced `theta_saturation_diagnosis.md/json`.
- AC2: Kept the lower theta floor at 200 K and widened the lower-30 ceiling from 400 K to 450 K. The comment in `_physics_boundary_step` records why this is an envelope guard, not a diurnal clamp.
- AC3: Added `load_hourly_land_state()` and wired `daily_pipeline` to refresh exposed state fields hourly from Gen2 wrfout TSK/SST/SMOIS/SH2O/TSLB payloads.
- AC4: Packed WRF-ordered 5-row boundary strips and extended `apply_lateral_boundaries()` to use the strip matching each relax-zone offset.
- AC5: Re-ran 24h 20260521 d02 pipeline and skill diff. Failed skill gate.
- AC6/AC7: Invariants pass and d02-only speedup remains above target.

## Commands Run And Output

`taskset -c 0-3 python .agent/sprints/2026-05-27-m7-skill-fix-iter2/theta_saturation_probe.py`

```text
{"json": ".../theta_saturation_diagnosis.json", "output": ".../theta_saturation_diagnosis.md", "status": "PASS"}
```

`taskset -c 0-3 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 1 --output-dir /tmp/m7_skillfix_iter2_ac2_1h --proof-dir .agent/sprints/2026-05-27-m7-skill-fix-iter2/probe_ac2 --restart-at-hour 0`

```text
exit=2; PIPELINE_BLOCKED; reason="--restart-at-hour must be between 1 and hours - 1"
```

`taskset -c 0-3 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 1 --output-dir /tmp/m7_skillfix_iter2_ac2_1h_rerun --proof-dir .agent/sprints/2026-05-27-m7-skill-fix-iter2/probe_ac2_rerun`

```text
exit=2; PIPELINE_PARTIAL because --score was not requested.
finite=true; inventory=PASS; speedup=PASS; wall_clock_total_s=306.2077883799939.
```

`taskset -c 0-3 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 1 --output-dir /tmp/m7_skillfix_iter2_ac3_1h --proof-dir .agent/sprints/2026-05-27-m7-skill-fix-iter2/probe_ac3`

```text
exit=2; PIPELINE_PARTIAL because --score was not requested.
finite=true; inventory=PASS; speedup=PASS; land_refresh hour 1 PASS; wall_clock_total_s=301.7095990579983.
```

`taskset -c 0-3 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 1 --output-dir /tmp/m7_skillfix_iter2_ac4_1h --proof-dir .agent/sprints/2026-05-27-m7-skill-fix-iter2/probe_ac4`

```text
exit=2; PIPELINE_PARTIAL because --score was not requested.
finite=true; inventory=PASS; speedup=PASS; boundary schema=history-strip-pack-v2; bdy_width=5; wall_clock_total_s=319.24538214199856.
```

`taskset -c 0-3 pytest -q tests/test_m7_skill_fix_iter2.py tests/test_m7_skill_fix_algorithmic.py`

```text
first run: 1 failed, 7 passed; fixture used nz=2 where DycoreMetrics requires nz >= 3.
second run: 1 failed, 7 passed; expected relax-zone value corrected from 2.0 to 2.2.
final run: 8 passed in 13.43s.
```

`taskset -c 0-3 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 1 --output-dir /tmp/m7_skillfix_iter2_final_1h --proof-dir .agent/sprints/2026-05-27-m7-skill-fix-iter2/probe_final_1h`

```text
exit=2; PIPELINE_PARTIAL because --score was not requested.
finite=true; inventory=PASS; speedup=PASS; land_refresh hour 1 PASS; boundary bdy_width=5; wall_clock_total_s=338.90945423099765.
```

`taskset -c 0-3 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 24 --output-dir /tmp/m7_skillfix_iter2_20260521 --proof-dir .agent/sprints/2026-05-27-m7-skill-fix-iter2 --restart-at-hour 12`

```text
exit=2; PIPELINE_PARTIAL because --score was not requested.
finite=true; wrfout_inventory=PASS; restart_probe=PASS; speedup=PASS;
24 wrfout files; 24 hourly land refresh records PASS; boundary bdy_width=5;
wall_clock_total_s=732.6321056330053; forecast_only_s=687.8953256039749.
```

`taskset -c 0-3 python scripts/m7_gpu_vs_cpu_skill_diff.py --gpu-root /tmp/m7_skillfix_iter2_20260521 --output .agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json`

```text
{"common_valid_time_count": 24, "station_count_scored": 73, "status": "FAIL_SKILL_DIFF", "output": ".agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json"}
```

Skill summary:

```text
T2 gpu/cpu: bias 6.3547/0.3437, rmse 10.8013/2.1487, mae 8.2384/1.6807.
U10 gpu/cpu: bias 3.1052/-0.1352, rmse 7.2380/2.3065, mae 5.6278/1.7122.
V10 gpu/cpu: bias -0.0338/-0.6700, rmse 7.6217/2.7523, mae 6.0168/1.9725.
all_variables_within_20pct=false.
```

`taskset -c 0-3 python scripts/m7_cpu_per_domain_timing.py --pipeline-run .agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json --cpu-output .agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_cpu_timing.json --speedup-output .agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_speedup.json`

```text
{"status": "PASS", "selected_cpu_run_id": "20260521_18z_l3_24h_20260522T133443Z", "d02_only_speedup": 22.25579686534753}
```

`taskset -c 0-3 pytest -q tests/test_m7_restart_checkpoint_roundtrip.py tests/test_m6c_20260509_mu_regression.py tests/test_m6b6_coupled_step_parity.py tests/test_m6b_d2h_warmed_zero_v2.py -k 'not artifacts_present'`

```text
15 passed, 1 deselected in 49.03s
```

`python <invariant_preservation_iter2 generator>`

```text
{"status": "PASS", "checks": {"20260521_multistep_parity_step2_bitwise": true, "20260521_multistep_parity_final": true, "b6_savepoint_parity": true, "d2h_inter_kernel": true, "restart_bitwise": true}}
```

`python -m py_compile src/gpuwrf/runtime/operational_mode.py src/gpuwrf/io/land_state.py src/gpuwrf/integration/daily_pipeline.py src/gpuwrf/integration/d02_replay.py src/gpuwrf/coupling/boundary_apply.py .agent/sprints/2026-05-27-m7-skill-fix-iter2/theta_saturation_probe.py`

```text
exit=0; no stdout/stderr.
```

`taskset -c 0-3 python scripts/validate_agentos.py`

```json
{"errors":[],"ok":true,"required_files_checked":31,"skills_checked":13}
```

## Proof Objects Produced

- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/theta_saturation_diagnosis.md`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/theta_saturation_diagnosis.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/wrfout_inventory.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/restart_in_pipeline.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_speedup.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_cpu_timing.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/invariant_preservation_iter2.json`
- Probe folders: `probe_ac2/`, `probe_ac2_rerun/`, `probe_ac3/`, `probe_ac4/`, `probe_final_1h/`

## Risks

- `BLOCKED`: AC5 remains outside +/-20% for all three variables. This sprint's three named fixes are not sufficient for publication-grade skill.
- The hourly land refresh is a data replay path, not prognostic Noah-MP evolution. It improves lower-boundary realism but does not prove a production land model.
- The 5-row strip uses d02 hourly history strips ordered using wrfbdy width/orientation metadata. That is still replay/nudging, not native WRF `wrfbdy` forecast forcing.
- The 24h pipeline exits 2 because score mode was not requested; skill comparison is supplied by the separate CPU-vs-GPU diff artifact.

## Handoff

- objective: implement and validate M7 skill-fix iter2 theta envelope, land-state refresh, and 5-row boundary width changes.
- files changed: listed above; no governance, goal, reviewer, tester, manager, memory, dynamics, validation, checkpoint, or wrfout-writer files modified.
- commands run: listed above with outputs.
- proof objects produced: listed above.
- unresolved risks: skill gate failed; publication should characterize M7 skill as blocked, not fixed.
- next decision needed: manager should decide the next RCA sprint. Field-level evidence points to remaining surface/PBL and boundary-coupling error, not a broken invariant or speed regression.
