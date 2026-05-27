# Worker Report — M7 Skill Fix Algorithmic

Summary: Implemented the scoped A+B+C code changes, produced the required proof objects, and re-ran the 20260521 24h pipeline twice. Decision: `SKILL_IMPROVED_PARTIAL`. The fix is not shippable as `SKILL_FIXED`: AC2 still fails because `theta_lower_30_max_k` is bounded but pinned near 400 K, and AC6 remains outside ±20% skill tolerance. It is not `no improvement`: 6 of 9 T2/U10/V10 aggregate metrics improved versus the pre-fix skill-diff baseline, mainly wind metrics.

## Files Changed

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- `tests/test_m7_skill_fix_algorithmic.py`
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/invariant_preservation.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_bounds.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json`
- Additional pipeline byproducts in this sprint folder: `pipeline_run_20260521.json`, `restart_in_pipeline.json`, `wrfout_inventory.json`, `station_scores_20260521.json`, `speedup_vs_cpu_24h.json`, `repeatability.json`, `post_fix_cpu_timing.json`.

## Summary Of Code Changes

- Removed the whole-state theta/mu reset from `_physics_boundary_step`; RK3/acoustic output now flows forward through a bounded guard.
- Added theta and mu guard helpers: in-range RK theta is preserved; out-of-range/nonfinite theta falls back to the previous bounded state and is clipped to the sprint envelope; mu perturbation is kept only when total dry-column mass remains positive.
- Reordered operational physics to `thompson_adapter` → `surface_adapter` → `mynn_adapter` → optional `rrtmg_adapter`.
- Wired stored `theta_flux`, `qv_flux`, `tau_u`, and `tau_v` into MYNN bottom-boundary columns and applied bottom-level flux tendencies with the surface-layer sign convention.
- Changed `DailyPipelineConfig.radiation_cadence_steps` default from `999999` to `180`.

## Commands Run And Output

`taskset -c 0-3 pytest -q tests/test_m7_skill_fix_algorithmic.py`

```text
4 passed, 1 warning in 7.29s
```

`taskset -c 0-3 pytest -q tests/test_m7_skill_fix_algorithmic.py tests/test_m6_dummy_coupled.py`

```text
6 passed in 38.16s
```

`taskset -c 0-3 pytest -q tests/test_m7_restart_checkpoint_roundtrip.py tests/test_m6c_20260509_mu_regression.py tests/test_m6b6_coupled_step_parity.py tests/test_m6b_d2h_warmed_zero_v2.py -k 'not artifacts_present'`

```text
15 passed, 1 deselected in 53.03s
```

The full D2H artifact-presence test was also attempted and failed before the `-k` rerun because `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2/proof_warmed.nsys-rep` is absent in this checkout. The parsed JSON summary exists and records `d2h_inter_kernel=0`; `invariant_preservation.json` uses that committed parsed summary.

`taskset -c 0-3 python scripts/m7_daily_pipeline.py --hours 24 --output-dir /tmp/m7_skillfix_algorithmic_20260521 --proof-dir .agent/sprints/2026-05-27-m7-skill-fix-algorithmic --restart-at-hour 12`

```text
exit=2; PIPELINE_PARTIAL because --score was not requested.
restart_probe_status=PASS; wrfout_inventory_status=PASS; speedup_status=PASS;
radiation_cadence_steps=180; all_finite_check=true; theta=[200.0, 700.0];
wall_clock_total_s=700.8857639779962.
Superseded by r2 because hard clipping pinned theta_lower_30_max_k.
```

`taskset -c 0-3 python scripts/m7_daily_pipeline.py --hours 24 --output-dir /tmp/m7_skillfix_algorithmic_20260521_r2 --proof-dir .agent/sprints/2026-05-27-m7-skill-fix-algorithmic --restart-at-hour 12`

```text
exit=2; PIPELINE_PARTIAL because --score was not requested.
restart_probe_status=PASS; wrfout_inventory_status=PASS; speedup_status=PASS;
radiation_cadence_steps=180; all_finite_check=true;
theta=[200.0000762939453, 699.999755859375];
|u|max=50.158729553222656, |v|max=37.517173767089844, |w|max=0.2328642054893033;
wall_clock_total_s=708.3172624419967.
```

`taskset -c 0-3 python <post_fix_bounds generator>`

```json
{"status":"FAIL","theta_lower_30_max_variation_k":0.006744384765625,"output":".agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_bounds.json"}
```

`taskset -c 0-3 python scripts/m7_gpu_vs_cpu_skill_diff.py --gpu-root /tmp/m7_skillfix_algorithmic_20260521_r2 --output .agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json`

```json
{"status":"FAIL_SKILL_DIFF","common_valid_time_count":24,"station_count_scored":73,"output":".agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json"}
```

Post-processed skill verdict:

```json
{"contract_verdict":"improved but partial","metrics_improved":6,"metrics_worsened":3}
```

`taskset -c 0-3 python scripts/m7_cpu_per_domain_timing.py --pipeline-run .agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json --cpu-output .agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_cpu_timing.json --speedup-output .agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json`

```json
{"status":"PASS","selected_cpu_run_id":"20260521_18z_l3_24h_20260522T133443Z","d02_only_speedup":23.019785320190785}
```

`taskset -c 0-3 python <invariant_preservation generator>`

```json
{"status":"PASS","checks":{"20260521_multistep_parity_step2_bitwise":true,"20260521_multistep_parity_final":true,"b6_savepoint_parity":true,"d2h_inter_kernel":true,"restart_bitwise":true}}
```

`taskset -c 0-3 python scripts/validate_agentos.py`

```json
{"errors":[],"ok":true,"required_files_checked":31,"skills_checked":13}
```

## Proof Objects Produced

- AC2/AC5: `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_bounds.json` — `FAIL`; finite and wind/theta envelope pass, but `theta_lower_30_max_k` variation is only `0.006744384765625 K`.
- AC6: `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json` — `contract_verdict="improved but partial"`; 6/9 aggregate metrics improved, but all variables remain outside ±20%.
- AC7: `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` — PASS; d02-only speedup `23.019785320190785`.
- AC8: `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/invariant_preservation.json` — PASS.
- AC9: `tests/test_m7_skill_fix_algorithmic.py`.

## Risks

- AC2 remains failed. The lower-column state is no longer frozen in the mean/percentiles, but the maximum statistic is dominated by cells hitting the `[200,400]` guard envelope.
- AC6 remains failed. T2 worsened relative to the pre-fix baseline, while U10/V10 improved; remaining contributors likely include the already identified LU_INDEX/static-field issue and surface/land-state evolution gap.
- The D2H invariant proof depends on the parsed v2 JSON summary because the original `.nsys-rep` binary is absent from this checkout.

## Handoff

- objective: apply sprint A+B+C algorithmic fixes and honestly remeasure M7 skill.
- files changed: listed above; no governance, goal, reviewer, tester, manager, memory, dynamics, validation, IO, or checkpoint files modified.
- commands run: listed above with captured outputs.
- proof objects produced: post-fix bounds, skill diff, speedup, invariant preservation, pipeline JSONs, and focused tests.
- unresolved risks: AC2 and AC6 are not green; this is `SKILL_IMPROVED_PARTIAL`, not `SKILL_FIXED`.
- next decision needed: manager should decide whether to accept the partial wind improvement and dispatch the next defect sprint, likely static LU_INDEX/land-state refresh plus a guard-envelope redesign that avoids max-stat saturation without violating the theta bounds.
