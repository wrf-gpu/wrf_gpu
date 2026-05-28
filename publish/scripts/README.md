# Publication Test Scripts

These are the sprint-owned `pubtest_*.py` orchestrators used to produce
`.agent/sprints/2026-05-27-testing-plan-execution-redo/` proof objects.

## Re-run

From the repository root:

```bash
taskset -c 0-3 python scripts/pubtest_execute_high_priority.py \
  --proof-dir .agent/sprints/2026-05-27-testing-plan-execution-redo \
  --execution-root /tmp/pubtest_redo \
  --gpu-probe-timeout-s 5 \
  --run-savepoint-deep
```

## Proof Map

| Script | Purpose | Primary proof object |
|---|---|---|
| `pubtest_execute_high_priority.py` | Runs/aggregates all HIGH-priority publication tests. | `aggregate_report.json` |
| `pubtest_prepare_wrf_ideal.py` | Prepares WRF idealized-case provenance payloads. | `inputs/*_ic_summary.json` |
| `pubtest_run_wrf_reference.py` | WRF-reference wrapper; currently records missing runnable reference when not executed. | idealized proof JSONs |
| `pubtest_run_gpu_ideal.py` | GPU-ideal wrapper; currently records missing reviewed idealized runner. | idealized proof JSONs |
| `pubtest_compare_ideal.py` | Shared idealized comparison entry point. | `idealized_warmbubble.json` |
| `pubtest_prepare_density_current.py` | Builds Straka density-current IC summary. | `inputs/density_current_ic_summary.json` |
| `pubtest_compare_density_current.py` | Density-current comparison wrapper. | `idealized_density_current.json` |
| `pubtest_prepare_mountain_wave.py` | Builds Schaer mountain-wave IC summary. | `inputs/schaer_mountain_wave_ic_summary.json` |
| `pubtest_compare_mountain_wave.py` | Mountain-wave comparison wrapper. | `idealized_mountain_wave.json` |
| `pubtest_mass_budget.py` | Conservation mass wrapper. | `conservation_mass_24h.json` |
| `pubtest_energy_budget.py` | Conservation energy wrapper. | `conservation_energy_24h.json` |
| `pubtest_stability_cfl_sweep.py` | CFL sweep wrapper. | `stability_cfl_sweep.json` |
| `pubtest_acoustic_substep_sweep.py` | Acoustic-substep sweep wrapper. | `stability_acoustic_substep.json` |
| `pubtest_determinism_repeat.py` | Three-run determinism wrapper. | `determinism_repeat.json` |
| `pubtest_savepoint_parity_deep.py` | Savepoint-depth parity wrapper. | `savepoint_parity_deep.json` |
| `pubtest_select_canary_cases.py` | Canary CPU-history selector. | `canary_case_manifest.json` |
| `pubtest_aggregate_skill.py` | Canary skill aggregation wrapper. | `canary_multiday_skill.json` |
| `pubtest_first_error_growth.py` | First-error-growth placeholder wrapper. | `canary_first_error_growth.json` |
| `pubtest_common.py` | Shared helper library for proof JSONs. | N/A |
