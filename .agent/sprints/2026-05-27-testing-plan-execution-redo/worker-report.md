# Worker Report - Testing Plan Execution RE-DO

Decision: EXECUTION_PARTIAL

Summary: Re-ran the HIGH-priority publication-test execution path on the healthy RTX 5090. The redo consumed 1.226423 recorded GPU-hours and produced non-BLOCKED proof JSONs for all 10 HIGH items. Real GPU execution covered the runnable Canary d02 subset, Canary surrogate stability sweeps, and three-run determinism. The result is partial because the repo still lacks reviewed idealized GPU forecast runners, only three complete 24h distinct Canary d02 history days are locally runnable, conservation lacks the required closed-domain / CPU-envelope evidence, and savepoint parity reached 100 steps but not 1000/10000.

## Objective

Re-execute `.agent/sprints/2026-05-27-testing-plan-critique/test_plan_revised.md` HIGH-priority items using the existing pubtest infrastructure, replacing BLOCKED placeholders with honest PASS / FAIL / SKIP evidence under this redo sprint.

## Files Changed

- `scripts/pubtest_common.py`
- `scripts/pubtest_execute_high_priority.py`
- `scripts/pubtest_determinism_repeat.py`
- `tests/test_pubtest_execution.py`
- `publish/scripts/**`
- `.agent/sprints/2026-05-27-testing-plan-execution-redo/**`

## Commands Run and Output

- `taskset -c 0-3 python scripts/pubtest_execute_high_priority.py --proof-dir .agent/sprints/2026-05-27-testing-plan-execution-redo --execution-root /tmp/pubtest_redo --gpu-probe-timeout-s 5 --run-savepoint-deep`
  - exit `0`; stdout: `status=EXECUTION_PARTIAL`, `total_gpu_hours_used=1.226422552104446`, GPU preflight PASS on RTX 5090; stderr: XLA slow-compile warnings only. Full capture: `command_outputs/pubtest_execute_high_priority.*`.
- Determinism proof rerun after fixing proof logic: exit `0`; final proof `PASS_THREE_RUN_BITWISE`, max delta `0.0`.
- Savepoint proof compact rerun: exit `0`; retained JSON proof, not HDF5 intermediates.
- `python -m pytest -q`
  - exit `1`; output: `747 passed, 19 skipped, 47 failed in 1009.12s`. Failures are broad historical/data/toolchain tests outside this sprint's owned files. Full capture: `command_outputs/pytest_full.*`.
- Focused validation: `pytest tests/test_pubtest_idealized_cases.py tests/test_pubtest_execution.py -q`
  - exit `0`; final output: `5 passed in 1.39s`.
- Guardrail pytest: `pytest tests/test_m7_restart_checkpoint_roundtrip.py tests/test_m6b_operational_no_h2d.py -q`
  - exit `0`; output: `4 passed in 1.96s`.
- `python scripts/validate_agentos.py`
  - exit `0`; output: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`.
- JSON proof load check
  - exit `0`; output: `loaded_json_count=96`.
- Static guardrail artifact check
  - exit `0`; output: `static_guardrails_pass=True`, step-2 bitwise `True`, B6 depths `column:10,golden:10,patch16:10`, restart bitwise `True`, D2H inter-kernel `0`, H2D inside loop `0`.
- `python -m compileall -q scripts/pubtest_*.py tests/test_pubtest_*.py publish/scripts`
  - exit `0`; stdout/stderr empty.
- `python scripts/repo_status_snapshot.py`
  - exit `0`; output records only allowed dirty paths after restoring pytest-generated out-of-scope artifacts.

## Proof Objects Produced

- `aggregate_report.json` / `aggregate_report.md`: total GPU-hours `1.226423`; aggregate table.
- `idealized_warmbubble.json`: `SKIP_NO_IDEALIZED_GPU_FORECAST_RUNNER`.
- `idealized_density_current.json`: `SKIP_NO_DENSITY_CURRENT_GPU_FORECAST_RUNNER`.
- `idealized_mountain_wave.json`: `SKIP_NO_MOUNTAIN_WAVE_GPU_FORECAST_RUNNER`.
- `conservation_mass_24h.json`: `FAIL_MISSING_CLOSED_DOMAIN_AND_BOUNDARY_FLUX_CORRECTION`.
- `conservation_energy_24h.json`: `FAIL_MISSING_CPU_ENVELOPE`.
- `stability_cfl_sweep.json`: `SKIP_NO_WARMBUBBLE_GPU_RUNNER`, with real Canary surrogate runs.
- `stability_acoustic_substep.json`: `SKIP_NO_DENSITY_CURRENT_GPU_RUNNER`, with real Canary surrogate runs.
- `determinism_repeat.json`: `PASS_THREE_RUN_BITWISE`, three independent 1h runs, max delta `0.0`.
- `savepoint_parity_deep.json`: `FAIL_INSUFFICIENT_SAVEPOINT_DEPTH`, 100-step column parity passed; 1000/10000 not run.
- `canary_multiday_skill.json`: `FAIL_FIVE_DAY_OR_SKILL_GATE`, three complete 24h days plus one partial-history day executed; five complete distinct days unavailable locally.
- Per-run proofs under `canary_runs/`, `determinism_runs/`, `stability_cfl_runs/`, and `stability_acoustic_runs/`.
- Publication script staging under `publish/scripts/`.

## Risks

- The full pytest suite is not green on this branch due pre-existing unrelated failures; focused sprint validations and guardrails are green.
- The local `/mnt/data/canairy_meteo/runs/wrf_l3/` inventory did not contain five complete distinct 24h d02 history days, so AC10 remains a real FAIL.
- Idealized publication tests remain skipped until a reviewed GPU idealized forecast runner exists.
- HDF5 savepoint intermediates were generated but not committed; the retained proof is the JSON comparison output.

## Handoff

objective: HIGH-priority testing-plan execution redo with real GPU evidence.

files changed: listed above; no model core, governance, reviewer, tester, manager-closeout, or memory-patch files touched.

commands run: listed above; stdout/stderr/exit captures are in `command_outputs/`.

proof objects produced: all AC1-AC10 JSONs plus aggregate report, per-run subproofs, command captures, and `publish/scripts/` staging.

unresolved risks: five-day Canary gate failed on available local complete history; idealized/conservation/deep-savepoint gates remain incomplete by evidence, not by GPU blockage.

next decision needed: reviewer/manager should decide whether the paper message accepts `EXECUTION_PARTIAL` with these explicit FAIL/SKIP items, or whether to dispatch implementation sprints for idealized runners and additional complete Canary history.
