# Worker Report - Testing Plan Execution

Decision: EXECUTION_PARTIAL

Summary: Implemented the sprint-owned publication-test execution scaffolding and produced proof objects for all 10 HIGH-priority items. The execution is partial, not green: the actual GPU preflight timed out (`nvidia-smi` return code 124), so the idealized, conservation, and stability GPU integrations were not run. Existing committed guardrail artifacts show B6 10-step parity, restart bitwise, two-run repeatability, and D2H/H2D=0 remain PASS, but the revised plan's deeper publication gates still fail or remain blocked.

## Objective

Execute HIGH-priority publication tests honestly, reusing existing diagnostics/artifacts where available, adding the required idealized IC builders, and writing per-test proof JSONs under this sprint folder.

## Files Changed

- `src/gpuwrf/fixtures/idealized_cases/**`
- `scripts/pubtest_*.py`
- `tests/test_pubtest_*.py`
- `.agent/sprints/2026-05-27-testing-plan-execution/**`

## Commands Run and Output

- `taskset -c 0-3 python scripts/pubtest_execute_high_priority.py --proof-dir .agent/sprints/2026-05-27-testing-plan-execution --gpu-probe-timeout-s 5`
  - exit `0`
  - stdout: `status=EXECUTION_PARTIAL`; GPU preflight `available=false`, `returncode=124`, `timed_out=true`; wrote aggregate report and all proof objects.
  - stderr: empty.
- `taskset -c 0-3 python -m compileall -q src/gpuwrf/fixtures/idealized_cases scripts/pubtest_*.py`
  - exit `0`; stdout/stderr empty.
- `taskset -c 0-3 python -m pytest tests/test_pubtest_idealized_cases.py tests/test_pubtest_execution.py -q`
  - exit `0`; stdout: `5 passed in 0.26s`; stderr empty.
- `taskset -c 0-3 python <json proof loader>`
  - exit `0`; stdout: `loaded_json_count=18`; stderr empty.
- `taskset -c 0-3 python scripts/validate_agentos.py`
  - exit `0`; stdout: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`; stderr empty.
- `taskset -c 0-3 python <static guardrail artifact check>`
  - exit `0`; stdout: `static_guardrails_pass=true`, `m6b6_depths=column:10,golden:10,patch16:10`, `restart_fields=47`; stderr empty.
- Attempted `taskset -c 0-3 python -m pytest tests/test_m7_restart_checkpoint_roundtrip.py tests/test_m6b_operational_no_h2d.py -q`
  - stopped after >45 s with no output because JAX/GPU probing was hanging in the same pattern as `nvidia-smi`; exit recorded as `124`; stdout/stderr empty. Static artifact guardrail check above was used instead.

Full stdout/stderr/exit captures are in `.agent/sprints/2026-05-27-testing-plan-execution/command_outputs/`.

## Proof Objects Produced

- `idealized_warmbubble.json` / `idealized_warmbubble_summary.md`: BLOCKED, IC builder complete, no GPU/WRF integration.
- `idealized_density_current.json`: BLOCKED, Straka IC builder complete, no 900 s GPU integration.
- `idealized_mountain_wave.json` / `idealized_mountain_wave_summary.md`: BLOCKED, Schaer IC builder complete, no 5 h GPU integration.
- `conservation_mass_24h.json`: BLOCKED, reused `diagnostic_conservation_tracker.py` in smoke mode, no 24 h GPU state series.
- `conservation_energy_24h.json`: BLOCKED, no CPU envelope or 24 h GPU energy series.
- `stability_cfl_sweep.json`: BLOCKED, no GPU stability runner execution.
- `stability_acoustic_substep.json`: BLOCKED, no GPU acoustic sweep execution.
- `determinism_repeat.json`: FAIL, existing two-run bitwise repeatability PASS but required three-run gate unmet.
- `savepoint_parity_deep.json`: FAIL, existing B6 10-step parity PASS but required 100/1000/10000-depth checks unmet.
- `canary_case_manifest.json` and `canary_multiday_skill.json`: FAIL, 14-day CPU window exists but only one GPU day is available and T2/U10/V10 fail the +/-20% skill gate.
- `aggregate_report.md` / `aggregate_report.json`: aggregate PASS/FAIL/BLOCKED table and publication framing note.

## Risks

- Heavy GPU execution could not start because device probing timed out. No GPU-hour budget was consumed.
- WRF idealized compiles were not run; only WRF source/provenance was pinned in the proof JSONs.
- The added builders are IC generators, not forecast solvers. They should be treated as prerequisites for a later runnable idealized harness.
- Current evidence supports a partial/prototype publication framing only.

## Handoff

objective: HIGH-priority testing-plan execution, partial due GPU/preflight blocker.
files changed: sprint-owned proof/report files, new `pubtest_*` wrappers, new idealized IC builders, new focused tests.
commands run: listed above with captured outputs under `command_outputs/`.
proof objects produced: all AC1-AC10 JSONs plus aggregate report and worker report.
unresolved risks: GPU unavailable, no WRF idealized reference runs, no 14-day GPU corpus, no deep savepoint run.
next decision needed: manager should decide whether to rerun this sprint on a healthy GPU session or accept these proof objects as evidence for an honest partial-results paper.
