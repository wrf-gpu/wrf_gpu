# Sprint Contract — M8.B: Savepoint Harness Scaffold + Entry-Point Inventory

**Sprint ID**: `2026-05-28-m8b-savepoint-scaffold`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m8b-savepoint-scaffold`
**Worktree**: `/tmp/wrf_gpu2_m8b`
**Wall-time**: 2-4 h
**GPU usage**: NONE
**Verifier**: Opus 4.7 (after worker reports DONE)

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥ 15-case seasonal ensemble; ≥ 10× speedup preserved.

## Objective

Produce a real `tests/savepoint/` harness scaffold (the blinded planner flagged this missing — and replace the placeholder) and an authoritative inventory of every operational entry script. Both unblock M9 and every subsequent milestone-close proof object.

## Required inputs

1. `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md`
2. `tests/test_m6b6_coupled_step_parity.py` (existing 100-step coupled parity test — the model for the harness)
3. `scripts/m6b6_coupled_step_compare.py` (existing comparator — extract its pattern)
4. `.agent/sprints/2026-05-27-testing-plan-execution-redo/savepoint_parity_deep*.json` (existing deep-savepoint proof structure)
5. `src/gpuwrf/runtime/operational_mode.py` (operational entry path the harness must hook into)
6. `scripts/` directory — every entry script that runs Canary forecasts (find with `find scripts -name "run_canary*" -o -name "operational*" -o -name "validate_*"`)

## Acceptance

### AC1 — `tests/savepoint/` directory created with:

- `tests/savepoint/__init__.py`
- `tests/savepoint/README.md` — explains harness, lists every saved-state group, names the operator that produced each saved state, lists the WRF Fortran routine each compares against
- `tests/savepoint/conftest.py` — pytest fixtures for loading WRF Fortran reference states + JAX state under test
- `tests/savepoint/test_dycore_100_steps.py` — wraps the existing 100-step coupled parity test in the new structure
- `tests/savepoint/test_dycore_1000_steps_PLACEHOLDER.py` — skeleton + xfail with explicit reason "M9 will produce reference states"
- `tests/savepoint/test_physics_couplers_PLACEHOLDER.py` — skeleton + xfail with explicit reason "M9 will produce reference states"
- `tests/savepoint/test_operational_variables_PLACEHOLDER.py` — skeleton + xfail with explicit reason "M9 will produce reference states"

All placeholder tests use proper `pytest.skip` or `pytest.xfail` (not silent pass). The skip/xfail reason must reference the specific milestone that resolves the placeholder.

### AC2 — `.agent/sprints/2026-05-28-m8b-savepoint-scaffold/entry_point_inventory.json`

Schema:
```json
{
  "inventory_version": "1.0",
  "scripts": [
    {
      "path": "scripts/run_canary_l2_24h.sh",
      "exists": true,
      "purpose": "<text>",
      "produces_outputs": ["<list>"],
      "required_inputs": ["<list>"],
      "blocker_if_missing": "<text or null>"
    },
    ...
  ],
  "missing_required": [ { "expected_path": "<spec>", "needed_for_milestone": "<M9|M19|...>" } ]
}
```

If `scripts/run_canary_*.sh` are entirely absent (which the blinded planner noted), create placeholder scripts under `scripts/run_canary_l2_24h.sh.PLACEHOLDER` that document the intended interface and exit non-zero with a clear "M19 implements this" message.

### AC3 — `.agent/sprints/2026-05-28-m8b-savepoint-scaffold/worker-report.md`

Standard format: objective, files changed, commands run (pinned), proof objects, unresolved risks, next-decision.

### AC4 — Pytest smoke

`taskset -c 0-3 pytest -q tests/savepoint/ --collect-only` returns 0 (collects all tests cleanly).
`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES (the existing 100-step parity is preserved; this is a regression smoke).

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: NONE.
3. **Files writable**: `tests/savepoint/**`, `scripts/run_canary_*.sh.PLACEHOLDER`, `.agent/sprints/2026-05-28-m8b-savepoint-scaffold/**`.
4. **Files NOT writable**: any existing `tests/test_m*.py`, `src/**`, `scripts/m6b6_*.py`, governance files, public repo.
5. **B6 100-step regression must still pass**. If the harness migration breaks the existing test, revert and report.
6. **No remote push.**
7. **Manager repo ONLY**.
8. **Auto-notify on exit**: dispatcher sends `tmux send-keys -t 1 "AGENT REPORT: m8b-savepoint-scaffold DONE exit=$?" Enter`.
9. **End with verdict**: `M8B_COMPLETE / M8B_PARTIAL` + one-line summary.
