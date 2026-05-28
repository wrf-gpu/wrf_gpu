# Worker Report - M8.B Savepoint Harness Scaffold

## Objective

Create a real `tests/savepoint/` harness scaffold that preserves the existing 100-step M6B6 coupled-step parity guard, inventory current Canary forecast entry points, add required missing-entry placeholders, and keep all edits inside the sprint contract's writable scope.

## Files Changed

- `tests/savepoint/__init__.py`
- `tests/savepoint/README.md`
- `tests/savepoint/conftest.py`
- `tests/savepoint/test_dycore_100_steps.py`
- `tests/savepoint/test_dycore_1000_steps_PLACEHOLDER.py`
- `tests/savepoint/test_physics_couplers_PLACEHOLDER.py`
- `tests/savepoint/test_operational_variables_PLACEHOLDER.py`
- `scripts/run_canary_l2_24h.sh.PLACEHOLDER`
- `scripts/run_canary_l3_24h.sh.PLACEHOLDER`
- `.agent/sprints/2026-05-28-m8b-savepoint-scaffold/entry_point_inventory.json`
- `.agent/sprints/2026-05-28-m8b-savepoint-scaffold/worker-report.md`

## Commands Run

- `cat PROJECT_CONSTITUTION.md` - read required governance input.
- `cat AGENTS.md` - read required agent rules.
- `cat .agent/sprints/2026-05-28-m8b-savepoint-scaffold/sprint-contract.md` - read current sprint contract.
- `cat .agent/skills/building-wrf-oracles/SKILL.md` - read local WRF-oracle skill.
- `cat .agent/skills/validating-physics/SKILL.md` - read local validation skill.
- `cat .agent/skills/reporting-to-human/SKILL.md` - read local reporting skill.
- `cat .agent/decisions/PROJECT-RESET-PLAN-FINAL.md` - read required reset-plan input.
- `find scripts \( -name 'run_canary*' -o -name 'operational*' -o -name 'validate_*' \) -print | sort` - found only `validate_*` scripts; no `run_canary*` or `operational*` shell entry points.
- `python -m json.tool .agent/sprints/2026-05-28-m8b-savepoint-scaffold/entry_point_inventory.json` - PASS.
- `bash -n scripts/run_canary_l2_24h.sh.PLACEHOLDER` - PASS.
- `bash -n scripts/run_canary_l3_24h.sh.PLACEHOLDER` - PASS.
- `taskset -c 0-3 pytest -q tests/savepoint/ --collect-only` - PASS, 4 tests collected.
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` - PASS, `1 passed in 429.43s (0:07:09)`.
- `git add tests/savepoint scripts/run_canary_l2_24h.sh.PLACEHOLDER scripts/run_canary_l3_24h.sh.PLACEHOLDER .agent/sprints/2026-05-28-m8b-savepoint-scaffold/entry_point_inventory.json .agent/sprints/2026-05-28-m8b-savepoint-scaffold/worker-report.md` - FAIL, git metadata is outside the writable sandbox at `/home/enric/src/wrf_gpu2/.git/worktrees/wrf_gpu2_m8b/index.lock`.
- `tmux send-keys -t 1 "AGENT REPORT: m8b-savepoint-scaffold DONE exit=1" Enter` - FAIL, tmux socket access is not permitted in this sandbox.

## Proof Objects Produced

- `tests/savepoint/` scaffold with one real 100-step coupled-step parity wrapper and three M9 xfail placeholders.
- `.agent/sprints/2026-05-28-m8b-savepoint-scaffold/entry_point_inventory.json` with current operational/validation entry inventory and missing required `run_canary_*` entries.
- `scripts/run_canary_l2_24h.sh.PLACEHOLDER` and `scripts/run_canary_l3_24h.sh.PLACEHOLDER`, both nonzero placeholders with explicit `M19 implements this` messages.
- AC4 pytest proof: collect-only exit 0 and final 100-step wrapper exit 0.

## Unresolved Risks

- M9 still must produce operational reference states for 1000-step dycore, physics couplers, and operational variables; placeholders are explicit xfails only.
- Real `scripts/run_canary_l2_24h.sh` and `scripts/run_canary_l3_24h.sh` remain absent; placeholders only document the intended M19 interface.
- The real 100-step parity smoke is CPU-only but slow, about seven minutes on this worktree.
- Commit was blocked by sandbox permissions: this worktree's Git index/refs live under `/home/enric/src/wrf_gpu2/.git`, which is read-only in the current session.
- Auto-notify was attempted but blocked by sandbox tmux socket permissions.

## Next Decision

M9 should generate and register the missing reference states, then replace the three placeholder tests without weakening the 100-step guard. M19 should decide the final stable shell interface for `run_canary_l2_24h.sh` and `run_canary_l3_24h.sh`.

## Verdict

M8B_PARTIAL - savepoint scaffold, entry-point inventory, placeholders, and required CPU-pinned smoke proofs are complete, but `git add`/commit is blocked by read-only Git metadata outside the writable sandbox.
