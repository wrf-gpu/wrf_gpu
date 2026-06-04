# GPT Deadcode Cleanup Report - 2026-06-04

## Objective

Delete the orphaned legacy non-prep acoustic helper `_operational_acoustic_substep_core` and its dead tests for project task #41, without changing the production PREP-based operational forecast path.

## Outcome

Orphan confirmed: yes. No production `src/` caller exists. The daily-pipeline path calls `run_forecast_operational`, which enters `_rk_scan_step`, builds `small_step_prep_wrf`, and advances via `_acoustic_scan -> acoustic_substep_core` using `prep.c2a`.

Deleted:
- `gpuwrf.runtime.operational_mode._operational_acoustic_substep_core`
- `gpuwrf.runtime.operational_mode._carry_from_acoustic_core`
- `tests/test_m6b_operational_theta_fix.py`
- `tests/test_m6_acoustic_theta_fix.py`
- `tests/test_m6_rk_save_family_fix.py`
- `tests/unit/test_mu_persistence_two_substeps.py`

Adjusted historical diagnostics so no stale import remains:
- `scripts/m6_guard_disabled_debug.py`
- `scripts/m6b_real_ic_operational_compare.py`
- `scripts/f6_transaction_audit.py`

## Files Changed

- `src/gpuwrf/runtime/operational_mode.py`
- `scripts/m6_guard_disabled_debug.py`
- `scripts/m6b_real_ic_operational_compare.py`
- `scripts/f6_transaction_audit.py`
- `proofs/v090/deadcode_cleanup_report.json`
- `.agent/reviews/2026-06-04-gpt-deadcode-cleanup.md`

## Commands Run

- `taskset -c 0-3 rg -n "_operational_acoustic_substep_core" src scripts tests`
- `taskset -c 0-3 rg -n "_operational_acoustic_substep_core|_carry_from_acoustic_core" src scripts tests`
- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu ... python -m py_compile ...`
- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu ... python` import smoke
- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu ... python` `_rk_scan_step` PREP-path smoke
- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu ... python -m pytest tests/test_m6_guard_disabled_debug.py tests/test_m6b_shared_core_contract.py tests/test_m6b_fix_rk1_acoustic_loop.py -q`
- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu ... python -m pytest tests/ -q -k "operational" 2>&1 | tail -15`
- `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu ... timeout 90 python -m pytest tests/savepoint/test_dycore_100_steps.py -q`
- `taskset -c 0-3 git diff --check`

## Proof Objects

- `proofs/v090/deadcode_cleanup_report.json`

Key results:
- Post-delete grep for removed helper/constructor under `src scripts tests`: no matches.
- Import + py_compile: pass.
- PREP-path `_rk_scan_step` smoke: pass on CPU, finite theta/mu.
- Nearby affected checks: `17 passed in 4.74s`.
- Requested `-k operational` sweep: `4 failed, 28 passed, 4 skipped, 1167 deselected`; failures are pre-existing static-source policy tests (`jax.lax.cond` / `_m9_snapshot` already present in base HEAD), not introduced by this deletion.

## Unresolved Risks

- `tests/savepoint/test_dycore_100_steps.py` still exits `139` with a JAX-internal CPU-only segfault under the current JAX/Python environment. This is documented as a known env artifact and was not debugged.
- Historical M6 diagnostic scripts now fail explicitly if their removed legacy non-prep drilldown is invoked. Current diagnostics should use the production PREP-based path.
- Stale static tests outside task #41 remain and should be handled separately if the manager wants a fully green static sweep.

## Next Decision Needed

Decide whether to separately retire/update the remaining stale M6 static tests and historical diagnostics beyond this task's helper/test deletion scope.
