# Review: V0.14 Full Pre-RK Savepoint Hook

verdict: `FULL_PRE_RK_JAX_LOADER_BLOCKED_RK_FIXED_SOURCE_BOUNDARY`.

objective: create a CPU-WRF full pre-RK native-state savepoint at d02 step 6000 and run, or precisely block, the strict same-input one-step JAX comparison.

files changed:
- `proofs/v014/full_pre_rk_savepoint_hook.py`
- `proofs/v014/full_pre_rk_savepoint_hook.json`
- `proofs/v014/full_pre_rk_savepoint_hook.md`
- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_full.py`
- `proofs/v014/same_input_single_rk_parity_full.json`
- `proofs/v014/same_input_single_rk_parity_full.md`
- `.agent/reviews/2026-06-09-v014-full-pre-rk-savepoint-hook.md`

commands run:
- `python -m py_compile proofs/v014/full_pre_rk_savepoint_hook.py`
- `python -m json.tool proofs/v014/full_pre_rk_savepoint_hook.json >/tmp/full_pre_rk_savepoint_hook.validated.json`
- `python -m py_compile proofs/v014/same_input_single_rk_parity_full.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_full.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity_full.json >/tmp/same_input_single_rk_parity_full.validated.json`
- `git diff -- src/gpuwrf`
- CPU WRF scratch build/run commands are recorded in `proofs/v014/full_pre_rk_savepoint_hook.json`.

proof objects produced:
- `proofs/v014/full_pre_rk_savepoint_hook.json`
- `proofs/v014/full_pre_rk_savepoint_hook.md`
- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_full.json`
- `proofs/v014/same_input_single_rk_parity_full.md`

unresolved risks:
- The full native state exists only over the narrow target patch; it leaves one conservative mass score cell after an 8-cell halo.
- The strict comparison is blocked because current-step WRF source/save-family leaves are not available at the exact step-entry hook.
- No production `src/gpuwrf/**` files were edited, so no JAX wrapper source API was added.

next decision needed: Add a second accepted WRF source boundary, or move the proof boundary, so the same file set contains current-step DryPhysicsTendencies/save-family leaves from WRF. Then construct OperationalCarry/DryPhysicsTendencies and call _rk_scan_step_with_pre_halo_capture on CPU.
