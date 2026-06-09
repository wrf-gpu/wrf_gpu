# Tester Report

Decision: accepted as a validated blocked-boundary proof.

## Validation

Manager-side validation reran:

- `python -m py_compile proofs/v014/full_pre_rk_savepoint_hook.py proofs/v014/same_input_single_rk_parity_full.py`
- `python -m json.tool proofs/v014/full_pre_rk_savepoint_hook.json >/tmp/full_pre_rk_savepoint_hook.manager.validated.json`
- `python -m json.tool proofs/v014/same_input_single_rk_parity_full.json >/tmp/same_input_single_rk_parity_full.manager.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_full.py`
- `git diff -- src/gpuwrf`
- WRF patch forward dry-run against a temporary copy of
  `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF/dyn_em/solve_em.F`
- `git diff --check` on the targeted proof/review files

## Result

All manager-side gates passed. The final proof reproducibly emits
`FULL_PRE_RK_JAX_LOADER_BLOCKED_RK_FIXED_SOURCE_BOUNDARY`. Production
`src/gpuwrf/**` diff is empty.

## Acceptance Notes

The proof should not be treated as a dynamics mismatch. It is an instrumentation
boundary result: full pre-RK native state is available, but strict same-input RK
execution needs WRF source/save leaves from a later, carefully chosen boundary.

## Residual Risk

The next hook boundary is delicate. If placed after a state-changing update, it
will no longer be a strict same-input proof.
