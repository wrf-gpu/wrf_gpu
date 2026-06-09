# Worker Report

Summary: added and proved a default-off private pre-halo capture hook in
`src/gpuwrf/runtime/operational_mode.py`. The hook exposes the final RK3
post-refresh state before RK halo exchange for proof code. No numerical model
fix was attempted.

Files changed:

- `src/gpuwrf/runtime/operational_mode.py`
- `proofs/v014/jax_pre_halo_capture.py`
- `proofs/v014/jax_pre_halo_capture.json`
- `proofs/v014/jax_pre_halo_capture.md`
- `.agent/reviews/2026-06-09-v014-pre-halo-capture-hook.md`
- `tests/test_v014_pre_halo_capture.py`

Proof summary: verdict `HOOK_GREEN_COMPARE_BLOCKED_NO_JAX_H10_PRESTEP_CARRY`.
The hook is green on a CPU fixture and normal RK return is exact when capture is
disabled. The h10 same-surface comparison remains blocked by missing JAX
pre-step carry/checkpoint for `d02` step 6000.

Commands reported by worker:

- `python -m py_compile src/gpuwrf/runtime/operational_mode.py proofs/v014/jax_pre_halo_capture.py tests/test_v014_pre_halo_capture.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_pre_halo_capture.py`
- `python -m json.tool proofs/v014/jax_pre_halo_capture.json >/tmp/jax_pre_halo_capture.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_v014_pre_halo_capture.py`
- focused existing tests including `tests/test_m6_guard_disabled_debug.py`

Next recommendation: build or locate the h10 JAX pre-step `OperationalCarry`,
then rerun the hook against Boole's WRF green target.
