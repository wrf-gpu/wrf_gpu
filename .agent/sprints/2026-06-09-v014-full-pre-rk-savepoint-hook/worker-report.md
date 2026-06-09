# Worker Report

## Summary:

The worker built and ran a CPU-WRF full pre-RK savepoint hook for `d02` step
`6000`, then emitted a proof-only JAX wrapper verdict. The WRF hook succeeded
and produced full native step-entry state, but the strict same-input JAX
comparison is still blocked because current-step WRF `*_tendf` and `*_save`
source leaves do not exist at that exact step-entry boundary.

Final verdict:
`FULL_PRE_RK_JAX_LOADER_BLOCKED_RK_FIXED_SOURCE_BOUNDARY`.

## Files Changed

- `proofs/v014/full_pre_rk_savepoint_hook.py`
- `proofs/v014/full_pre_rk_savepoint_hook.json`
- `proofs/v014/full_pre_rk_savepoint_hook.md`
- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_full.py`
- `proofs/v014/same_input_single_rk_parity_full.json`
- `proofs/v014/same_input_single_rk_parity_full.md`
- `.agent/reviews/2026-06-09-v014-full-pre-rk-savepoint-hook.md`

No production `src/gpuwrf/**` files were changed.

## Commands Run

- CPU-WRF scratch build/run commands recorded in
  `proofs/v014/full_pre_rk_savepoint_hook.json`.
- `python -m py_compile proofs/v014/full_pre_rk_savepoint_hook.py`
- `python -m json.tool proofs/v014/full_pre_rk_savepoint_hook.json >/tmp/full_pre_rk_savepoint_hook.validated.json`
- `python -m py_compile proofs/v014/same_input_single_rk_parity_full.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_full.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity_full.json >/tmp/same_input_single_rk_parity_full.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects Produced

- `proofs/v014/full_pre_rk_savepoint_hook.json`
- `proofs/v014/full_pre_rk_savepoint_hook.md`
- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_full.json`
- `proofs/v014/same_input_single_rk_parity_full.md`
- `.agent/reviews/2026-06-09-v014-full-pre-rk-savepoint-hook.md`

## Findings

The WRF run reached `2026-05-02_04:00:00` and completed successfully. The hook
wrote two tile files with full vertical state:

- `MASS_FULL`: `12716`
- `U_FULL`: `13464`
- `V_FULL`: `13464`
- `WPH_FULL`: `13005`
- `MOIST_FULL`: `76296`
- `SCALAR_FULL`: `38148`

Duplicate tile overlap max delta is `0.0`. The patch leaves one conservative
mass score cell after an 8-cell halo, so width is not the primary blocker.

The strict one-step comparison is blocked because `ru_tendf`, `rv_tendf`,
`rw_tendf`, `ph_tendf`, `t_tendf`, `mu_tendf`, `h_diabatic`, `u_save`, `v_save`,
`w_save`, `ph_save`, `t_save`, `moist_old`, and `scalar_old` are unavailable at
the exact step-entry hook.

## Unresolved Risks

- The next source boundary must be placed after WRF produces source/save-family
  leaves and before any state-changing dynamics update that would change the
  one-step initial state.
- The scored patch remains narrow. If the next boundary runs but one score cell
  is too weak for attribution, widen the hook.
- No model source defect is localized yet.

## Next Decision Needed

Open a source-boundary sprint that emits current-step WRF
`DryPhysicsTendencies`/save-family leaves at the first boundary where they exist
without changing the same-input initial state. Then rerun the same-input
one-step JAX wrapper.
