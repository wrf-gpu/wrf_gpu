# Tester Report

Decision: accepted as a validated blocked-proof sprint, not as a grid-parity or
dycore-fix result.

## Validation

Manager-side validation completed:

- `python -m py_compile proofs/v014/source_save_boundary_hook.py proofs/v014/same_input_single_rk_parity_sources.py`
- `python -m json.tool proofs/v014/source_save_boundary_hook.json >/tmp/source_save_boundary_hook.manager.validated.json`
- `python -m json.tool proofs/v014/same_input_single_rk_parity_sources.json >/tmp/same_input_single_rk_parity_sources.manager.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_sources.py`
- `git diff -- src/gpuwrf`

## Result

Both JSON proof objects validate. The proof script reproduces the verdict
`SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER`.
The production-source diff is empty.

The WRF source/save boundary itself is accepted. It is positioned before the
first dry/acoustic mutation, dry source/save leaves are present, and the native
dry state is exact on overlap versus the previous full pre-RK savepoint.

## Acceptance Notes

The sprint contract allowed a precise blocked verdict if strict execution could
not run. The worker met that fallback by naming the exact remaining blockers:
proof-only JAX wrapper, full-domain same-boundary carry/boundary surface,
full-domain/full-vertical truth, and `scalar_old`/old-field handling.

## Residual Risk

The next proof must avoid a weak comparison. Do not mix the existing JAX
step5999 checkpoint with WRF-emitted source leaves, and do not use the current
17x17 patch as a final full-grid parity surface.
