# Tester Report

Decision: accepted as a fail-closed proof, not as a dynamics parity result.

## Validation

Manager-side validation completed:

- `python -m py_compile proofs/v014/full_domain_source_truth.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/full_domain_source_truth.py`
- `python -m json.tool proofs/v014/full_domain_source_truth.json >/tmp/full_domain_source_truth.manager.validated.json`
- `python -m py_compile proofs/v014/same_input_single_rk_parity_wrapped.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_wrapped.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity_wrapped.json >/tmp/same_input_single_rk_parity_wrapped.manager.validated.json`
- `git diff -- src/gpuwrf`
- `git diff --check -- targeted sprint files`

## Result

Both JSON proof objects validate. The wrapper proof reproduces
`FULL_DOMAIN_WRAPPER_BLOCKED_TRUTH_SURFACE_PATCH_ONLY_AND_CARRY_LEAVES`.
No production source diff exists.

## Acceptance Notes

The sprint contract allowed a precise blocked verdict if strict execution could
not run. The result names the blocker without emitting a weak comparison.

## Residual Risk

The h10/step-6000 path has now produced another blocked result. Per management
review, the next proof should bisect from early shared-`wrfinput` steps.
