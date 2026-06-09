# Worker Report

## Summary:

Built and executed the first strict full-domain d02 step-1 same-input
WRF-vs-JAX comparison.

Final verdict:
`STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_T`.

The comparison is the accepted one: CPU-WRF step-1
`post_after_all_rk_steps_pre_halo` truth versus JAX one-step
`_rk_scan_step_with_pre_halo_capture(...).pre_halo_state` from the same initial
`OperationalCarry`/`OperationalNamelist`. The forbidden initial-state-vs-WRF
post-step comparison was not used.

## Files Changed

- `proofs/v014/step1_same_input_truth.py`
- `proofs/v014/step1_same_input_truth.json`
- `proofs/v014/step1_same_input_truth.md`
- `proofs/v014/step1_same_input_truth_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-same-input-truth.md`

No production `src/gpuwrf/**` files were changed.

## Commands Run

- `python -m py_compile proofs/v014/step1_same_input_truth.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_same_input_truth.py`
- `python -m json.tool proofs/v014/step1_same_input_truth.json >/tmp/step1_same_input_truth.validated.json`
- `git diff -- src/gpuwrf`
- `tcsh ./compile em_real` in the disposable WRF tree
- `mpirun --oversubscribe -np 28 /mnt/data/wrf_gpu2/v014_step1_same_input_truth/run/wrf.exe`

## Proof Objects Produced

- `proofs/v014/step1_same_input_truth.json`
- `proofs/v014/step1_same_input_truth.md`
- `proofs/v014/step1_same_input_truth_wrf_patch.diff`
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`

## Result

The strict comparison executed. First divergent field in schema order is `T`.
Largest residual fields are base/mass related:

- `MUB`: max_abs `2635.640625`, RMSE `98.13000038547803`
- `PB`: max_abs `2627.3828125`, RMSE `47.826296821589736`
- `PHB`: max_abs `2237.9423828125`, RMSE `45.35253861292826`
- `P`: max_abs `1561.1123921205437`, RMSE `305.75054216524205`

## Unresolved Risks

The proof names the first divergent field and the dominant residual family, but
does not yet localize the responsible source operator. The dominant `MUB/PB/PHB`
pattern is consistent with the prior live-nest/base-state split evidence and
must be fixed or falsified before FP32, memory follow-ups, Switzerland, or TOST.

## Next Decision

Open a source/falsifier sprint for native live-nest child base-state
initialization. The sprint should make initial d02 base fields match CPU-WRF
post-initialization truth, then rerun this step-1 same-input comparison.
