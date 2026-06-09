# Worker Report

## Summary:

Reran the strict d02 step-1 same-input WRF-vs-JAX comparison with the existing
native live-nest child base initialization semantics wired into the CPU-only
proof loader.

Final verdict:
`STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_T`.

The comparison executed against the existing CPU-WRF d02 step-1
`post_after_all_rk_steps_pre_halo` truth. The proof avoided the forbidden raw
wrfinput headline and did not compare WRF post-step truth to a JAX initial
state.

## Files Changed

- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_live_nest_init_rerun.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-init-rerun.md`

No production `src/gpuwrf/**` files were changed.

## Commands Run

- `python -m py_compile proofs/v014/step1_live_nest_init_rerun.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py`
- `python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects Produced

- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_live_nest_init_rerun.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-init-rerun.md`

The CPU-WRF truth npz was reused, not rebuilt:
`/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`.

## Result

The live-nest init rerun closes the dominant base residual family:

- `MUB`: max_abs `0.05002361937658861`, RMSE `0.008025019829604947`
- `PB`: max_abs `0.05357326504599769`, RMSE `0.004296943965085442`
- `PHB`: max_abs `0.10811684231157415`, RMSE `0.02459295000326211`

The strict comparison still diverges. First divergent schema field is `T`.
Largest remaining residuals are:

- `P`: max_abs `1561.2503728885986`, RMSE `305.9413510899027`
- `PH`: max_abs `77.6192303625287`, RMSE `19.320745387744648`
- `MU`: max_abs `36.543234083976586`, RMSE `1.6018856311784238`
- `T`: max_abs `5.483713205511606`, RMSE `1.9175729083315645`

## Unresolved Risks

The proof-local CPU loader mirrors the production live-nest initialization
semantics but still bypasses `build_replay_case` because `State.zeros` is
GPU-only. That is acceptable for this proof but should not be mistaken for a
production-path integration test.

The remaining residuals identify the next field/operator target but do not yet
name the exact dycore, source, or coupling operator.

## Next Decision

Open an operator-localization sprint for d02 step 1. It should localize the
first `T` divergence and the dominant `P/PH/MU` residuals across substage
boundaries before any TOST, Switzerland, FP32, or memory follow-up work resumes.
