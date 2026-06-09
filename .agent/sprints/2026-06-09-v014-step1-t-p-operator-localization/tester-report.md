# Tester Report

## Decision:

Pass. The proof is reproducible under CPU-only JAX, validates as JSON, and
leaves production `src/gpuwrf/**` unchanged.

## Manager Re-Run Commands

- `python -m py_compile proofs/v014/step1_t_p_operator_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_t_p_operator_localization.py`
- `python -m json.tool proofs/v014/step1_t_p_operator_localization.json >/tmp/step1_t_p_operator_localization.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- Python compilation passed.
- CPU proof rerun reproduced verdict
  `STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`.
- JSON validation passed.
- `git diff -- src/gpuwrf` was empty.
- JSON records `cpu_only=true`, `gpu_used=false`,
  `production_src_edits=false`, `weak_comparison_avoided=true`, and no TOST,
  Switzerland, FP32, or memory source work.

## Coverage

The proof parses 168 WRF raw substage truth files from the scratch run and
compares full d02 arrays at the emitted early RK surfaces:

- `after_rk_addtend_before_small_step_prep`
- `after_small_step_prep_calc_p_rho`

It also reruns the final accepted Step-1 post-RK/pre-halo comparison.

## Residual Risk

The proof localizes a boundary but does not fix it. The next gate must compare
WRF `first_rk_step_part1/part2` outputs directly against JAX
`_physics_step_forcing` state/tendency/carry output.
