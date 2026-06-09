# Tester Report

## Decision:

Pass. The proof is reproducible under CPU-only JAX, validates as JSON, and
leaves production `src/gpuwrf/**` unchanged.

## Manager Re-Run Commands

- `python -m py_compile proofs/v014/step1_rk1_source_boundary.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_rk1_source_boundary.py`
- `python -m json.tool proofs/v014/step1_rk1_source_boundary.json >/tmp/step1_rk1_source_boundary.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- Python compilation passed.
- CPU proof rerun reproduced verdict
  `STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.
- JSON validation passed.
- `git diff -- src/gpuwrf` was empty.
- JSON records `cpu_only=true`, `gpu_used=false`,
  `production_src_edits=false`, `weak_comparison_avoided=true`, no
  JAX-vs-JAX conclusion, no one-cell proof, and no TOST, Switzerland, FP32, or
  memory source work.

## Coverage

The proof parses WRF RK1 source-boundary truth files from scratch WRF and
compares full d02 arrays for the relevant Step-1 boundaries. It specifically
checks WRF `after_first_rk_step_part1` against both JAX operational carry and
`_physics_step_forcing.state`.

## Residual Risk

The proof localizes a boundary but does not fix it. The next gate must split WRF
`first_rk_step_part1` internals against JAX physics adapter state/tendency
outputs before selecting a source representation or applying a production fix.
