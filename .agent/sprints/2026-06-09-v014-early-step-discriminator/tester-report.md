# Tester Report

Decision: accepted as a fail-closed proof, not as a dynamics parity result.

## Validation

Worker validation passed:

- `python -m py_compile proofs/v014/early_step_discriminator.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/early_step_discriminator.py`
- `python -m json.tool proofs/v014/early_step_discriminator.json >/tmp/early_step_discriminator.validated.json`
- `git diff -- src/gpuwrf`

Manager validation additionally passed:

- `python -m json.tool proofs/v014/early_step_discriminator.json >/tmp/early_step_discriminator.manager.validated.json`
- `git diff -- src/gpuwrf`

## Result

The JSON validates and records the consolidated blocker verdict:
`EARLY_STEP_DISCRIMINATOR_BLOCKED_CPU_REALCASE_LOADER_GPU_ONLY_NO_CANDIDATE_WRF_PREHALO_TRUTH_NO_SAME_INPUT_CARRY_CONTRACT`.

The proof explicitly covers candidate steps `1`, `60`, `600`, `3000`, and
`5999`. `strict_same_input_comparison_run` is `false`; the proof avoids weak
WRF-output, JAX-vs-JAX, one-cell, and mixed-source comparisons.

## Acceptance Notes

The sprint contract allowed a single consolidated blocker if strict execution
could not run. This result satisfies that fail-closed branch and avoids another
one-blocker ladder result.

## Residual Risk

Because no strict comparison ran, no source-changing dynamics fix is authorized
from this sprint alone.
