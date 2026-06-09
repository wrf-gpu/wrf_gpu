# Tester Report: V0.14 Step-1 P/PH/MU Boundary Localization

## Tests Added Or Run

The worker and manager both ran the CPU-only proof path. No GPU, TOST,
Switzerland, FP32, memory source work, or Hermes was used.

Commands:

- `python -m py_compile proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `python -m json.tool proofs/v014/step1_p_ph_mu_boundary_localization.json >/tmp/step1_p_ph_mu_boundary_localization.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

The authoritative proof verdict is
`STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`.

Validation facts:

- JSON validation succeeded.
- CPU-only execution is recorded (`gpu_used=false`).
- Required ancestor `3aa5f15b` is present.
- `git diff -- src/gpuwrf` is empty; no production source edits were made.

The proof is a localization proof, not a fix proof. The strict final Step-1
comparison remains red with `P` as the largest residual.

## Gaps

Existing WRF truth does not emit raw boundary-package leaves or a
post-acoustic/pre-refresh split, so the sprint correctly stopped short of a
source edit.

Decision:

Accept the proof as a valid localization artifact. Continue with a narrower WRF
scratch surface inside `first_rk_step_part1` before source changes.
