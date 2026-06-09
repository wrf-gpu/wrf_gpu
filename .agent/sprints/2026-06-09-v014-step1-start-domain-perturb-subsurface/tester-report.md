# Tester Report: V0.14 Step-1 Start-Domain Perturbation Subsurface

Decision: `PASS_LOCALIZATION_NO_SOURCE_PATCH`.

Validated artifacts:

- `proofs/v014/step1_start_domain_perturb_subsurface.py`
- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- `proofs/v014/step1_start_domain_perturb_subsurface.md`
- `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

Manager validation commands:

- `python -m py_compile proofs/v014/step1_start_domain_perturb_subsurface.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_start_domain_perturb_subsurface.py`
- `python -m json.tool proofs/v014/step1_start_domain_perturb_subsurface.json >/tmp/step1_start_domain_perturb_subsurface.manager_final.json`
- `git diff -- src/gpuwrf`
- `git diff --check -- proofs/v014/step1_start_domain_perturb_subsurface.py proofs/v014/step1_start_domain_perturb_subsurface.json proofs/v014/step1_start_domain_perturb_subsurface.md proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff .agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

Result:

- Python compile passed.
- CPU-only proof rerun passed and rewrote JSON/MD with verdict
  `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_CURRENT_JAX_AL_ALT_BASE_INPUT_GAP`.
- JSON validation passed.
- `git diff -- src/gpuwrf` was empty, confirming no production source edit.
- Diff whitespace check passed for the sprint artifacts.

Residual risk:

The proof is a localization result, not a fix. It explicitly refutes patching
`P/MU` with current JAX inputs because the residual remains above the material
gate.
