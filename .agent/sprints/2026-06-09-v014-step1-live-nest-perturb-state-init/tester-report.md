# Tester Report: V0.14 Step-1 Live-Nest Perturbation-State Init

Decision: pass localization-proof validation.

validation commands:

- `python -m py_compile proofs/v014/step1_live_nest_perturb_state_init.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_perturb_state_init.py`
- `python -m json.tool proofs/v014/step1_live_nest_perturb_state_init.json >/tmp/step1_live_nest_perturb_state_init.manager.validated.json`
- `git diff -- src/gpuwrf`
- `git diff --check`
- `python scripts/validate_memory_patch.py .agent/sprints/2026-06-09-v014-step1-live-nest-perturb-state-init/memory-patch.md`

result:

- All validation commands passed.
- `git diff -- src/gpuwrf` produced no output.
- No GPU was used.
- No TOST, Switzerland, FP32 source work, memory source work, or Hermes was
  used.

residual risk:

- This is a localization proof, not a model-code fix.
