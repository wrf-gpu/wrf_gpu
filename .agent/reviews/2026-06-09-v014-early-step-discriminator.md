# Review: V0.14 Early-Step Same-Input Discriminator

Verdict: `EARLY_STEP_DISCRIMINATOR_BLOCKED_CPU_REALCASE_LOADER_GPU_ONLY_NO_CANDIDATE_WRF_PREHALO_TRUTH_NO_SAME_INPUT_CARRY_CONTRACT`.

Objective: run the consolidated early-step same-input discriminator from shared `wrfinput_d02`, executing a strict same-input comparison if technically possible; otherwise emit one consolidated blocker across steps 1, 60, 600, 3000, and 5999.

Files changed:
- `proofs/v014/early_step_discriminator.py`
- `proofs/v014/early_step_discriminator.json`
- `proofs/v014/early_step_discriminator.md`
- `.agent/reviews/2026-06-09-v014-early-step-discriminator.md`

Commands run:
- `git branch --show-current`
- `git log -1 --oneline`
- `git status --short`
- `python -m py_compile proofs/v014/early_step_discriminator.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/early_step_discriminator.py`
- `python -m json.tool proofs/v014/early_step_discriminator.json >/tmp/early_step_discriminator.validated.json`
- `git diff -- src/gpuwrf`

Proof objects produced:
- `proofs/v014/early_step_discriminator.json`
- `proofs/v014/early_step_discriminator.md`
- `/tmp/early_step_discriminator.validated.json`

Result:
- No strict comparison ran; weak WRF-output, JAX-vs-JAX, one-cell, and mixed-source comparisons were avoided.
- CPU probe of `build_replay_case(..., domain="d02", load_lateral_boundaries=False)` blocked at `State.zeros requires a GPU device`.
- No candidate-step WRF post-RK/pre-halo surface exists under the allowed scratch root or inspected prior proof roots.
- Existing step-6000 surfaces are non-candidate and patch-only.
- `git diff -- src/gpuwrf` is empty.

Unresolved risks:
- No numerical first-divergent field/operator is named.
- A direct CPU-only proof loader might be possible, but it needs an accepted contract to avoid fabricating a non-production state.

Next decision: authorize one contract-building sprint for a CPU-compatible `wrfinput -> OperationalCarry` loader plus candidate-step WRF post-RK/pre-halo full-field surface, then rerun this discriminator.
