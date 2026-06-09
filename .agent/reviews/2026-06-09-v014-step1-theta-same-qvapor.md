# Review: V0.14 Step-1 Theta Same-Boundary QVAPOR

Verdict: `STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`.

objective: rerun the Step-1 live-nest theta semantics proof with the validated same-boundary pre-call QVAPOR root and classify the final residual as boundary-local or interior.

files changed:
- `proofs/v014/step1_theta_same_qvapor.py`
- `proofs/v014/step1_theta_same_qvapor.json`
- `proofs/v014/step1_theta_same_qvapor.md`
- `.agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md`

commands run:
- `python -m py_compile proofs/v014/step1_theta_same_qvapor.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_theta_same_qvapor.py`
- `python -m json.tool proofs/v014/step1_theta_same_qvapor.json >/tmp/step1_theta_same_qvapor.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_theta_same_qvapor.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_theta_same_qvapor.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md`
- `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only` reused, not rebuilt

unresolved risks:
- The final same-boundary QVAPOR candidate still has an interior residual above 1e-3 K.
- A source patch is not authorized without WRF intermediate theta/adjust_tempqv pressure inputs or an equivalent proof.

next decision needed: Emit or recover WRF theta_m/adjust_tempqv intermediate inputs for the residual cell before patching production.
