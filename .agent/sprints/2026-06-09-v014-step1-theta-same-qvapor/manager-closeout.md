# Manager Closeout

## Outcome

The sprint is closed as a validated same-boundary QVAPOR rerun.

Final verdict:
`STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`.

Same-boundary QVAPOR is no longer the blocker. The remaining theta tail is
interior under the configured `distance_to_edge <= 5` boundary-band rule, so an
init-only production patch is not authorized yet.

## Proof Objects

- `proofs/v014/step1_theta_same_qvapor.py`
- `proofs/v014/step1_theta_same_qvapor.json`
- `proofs/v014/step1_theta_same_qvapor.md`
- `.agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md`

## Merge Decision:

Merge proof, review, sprint closeout, roadmap, and memory updates only. No
production model source patch is included.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_theta_same_qvapor.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_theta_same_qvapor.py`
- `python -m json.tool proofs/v014/step1_theta_same_qvapor.json >/tmp/step1_theta_same_qvapor.manager.validated.json`
- `git diff -- src/gpuwrf`

The proof reproduced the verdict, validated JSON, recorded `gpu_used=false`,
and left `src/gpuwrf` unchanged.

## Key Findings

- `theta_m + adjust_tempqv` still reduces `T_STATE` max_abs from
  `5.490173101425171 K` to `0.00541785382188209 K`.
- Same-boundary QVAPOR confirms the moisture piece is close:
  candidate QVAPOR max_abs versus WRF pre-call truth is
  `3.838436518426372e-06`.
- The worst residual cell is not in the horizontal boundary band:
  zero `{k:1,y:9,x:17}`, Fortran `{i:18,j:10,k:2}`, distance `9`.
- Boundary band max_abs is below the material threshold at
  `0.0005722015491755883 K`; interior max_abs remains above threshold at
  `0.00541785382188209 K`.

## Next Sprint

Open a CPU-only disposable-WRF intermediate savepoint sprint for exact
`adjust_tempqv` internals at the residual path. The sprint should emit WRF
pre/post `t_2`, QVAPOR, pressure inputs, and base inputs before any production
theta patch.
