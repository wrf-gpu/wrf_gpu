# Manager Closeout

## Outcome

The sprint is closed as a validated localization proof.

Final verdict:
`STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.

The v0.14 grid-parity bug is now localized earlier than the previous
pre-`small_step_prep` source/tendency boundary. The first material mismatch is
inside or immediately at WRF `first_rk_step_part1`, specifically `T_STATE`
after that routine's RK1 output.

## Proof Objects

- `proofs/v014/step1_rk1_source_boundary.py`
- `proofs/v014/step1_rk1_source_boundary.json`
- `proofs/v014/step1_rk1_source_boundary.md`
- `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-rk1-source-boundary.md`
- `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/wrf_truth`

## Merge Decision:

Merge proof, review, sprint-closeout, roadmap, and pending-memory artifacts only.
No production model source changed in this sprint.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_rk1_source_boundary.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_rk1_source_boundary.py`
- `python -m json.tool proofs/v014/step1_rk1_source_boundary.json >/tmp/step1_rk1_source_boundary.manager.validated.json`
- `git diff -- src/gpuwrf`

The rerun reproduced the verdict and left `src/gpuwrf` unchanged.

## Key Numbers

- First localized surface: `after_first_rk_step_part1`.
- Field: `T_STATE`.
- WRF vs JAX operational carry: max_abs `5.490173101425171`, RMSE `1.9175184863907806`.
- WRF vs `_physics_step_forcing.state`: max_abs `5.490142455570492`, RMSE `1.9174736017582765`.
- RK1 `small_step_prep` continuity remains exact for `T_WORK` and `P_WORK`, both max_abs `0.0`.

## Scope Changes

None. WRF instrumentation was disposable scratch only, CPU-only, and env-gated.
No TOST, Switzerland, FP32, memory source work, GPU, or production source edit
was performed.

## Lessons

The fastest rigorous path remains a focused savepoint/comparator ladder. The
evidence rules out continuing acoustic/small-step debugging as the next move.
The next sprint must split WRF `first_rk_step_part1` internals against JAX
physics adapter output and decide whether the missing behavior is a state
mutation, a tendency/forcing leaf, or a state/carry handoff.

## Next Sprint

Open `v014-step1-part1-physics-state-mutation`: instrument or compare the
internal WRF `first_rk_step_part1` surfaces for `T_STATE` and adjacent
tendency/forcing leaves, then map them to JAX `_physics_step_forcing` and the
operational carry. The proof gate is an exact internal source boundary or a
small performance-compatible fix with before/after Step-1 proof.
