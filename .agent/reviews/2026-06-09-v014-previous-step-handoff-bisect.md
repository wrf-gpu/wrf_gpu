# Review: V0.14 Previous-Step Handoff Bisection

verdict: `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`

objective: bisect the live nested replay producer path that writes the bad h10 d02 step-5999 OperationalCarry.

files changed:
- `proofs/v014/previous_step_handoff_bisect.py`
- `proofs/v014/previous_step_handoff_bisect.json`
- `proofs/v014/previous_step_handoff_bisect.md`
- `.agent/reviews/2026-06-09-v014-previous-step-handoff-bisect.md`

commands run:
- `python -m py_compile proofs/v014/previous_step_handoff_bisect.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/previous_step_handoff_bisect.py`
- `python -m json.tool proofs/v014/previous_step_handoff_bisect.json >/tmp/previous_step_handoff_bisect.validated.json`
- `WRFGPU2_PREVIOUS_STEP_HANDOFF_BISECT_ALLOW_GPU=1 WRFGPU2_PREVIOUS_STEP_HANDOFF_BISECT_FORCE_REPLAY=1 JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src python proofs/v014/previous_step_handoff_bisect.py`

proof objects produced:
- `proofs/v014/previous_step_handoff_bisect.json`
- `proofs/v014/previous_step_handoff_bisect.md`
- `.agent/reviews/2026-06-09-v014-previous-step-handoff-bisect.md`
- `/mnt/data/wrf_gpu2/v014_previous_step_handoff_bisect/previous_step_handoff_bisect.live_replay_compact.json`

unresolved risks:
- The WRF oracle is the existing final h10 pre-RK patch only; no WRF step-5997 or step-5998 oracle was generated in this evidence sprint.
- The final RK3 pre-halo internal state remains behind a missing _advance_chunk/_physics_boundary_step hook.
- The live replay required GPU because the current native-domain loader is not CPU-capable.

next decision needed: Open a narrower earlier-handoff/source sprint before d02 step 5997; do not target _operational_force or final child _advance_chunk first.
