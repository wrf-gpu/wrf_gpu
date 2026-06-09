# Review: V0.14 H10 Pre-Step Carry Checkpoint

Verdict: `JAX_MISMATCH_T`.

objective: complete the CPU-only H10 pre-step carry checkpoint and compare through the JAX pre-halo hook if a real carry is available.

files changed:
- `proofs/v014/jax_h10_prestep_carry.py`
- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_h10_prestep_carry.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md`

commands run:
- `python -m py_compile proofs/v014/jax_h10_prestep_carry.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_h10_prestep_carry.py`
- `python -m json.tool proofs/v014/jax_h10_prestep_carry.json >/tmp/jax_h10_prestep_carry.validated.json`

proof objects produced:
- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_h10_prestep_carry.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md`

result:
- Same-surface comparison ran with verdict `JAX_MISMATCH_T`.

unresolved risks:
- Only the selected Boole h10 patch was compared; broader field coverage remains a follow-up.

next decision needed: Open a T history/source-attribution sprint before any production source fix; compare JAX theta/history candidates against WRF T_HIST_SRC/grid%th_phy_m_t0 and THM-side candidates.
