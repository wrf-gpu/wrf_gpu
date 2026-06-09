# Review: V0.14 H10 Pre-Step Carry Producer

verdict: `JAX_MISMATCH_T`

objective: produce a CPU-loadable full JAX OperationalCarry checkpoint at completed d02 step 5999 and run the h10 pre-halo comparison if produced.

files changed:
- `proofs/v014/jax_h10_prestep_carry_producer.py`
- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/jax_h10_prestep_carry_producer.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`

commands run:
- `python -m py_compile proofs/v014/jax_h10_prestep_carry_producer.py`
- `WRFGPU2_H10_PRODUCER_ALLOW_GPU=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=src python proofs/v014/jax_h10_prestep_carry_producer.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_h10_prestep_carry_producer.py`
- `python -m json.tool proofs/v014/jax_h10_prestep_carry_producer.json >/tmp/jax_h10_prestep_carry_producer.validated.json`

proof objects produced:
- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/jax_h10_prestep_carry_producer.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`
- `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`

unresolved risks:
- The comparison covers Boole's selected h10 patch, not the full grid.
- The producer uses private proof/runtime helpers, intentionally without landing a public checkpoint API.

next decision needed: Open a T history/source-attribution sprint before any production source fix; compare JAX theta/history candidates against WRF T_HIST_SRC/grid%th_phy_m_t0 and THM-side candidates.
