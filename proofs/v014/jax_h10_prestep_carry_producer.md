# V0.14 H10 Pre-Step Carry Producer

Verdict: `JAX_MISMATCH_T`.

## Result

- Checkpoint produced: `True`.
- Checkpoint path: `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`.
- CPU-loadable: `True`.
- GPU used: `True`.
- Comparison run: `RAN`.
- Comparison verdict: `JAX_MISMATCH_T`.

## Next Decision

Open a T history/source-attribution sprint before any production source fix; compare JAX theta/history candidates against WRF T_HIST_SRC/grid%th_phy_m_t0 and THM-side candidates.
