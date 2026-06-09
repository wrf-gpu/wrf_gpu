You are GPT-5.5 xhigh acting as a verifier/debug worker for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-t-history-source-attribution/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Determine whether `JAX_MISMATCH_T` is caused by comparing the wrong JAX theta or
history source against WRF history `T`, or whether it is a real theta-evolution
mismatch. This is read-only attribution. Do not edit production `src/`, do not
run TOST, do not run Switzerland validation, and do not land FP32 work.

Key context:

- The green WRF target is immediately after
  `dyn_em/solve_em.F::after_all_rk_steps` and before RK halo exchanges.
- WRF history `T` at that target is `grid%th_phy_m_t0`, recorded as
  `MASS_K1.T_HIST_SRC`; `MASS_K1.T_THM` is a separate THM-side candidate and
  can mislead.
- Produced JAX checkpoint:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`
- Current canonical proof: `proofs/v014/jax_h10_prestep_carry.json`
  verdict `JAX_MISMATCH_T`, first mismatch `T` max_abs
  `3.3545763228707983`.

Deliver:

- `proofs/v014/jax_t_history_source_attribution.py`
- `proofs/v014/jax_t_history_source_attribution.json`
- `proofs/v014/jax_t_history_source_attribution.md`
- `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`

Required terminal behavior:

Print one compact verdict line and artifact paths. Keep large tables in JSON.

Required validation:

```bash
python -m py_compile proofs/v014/jax_t_history_source_attribution.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/jax_t_history_source_attribution.py
python -m json.tool proofs/v014/jax_t_history_source_attribution.json \
  >/tmp/jax_t_history_source_attribution.validated.json
```

When done, print:

`GPT T_HISTORY_SOURCE_ATTRIBUTION DONE`
