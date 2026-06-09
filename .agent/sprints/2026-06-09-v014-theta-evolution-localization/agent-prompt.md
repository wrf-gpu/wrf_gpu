You are GPT-5.5 xhigh acting as a verifier/debug worker for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-theta-evolution-localization/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Localize the confirmed h10 theta evolution mismatch to the narrowest reachable
JAX stage/cadence/component boundary before any production source fix. This is
read-only localization. Do not edit production `src/`, do not run TOST, do not
run Switzerland validation, and do not land FP32 work.

Key context:

- `proofs/v014/jax_t_history_source_attribution.json` verdict is
  `T_EVOLUTION_MISMATCH_CONFIRMED`.
- Best WRF history `T_HIST_SRC` candidate was
  `captured_pre_halo_state.theta_minus_300`, still max_abs
  `3.3545763228707983`.
- Produced checkpoint:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`
- WRF source-derived layers:
  `proofs/v014/wrf_dynamic_term_localization.json` and
  `proofs/v014/wrf_post_rk_refresh_localization.json`.

Deliver:

- `proofs/v014/jax_theta_evolution_localization.py`
- `proofs/v014/jax_theta_evolution_localization.json`
- `proofs/v014/jax_theta_evolution_localization.md`
- `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`

Required validation:

```bash
python -m py_compile proofs/v014/jax_theta_evolution_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/jax_theta_evolution_localization.py
python -m json.tool proofs/v014/jax_theta_evolution_localization.json \
  >/tmp/jax_theta_evolution_localization.validated.json
```

Required terminal behavior:

Print one compact verdict line and artifact paths. Keep large tables in JSON.

When done, print:

`GPT THETA_EVOLUTION_LOCALIZATION DONE`
