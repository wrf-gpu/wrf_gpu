# Worker Report

Summary:

The read-only attribution proof confirmed that `JAX_MISMATCH_T` is not explained
by selecting the wrong JAX theta/history leaf. No inspected JAX candidate matches
WRF history `T` or WRF `T_THM` within the frozen h10 tolerance.

Objective:

Attribute whether the first same-surface h10 mismatch is a source/cadence
mapping issue or a real theta-evolution mismatch.

Files changed:

- `proofs/v014/jax_t_history_source_attribution.py`
- `proofs/v014/jax_t_history_source_attribution.json`
- `proofs/v014/jax_t_history_source_attribution.md`
- `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`

Commands run:

- `python -m py_compile proofs/v014/jax_t_history_source_attribution.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_t_history_source_attribution.py`
- `python -m json.tool proofs/v014/jax_t_history_source_attribution.json >/tmp/jax_t_history_source_attribution.validated.json`

Proof objects produced:

- `proofs/v014/jax_t_history_source_attribution.json`
- `proofs/v014/jax_t_history_source_attribution.md`
- `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`

Result:

`T_EVOLUTION_MISMATCH_CONFIRMED`.

Best WRF history `T_HIST_SRC` candidate is
`captured_pre_halo_state.theta_minus_300`, still max_abs
`3.3545763228707983` and RMSE `1.0296598586362888`. Best WRF `T_THM`
candidate is `captured_final_carry.t_2ave_minus_300`, still max_abs
`3.677881697025043`. P/PB/MU/MUB are also divergent on the same patch.

Unresolved risks:

- Only Boole's selected h10 patch was attributed, not the full grid.
- The proof is CPU-only and intentionally did not run TOST or Switzerland
  validation.

Next decision needed:

Open a theta-evolution localization sprint. Do not spend the next sprint on
JAX-vs-WRF history source remapping for `T`.
