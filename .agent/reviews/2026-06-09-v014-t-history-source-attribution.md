# Review: V0.14 T History Source Attribution

verdict: `T_EVOLUTION_MISMATCH_CONFIRMED`

objective: determine whether `JAX_MISMATCH_T` is a JAX theta/history-source mapping error or a real theta-evolution mismatch.

files changed:
- `proofs/v014/jax_t_history_source_attribution.py`
- `proofs/v014/jax_t_history_source_attribution.json`
- `proofs/v014/jax_t_history_source_attribution.md`
- `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`

commands run:
- `python -m py_compile proofs/v014/jax_t_history_source_attribution.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_t_history_source_attribution.py`
- `python -m json.tool proofs/v014/jax_t_history_source_attribution.json >/tmp/jax_t_history_source_attribution.validated.json`

proof objects produced:
- `proofs/v014/jax_t_history_source_attribution.json`
- `proofs/v014/jax_t_history_source_attribution.md`
- `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`

result:
- No inspected JAX theta/history candidate matches WRF history T or WRF T_THM within the frozen tolerance.
- The WRF history source and THM-side source are explicitly separated.
- The checkpoint hash matches both the producer proof and the canonical h10 comparison proof.

unresolved risks:
- Only Boole's selected h10 patch was compared; this attributes the first mismatch, not full-domain parity.
- The proof is CPU-only and intentionally does not run TOST or Switzerland validation.

next decision needed: Open a theta-evolution localization sprint; do not spend the next sprint on JAX-vs-WRF history source remapping for `T`.
