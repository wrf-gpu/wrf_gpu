# Worker Report

Summary: produced a compact CPU-WRF source-derived dynamic layer for the h10
same-state grid-parity investigation. The emitted layer is bounded around
final-stage `small_step_finish` in disposable WRF scratch, with no repo `src/`
edits, no GPU use, and no Hermes traffic.

Files changed:

- `proofs/v014/wrf_dynamic_term_localization.py`
- `proofs/v014/wrf_dynamic_term_localization.json`
- `proofs/v014/wrf_dynamic_term_localization.md`
- `proofs/v014/wrf_dynamic_term_localization_patch.diff`
- `.agent/reviews/2026-06-09-v014-dynamic-term-localization.md`

Proof summary: verdict `TERM_LAYER_EMITTED_final_stage_small_step_finish`.
The post-RK marker remains green against CPU h10, while retained GPU/JAX h10
still diverges on the same marker patch. The emitted tile-local
`post_small_step_finish` surface is useful but not history-aligned for `P/V/W`.

Commands reported by worker:

- `python -m py_compile proofs/v014/wrf_dynamic_term_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/wrf_dynamic_term_localization.py`
- `python -m json.tool proofs/v014/wrf_dynamic_term_localization.json >/tmp/wrf_dynamic_term_localization.validated.json`

Next recommendation: instrument the pressure/rho/post-RK refresh path before or
around `after_all_rk_steps`, then compare JAX at the first green WRF surface.
