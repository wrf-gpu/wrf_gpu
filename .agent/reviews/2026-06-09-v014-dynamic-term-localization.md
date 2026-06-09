# V0.14 Dynamic Term Localization Review

- objective: produce the first compact source-derived WRF dynamic term layer from the green h10 same-state marker.
- verdict: `TERM_LAYER_EMITTED_final_stage_small_step_finish`.
- files changed: `proofs/v014/wrf_dynamic_term_localization.py`, `.json`, `.md`, `_patch.diff`, and this review.
- WRF copy/run paths: `/mnt/data/wrf_gpu2/v014_dynamic_terms/WRF`, `/mnt/data/wrf_gpu2/v014_dynamic_terms/run_case3`.
- patch hash: `84e0bc0d0494147af903c0307d3ec4e5cd94e604ffe1d6d596d349d035111ba4`.
- executable hash: `b042810093d715e915372e5dd8a5cdbbab44e32bd28d9b5233767fb61d422ab7`.
- proof objects: `/home/enric/src/wrf_gpu2/proofs/v014/wrf_dynamic_term_localization.json`, `/home/enric/src/wrf_gpu2/proofs/v014/wrf_dynamic_term_localization.md`, `/home/enric/src/wrf_gpu2/proofs/v014/wrf_dynamic_term_localization_patch.diff`.
- emitted fields/terms: native `T/P/PB/U/V/W/PH`, `MU`, `MUT/MUTS`, mass-coupled `MUU/MUUS/MUV/MUVS`, and `RU/RV/RW/T/PH/MU_TEND` plus `*_TENDF`.
- unresolved risks: no JAX same-state wrapper run; only first selected patch and surface K1/KSTAG01 emitted; tile-local post-`small_step_finish` is not yet the green history surface for `P/V/W`.
- next decision needed: narrower WRF emitter around pressure/rho/post-RK cadence, then JAX CPU wrappers for the green same-state surface.
