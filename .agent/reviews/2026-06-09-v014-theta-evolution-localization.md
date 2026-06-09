# Review: V0.14 Theta Evolution Localization

verdict: `THETA_MISMATCH_PRESTEP_OR_INPUT`

objective: localize the confirmed h10 theta evolution mismatch to the narrowest reachable JAX stage/cadence/component boundary before any production source fix.

files changed:
- `proofs/v014/jax_theta_evolution_localization.py`
- `proofs/v014/jax_theta_evolution_localization.json`
- `proofs/v014/jax_theta_evolution_localization.md`
- `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`

commands run:
- `python -m py_compile proofs/v014/jax_theta_evolution_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_theta_evolution_localization.py`
- `python -m json.tool proofs/v014/jax_theta_evolution_localization.json >/tmp/jax_theta_evolution_localization.validated.json`

proof objects produced:
- `proofs/v014/jax_theta_evolution_localization.json`
- `proofs/v014/jax_theta_evolution_localization.md`
- `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`

result:
- The earliest available WRF start-of-step/RK-reference theta surface (final-stage pre-small_step_finish T_OLD/grid%t_1) already differs from the real JAX step-5999 carry input before current-step physics or RK.
- The proof used the exact h10 full-carry checkpoint recorded by the prior T attribution sprint.
- No production `src/` files, WRF source, TOST, Switzerland validation, or FP32 work were touched.

unresolved risks:
- Only the selected Boole h10 patch and k=1 mass layer / kstag 0..1 W/PH source emitters were compared.
- WRF has no separate full-domain step-5999 prestep emitter in the current artifact set; `T_OLD` is the narrowest available WRF input/reference theta surface.
- The proof is CPU-only and intentionally does not run TOST or Switzerland validation.

next decision needed: Open a WRF/JAX input-boundary emitter or hook sprint for explicit step-6000 pre-RK T/P/PB/MU/MUB before deciding any source-changing fix; do not start by editing final small_step_finish or history-source mapping.
