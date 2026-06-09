# Review: V0.14 Same-State Momentum/Mass

Verdict: `JAX_MISMATCH_U_post_after_all_rk_steps_pre_halo`.

objective: produce a CPU-only JAX-vs-WRF comparison at the nearest named post-RK momentum/mass surface, or name the exact missing wrapper/input.

files changed:
- `proofs/v014/same_state_momentum_mass.py`
- `proofs/v014/same_state_momentum_mass.json`
- `proofs/v014/same_state_momentum_mass.md`
- `.agent/reviews/2026-06-09-v014-same-state-momentum-mass.md`

commands run:
- `python -m py_compile proofs/v014/same_state_momentum_mass.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_state_momentum_mass.py`
- `python -m json.tool proofs/v014/same_state_momentum_mass.json >/tmp/same_state_momentum_mass.validated.json`

proof objects produced:
- `proofs/v014/same_state_momentum_mass.json`
- `proofs/v014/same_state_momentum_mass.md`
- `.agent/reviews/2026-06-09-v014-same-state-momentum-mass.md`

result:
- Same-surface comparison ran at `post_after_all_rk_steps_pre_halo`.
- First failing field/surface: `U` at `post_after_all_rk_steps_pre_halo`, max_abs `6.292358893898424`, rmse `2.032497018496295`.
- `PHB` truth came from the green WRF h10 wrfout because the post-RK text hook did not emit PHB.
- The live-nest base fix remains partial and post-dates the carry; rerun after a fresh carry before interpreting base-field residuals.

unresolved risks:
- The checkpoint was produced before `live_nest_base_source_fix.json`; dynamic residuals are actionable, but PB/MUB/PHB residuals need a regenerated carry after that partial fix.
- The comparison covers the selected h10 patch and K1/KSTAG01 layers, not a full-domain/full-column proof.

next decision needed: run a fresh h10 carry after the base partial fix, then localize one layer earlier inside final RK U/V, mass, and theta-pressure source assembly.
