# Worker Report

Summary:

Localize the confirmed h10 `T`/theta mismatch to the narrowest reachable JAX
stage, cadence, or component boundary without production source edits.

Files changed:

- `proofs/v014/jax_theta_evolution_localization.py`
- `proofs/v014/jax_theta_evolution_localization.json`
- `proofs/v014/jax_theta_evolution_localization.md`
- `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`

Commands run:

- `python -m py_compile proofs/v014/jax_theta_evolution_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_theta_evolution_localization.py`
- `python -m json.tool proofs/v014/jax_theta_evolution_localization.json >/tmp/jax_theta_evolution_localization.validated.json`
- focused `jq` summaries of the generated JSON

Proof objects produced:

- `proofs/v014/jax_theta_evolution_localization.py`
- `proofs/v014/jax_theta_evolution_localization.json`
- `proofs/v014/jax_theta_evolution_localization.md`
- `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`

Verdict:

`THETA_MISMATCH_PRESTEP_OR_INPUT`.

The earliest available WRF start-of-step/RK-reference theta surface
(`T_OLD`/`grid%t_1` in the final-stage pre-`small_step_finish` emitter) already
differs from the real JAX step-5999 carry input before current-step physics or
RK. First theta comparison max_abs is `6.218735851548047`, RMSE is
`4.638818160588427`. JAX mirror parity against the existing pre-halo helper is
`0.0`, so the proof-local RK mirror is not the source of this verdict.

Unresolved risks:

- The current WRF artifacts do not expose full explicit step-6000 pre-RK
  `P/PB/MUB`; only `MU_OLD` is available as input-boundary context.
- The proof covers the selected h10 patch and source-emitter levels, not a
  full-domain all-level WRF input-boundary surface.
- No GPU, TOST, Switzerland validation, or FP32 source work was run.

Next decision:

Open a WRF/JAX input-boundary emitter or hook sprint for explicit step-6000
pre-RK `T/P/PB/MU/MUB` before deciding any source-changing fix.
