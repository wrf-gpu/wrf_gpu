# V0.14 H10 Pre-Step Carry Checkpoint

Verdict: `JAX_MISMATCH_T`.

## Target

- WRF target: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.
- Domain/step: `d02`, step `6000`, valid `2026-05-02T04:00:00+00:00`.
- Required JAX pre-step checkpoint: completed step `5999`.

## Checkpoint Probe

- Candidates inspected: `7`.
- Usable h10 pre-step candidates: `1`.
- Real same-surface comparison run: `True`.

## Comparison

- First mismatch: `T` max_abs `3.3545763228707983` rmse `1.0296598586362888`.
