# V0.14 JAX Theta Evolution Localization

Verdict: `THETA_MISMATCH_PRESTEP_OR_INPUT`.

## Boundary

- First mismatch: `checkpoint_prestep_carry.state.theta_minus_300` vs `wrf_final_stage_pre_small_step_finish.T_OLD`.
- First comparison: max_abs `6.218735851548047`, rmse `4.638818160588427`.
- Final pre-halo history comparison: max_abs `3.3545763228707983`, rmse `1.0296598586362888`.
- Final pre-finish `T_THM` diagnostic comparison: max_abs `824.4318205090697`, rmse `108.3608714587911`.

## Context

- The checkpoint hash matches the prior T history-source attribution proof.
- WRF `T_HIST_SRC` is treated as history T; WRF `T_THM` is diagnostic only.
- P/PB/MU/MUB context is included in JSON for every state boundary where the WRF surface exposes the matching fields.

## Next Decision

Open a WRF/JAX input-boundary emitter or hook sprint for explicit step-6000 pre-RK T/P/PB/MU/MUB before deciding any source-changing fix; do not start by editing final small_step_finish or history-source mapping.
