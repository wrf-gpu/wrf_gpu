# Pending Memory: V0.14 Dynamic Root-Cause Opus Critic

Status: pending promotion after strict same-input single-RK-step parity proof.

Lesson:

- Do not localize final-RK output coupling while the pre-RK input carry is
  already divergent from WRF.
- Opus found that `proofs/v014/pre_rk_input_boundary.json` already shows JAX
  step-6000 input mismatch before final RK: `MU'` worst about `267` Pa, `P'`
  worst about `590` Pa, and `T_OLD` worst about `6.2` K.
- `proofs/v014/same_state_momentum_mass.json` is useful as a warning but not as
  a source-edit localizer because it used a stale pre-base-fix carry and is not
  strict same-input.
- The next proof boundary is WRF pre-RK input -> one JAX dynamics step -> WRF
  post-RK/pre-halo output at d02 step 6000, with controlled tendencies and
  stencil-valid patch scoring.

Evidence:

- `.agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md`
- `proofs/v014/dynamic_root_cause_opus_critic.json`
- `.agent/sprints/2026-06-09-v014-dynamic-root-cause-opus-critic/manager-closeout.md`
