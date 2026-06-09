# Worker Report

Summary:

Opus reviewed the current v0.14 dynamic root-cause evidence after two GPT debug
sprints did not close the grid symptom. It challenged the manager's proposed
final-RK pressure-gradient/mass-wind target and found that the current evidence
does not justify editing final-RK coupling yet because the input to the final RK
step is already divergent.

Files Changed:

- `.agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md`
- `proofs/v014/dynamic_root_cause_opus_critic.json`

Commands Run:

- Read project rules, sprint contract, and trigger proof artifacts.
- Read `proofs/v014/same_state_momentum_mass.json`,
  `proofs/v014/grid_after_live_nest_base.json`,
  `proofs/v014/live_nest_base_source_fix.json`, and
  `proofs/v014/pre_rk_input_boundary.json`.
- Inspected dynamics entry points and savepoint availability.
- `python -m json.tool proofs/v014/dynamic_root_cause_opus_critic.json >/tmp/dynamic_root_cause_opus_critic.validated.json`
- `git diff -- src`

Proof Objects:

- `.agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md`
- `proofs/v014/dynamic_root_cause_opus_critic.json`

Result:

Verdict is
`MANAGER_FINAL_RK_TARGET_NOT_JUSTIFIED_INPUT_ALREADY_DIVERGED`.

The key point is that `proofs/v014/pre_rk_input_boundary.json` already shows
the JAX carry input to step 6000 diverged before the final RK step starts:
`MU'` worst about `267` Pa, `P'` worst about `590` Pa, and `T_OLD` worst about
`6.2` K. Therefore output-side final-RK mismatches are currently non-probative.

Handoff:

Dispatch a strict same-input single-RK-step parity sprint at d02 step 6000. Use
WRF's own pre-RK input savepoint, run one JAX dynamics step, and compare against
WRF's post-RK/pre-halo savepoint. Control tendency input and patch-width
validity before drawing any operator conclusion.
