# Manager Closeout

## Outcome

The Opus critic sprint is accepted. It materially improved the debug method by
showing that my previous next target was too late in the timestep.

Accepted verdict:
`MANAGER_FINAL_RK_TARGET_NOT_JUSTIFIED_INPUT_ALREADY_DIVERGED`.

## Proof Objects

- `.agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md`
- `proofs/v014/dynamic_root_cause_opus_critic.json`

## Merge Decision:

Merge the review and compact JSON. Do not merge any production source edit from
this sprint.

## Key Evidence

`proofs/v014/pre_rk_input_boundary.json` already shows the JAX pre-step carry
input to step 6000 diverged before final RK: `MU'` worst about `267` Pa, `P'`
worst about `590` Pa, and `T_OLD` worst about `6.2` K.

That makes final-RK output mismatch non-probative. The next proof must start
from WRF's own pre-RK input.

## Next Sprint

Open and dispatch `same_input_single_rk_parity`: WRF pre-RK input savepoint at
d02 step 6000 -> one JAX dynamics step -> WRF post-RK/pre-halo comparison. The
sprint must control tendencies and score only stencil-valid patch cells.
