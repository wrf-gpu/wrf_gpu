# Memory Patch Proposal

## Scope

Project memory update for the v0.14 same-input contract-builder sprint.

## Evidence

- `proofs/v014/same_input_contract_builder.json` records verdict
  `SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1`.
- The proof-local CPU loader constructs the initial d02 `State`, `Tendencies`,
  `BaseState`/metrics, `OperationalNamelist`, and `OperationalCarry` without
  `State.zeros`.
- The schema for `T`, `P`, `PB`, `PH`, `PHB`, `MU`, `MUB`, `U`, `V`, `W`,
  `QVAPOR`, `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, and `QGRAUP` is frozen.
- No strict numerical comparison ran because the full-domain CPU-WRF d02 step-1
  `post_after_all_rk_steps_pre_halo` truth surface does not exist.
- Manager validation reran CPU proof, JSON validation, Python compilation, and
  confirmed `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-same-input-contract-builder.md`

Also update `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md` and
`.agent/decisions/V0140-VALIDATION-PLAN.md` to record that the same-input CPU
loader/schema is ready and that the next validation-enabling sprint is a
disposable CPU-WRF step-1 full-domain truth hook.

## Patch

Record that v0.14 grid-parity debugging is now blocked on one concrete WRF truth
surface, not on the JAX CPU loader or field schema. The next sprint should patch
and run a disposable CPU-WRF tree to emit the accepted step-1 npz truth contract,
then rerun the contract builder to obtain the first strict WRF-vs-JAX residual
table.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
