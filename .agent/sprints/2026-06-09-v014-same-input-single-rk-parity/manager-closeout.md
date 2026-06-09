# Manager Closeout

## Outcome

The sprint is closed as a valid blocked-instrumentation proof with verdict
`SAME_INPUT_TENDENCY_INPUT_BLOCKED_PRE_RK_FULL_NATIVE_STATE_RK_TENDF_AND_HISTORY_SOURCE_FIELDS`.

No same-input JAX-vs-WRF parity comparison was run. That is the correct result
for the current artifacts: the available WRF pre-RK hook is too narrow to build
the required JAX state and tendency/source inputs without making a misleading
comparison.

## Proof Objects

- `proofs/v014/same_input_single_rk_parity.py`
- `proofs/v014/same_input_single_rk_parity.json`
- `proofs/v014/same_input_single_rk_parity.md`
- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`

Key blocker:

- Current WRF pre-RK input contains only `MASS_K1` records for `T_THM`, `T_OLD`,
  `T_HIST_SRC`, `P`, `PB`, `MU_NEW`, `MU_OLD`, and `MUB`.
- Missing full native U/V/W/PH state, full mass columns, moisture/scalar/carry
  leaves, JAX base `Tendencies`, WRF `DryPhysicsTendencies`, history/source
  leaves, and a proof-only `OperationalCarry` loader.

## Merge Decision:

Merge proof/review/sprint artifacts only. Do not merge or authorize any
production dycore edit from this sprint.

## Scope Changes

No production `src/` code changed. No GPU, TOST, Switzerland validation, FP32, or
memory source work was run.

## Lessons

The Opus critic's recommended boundary was correct, but the existing savepoint
is not rich enough to execute it. The next source-adjacent task is not a model
fix; it is a full pre-RK WRF state/tendency hook plus a proof-only JAX loader so
the same-input comparison can finally be made.

## Next Sprint

Open a full pre-RK native-state/tendency savepoint sprint. It should emit
U/V/W/T/P/PB/PH/PHB/MU/MUB/QV and active scalar/moisture state on native grids,
RK history/source fields, WRF dry physics/source tendencies, and the metadata
needed to build an `OperationalCarry`. Then rerun the same-input single-RK
parity proof on halo-valid cells.
