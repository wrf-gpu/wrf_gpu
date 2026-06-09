# Worker Report

## Summary:

The worker produced a CPU-only proof harness for the requested strict same-input
single-RK-step parity boundary, but did not run a JAX-vs-WRF parity comparison
because the available WRF pre-RK savepoint is not sufficient to build the JAX RK
input state and controlled tendency/source inputs.

Verdict:
`SAME_INPUT_TENDENCY_INPUT_BLOCKED_PRE_RK_FULL_NATIVE_STATE_RK_TENDF_AND_HISTORY_SOURCE_FIELDS`.

## Files Changed

- `proofs/v014/same_input_single_rk_parity.py`
- `proofs/v014/same_input_single_rk_parity.json`
- `proofs/v014/same_input_single_rk_parity.md`
- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`

No `src/` files were changed.

## Commands Run

- `python -m py_compile proofs/v014/same_input_single_rk_parity.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity.json >/tmp/same_input_single_rk_parity.validated.json`
- `git diff -- src`

## Proof Objects Produced

- `proofs/v014/same_input_single_rk_parity.json`
- `proofs/v014/same_input_single_rk_parity.md`
- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`

## Findings

The current WRF pre-RK hook emits only `MASS_K1` fields:
`T_THM`, `T_OLD`, `T_HIST_SRC`, `P`, `PB`, `MU_NEW`, `MU_OLD`, and `MUB`.
That is not enough to construct full native-staggered JAX state, base
`Tendencies`, WRF `DryPhysicsTendencies`, history/source fields, or an
`OperationalCarry` for `_rk_scan_step_with_pre_halo_capture`.

The 17x17 patch width is not the primary blocker. If full inputs existed, only
one conservative mass-grid score cell would remain with an 8-cell halo, but the
proof is blocked before scoring.

## Unresolved Risks

- A full pre-RK WRF hook may need a wider patch to produce enough interior score
  cells after halo exclusion.
- The current proof does not support an upstream-drift, final-RK PGF/mass-wind,
  or theta/source conclusion. It only proves that the present instrumentation is
  insufficient.

## Next Decision Needed

Add a full CPU-WRF pre-RK native-state plus RK-fixed tendency/source savepoint
and a proof-only JAX loader/wrapper, then rerun this exact same-input boundary.
