# Tester Report

Decision: accepted as a blocked-instrumentation proof, not as a dynamics parity
result.

## Validation

Manager-side validation to run at closeout:

- `python -m py_compile proofs/v014/same_input_single_rk_parity.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity.json >/tmp/same_input_single_rk_parity.manager.validated.json`
- `git diff -- src`
- `python scripts/close_sprint.py .agent/sprints/2026-06-09-v014-same-input-single-rk-parity`
- `python scripts/validate_memory_patch.py .agent/sprints/2026-06-09-v014-same-input-single-rk-parity/memory-patch.md`

## Result

The proof JSON validates and records that no strict comparison was run. The
reason is specific and actionable: the available WRF pre-RK surface does not
contain full native-staggered state, base tendencies, RK-fixed physics/source
tendencies, or the wrapper inputs required to construct a JAX `OperationalCarry`
with WRF-controlled tendencies.

## Acceptance Notes

This satisfies the sprint contract's fallback requirement: if a strict same-input
comparison cannot be made without reconfounding tendencies or stencils, emit a
blocked verdict naming the missing fields and next hook.

It does not satisfy any source-fix criterion. No production `dynamics/`,
`runtime/`, or `physics/` code should be edited from this result alone.

## Residual Risk

After the full hook exists, the patch may still need to be widened. That should
be handled inside the next proof sprint instead of guessed now.
