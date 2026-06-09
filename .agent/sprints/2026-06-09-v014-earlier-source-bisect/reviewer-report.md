# Reviewer Report

## Findings

No blocking findings for this evidence sprint.

The result is specific and actionable: the initial JAX child carry follows
`wrfinput_d02`, while CPU-WRF h0/h1/h10 and pre-RK truth use a stable different
`PB/MUB` base-state split. The run therefore does not need replay-time drift to
explain the bad h10 base carry.

## Contract Compliance

Decision: compliant.

- No production `src/` files were edited.
- No WRF source was edited.
- GPU was used only for the targeted replay because CPU native load is
  `State.zeros`/GPU-gated.
- No TOST, Switzerland validation, broad validation, or FP32 source work was
  run.
- Top-level Markdown stays compact; large details are in JSON.

## Correctness Risks

The source formula for WRF's h0 base-state split is not yet identified. A fix
that simply reads CPU-WRF wrfout h0 in normal production would be a shortcut and
must not be accepted without an explicit validation-only or oracle-only boundary.

## Performance Risks

The targeted replay peaked at sampled VRAM 9091 MiB. This is a debug proof, not
a production memory claim.

## Required Fixes

None before committing the evidence. Next source sprint should patch
`build_replay_case` or emit a blocked verdict with the exact WRF routine/formula
needed.

## Decision

Accept and commit proof artifacts. Open the base-state split fix sprint.
