# Reviewer Report

## Findings

- The proof answers the sprint contract: it splits raw child state, live child
  state, boundary package, initial carry, and haloed step-entry.
- Verdict is
  `STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.
- `T_STATE` is unchanged after raw state load; all stage-to-stage transition
  max_abs values are `0.0`.
- Live-nest base init materially changes and improves `PB/PHB/MUB`, so the
  issue is a state/base semantic split rather than a no-op live-nest path.
- Interior and boundary-band metrics both remain material, so this is not a
  lateral-boundary-only package problem.
- Production `src/gpuwrf/**` remained unchanged.

## Contract Compliance

The worker stayed CPU-only, reused the accepted WRF pre-call truth, avoided WRF
rebuilds and long validation, wrote the required proof/review artifacts, and
reported detailed metrics in JSON with a short markdown summary.

## Correctness Risks

The proof does not yet identify the exact WRF formula or call sequence that
updates `grid%t_2` during live-nest initialization. Source evidence points to
`med_nest_initial` calling `start_domain_em(..., nest%t_2, nest%p, QV...)`
after terrain/base blending, but that must be proven against the actual WRF
truth before patching production JAX.

## Performance Risks

None introduced. No production source changed. The likely future fix is an
initialization-only GPU/JAX array transform, not a timestep-loop operation, but
that still must be kept out of the hot loop and validated.

## Required Fixes

Open a focused WRF live-nest theta semantics/fix sprint. It should compare
candidate JAX reconstructions of `t_2`/`p` after WRF `start_domain_em` against
the existing WRF pre-call truth and apply the smallest source patch only after
that candidate closes the residual.

## Decision:

Accept.
