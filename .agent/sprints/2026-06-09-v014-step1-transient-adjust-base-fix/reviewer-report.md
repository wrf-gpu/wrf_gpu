# Reviewer Report: V0.14 Step-1 Transient Adjust-Base Fix

Date: 2026-06-09

Decision: ACCEPT as a narrow source-helper plus proof sprint. Require the next
source sprint to wire the helper into the production live-nest init consumer
before claiming production Step-1 closure.

## Findings

- HIGH: The new helper exactly targets the proven WRF surface:
  post-`blend_terrain`/pre-`start_domain` current `MUB` for
  `adjust_tempqv`. It does not mutate `_apply_live_nest_base_init` or the final
  post-`start_domain` BaseState path.
- HIGH: Corrected theta closes the prior residual by about `93.6x`, from
  `0.00541785382188209 K` to `5.788684885033035e-05 K`, below the
  `0.001 K` gate.
- MEDIUM: This is not fully wired production behavior yet. The helper is
  production-callable, but the live-nest init consumer still needs explicit
  theta_m + `adjust_tempqv` wiring.
- LOW: Existing unrelated dirty file
  `proofs/v060/sfclayrev1_savepoint_parity_report.json` predates this sprint
  and must not be staged with this work.

## Evidence

- `proofs/v014/step1_transient_adjust_base_fix.md`
- `proofs/v014/step1_transient_adjust_base_fix.json`
- `.agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md`

## Required Next Sprint

Wire WRF theta_m conversion plus `adjust_tempqv`, using
`_wrf_live_nest_transient_adjust_mub`, into the production live-nest init
consumer. Then run the full Step-1 same-input d02 comparison across the 16-field
schema.
