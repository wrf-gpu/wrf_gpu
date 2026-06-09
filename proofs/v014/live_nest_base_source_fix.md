# V0.14 Live-Nest Base Source Fix

Verdict: `LIVE_NEST_BASE_SOURCE_PARTIAL_NO_GRID_SYMPTOM_PROOF`.

## Summary

- Production source was patched to apply native live-nest child base initialization before timestepping.
- CPU-WRF h0/h10 are used only as validation oracles.
- Scope is deliberately narrow: this closes the live-nest base-state mismatch, not the V10/grid-field divergence.
- No init-override falsifier or direct V10/grid-field proof has been run on this patch; TOST remains paused.
- The original target-patch base deltas are closed to formula-level residuals:
  - PB `1047.015625` -> `0.04890023032203317` Pa.
  - MUB `1050.3046875` -> `0.044447155625675805` Pa.
  - PHB fixed max `0.09328280997578986` m2/s2; HGT fixed max `2.4167598553503922e-05` m.
- Total-state target-patch deltas also improve materially:
  - P_TOTAL `1080.4921875` -> `33.43062101097894` Pa.
  - MU_TOTAL `1038.0496826171875` -> `12.299452038438176` Pa.
  - PH_TOTAL `878.0291748046875` -> `0.09377109122578986`.
- The state-visible base split (`total - perturbation`) matches the recomputed base fields.
- Remaining dynamic perturbation residuals are not hidden: fixed initial P/MU perturbation patch max is P `33.4765625` Pa and MU `12.2550048828125` Pa against h0.
- Next required gate: run the init-override/direct grid-field proof and same-state momentum/mass tendency localization before any TOST or grid-parity closure claim.

## Runtime Impact

- A host-side full-SINT terrain interpolation runs once during child initialization.
- No CPU-WRF history file is used as production input.
- No host/device transfer is added inside timestep loops.
- Standalone single-domain initialization is unchanged unless `live_nest_parent` is explicitly passed.

Full field tables are in `proofs/v014/live_nest_base_source_fix.json`.
