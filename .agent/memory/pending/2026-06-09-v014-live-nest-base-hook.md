# V0.14 Live-Nest Base Hook

Date: 2026-06-09

Status: pending stable-memory review.

Fact:

- `proofs/v014/live_nest_base_hook.json` verdict is
  `NATIVE_PORT_PLAN_READY`.
- The d02 `PB/MUB/PHB/HGT` mismatch is not fixable by reading naked
  `wrfinput_d02` or by a local `PB/MUB` formula patch.
- The missing production path is WRF's live-nest initialization sequence:
  `med_interp_domain` parent interpolation, generated
  `nest_interpdown_interp.inc` with `interp_fcn_sint`/`sint.F`,
  `blend_terrain`, then `start_domain_em` base recomputation.
- Native `wrfinput_d02` vs CPU-WRF h0 target-patch deltas are large:
  `HGT` max `89.50347900390625` m, `PB` max `1047.015625` Pa,
  `MUB` max `1050.3046875` Pa.
- WRF base formulas on CPU-WRF h0 terrain reproduce h0 base fields:
  `PB` residual max `0.04889917548280209` Pa, `MUB`
  `0.044447155625675805` Pa, `PHB` `0.09328280997578986` m2/s2.

Operational consequence:

- Next sprint should implement native live-nest base initialization as setup
  logic, not as a timestep-loop correction.
- CPU-WRF `wrfout_h0` may be used only as validation oracle, never as normal
  production input.
- TOST remains paused until the native fix is validated by target-patch and
  whole-domain grid-field comparisons.
