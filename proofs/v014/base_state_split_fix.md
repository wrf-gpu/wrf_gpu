# V0.14 Base-State Split Fix

Verdict: `BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL`.

## Summary

- No production source patch was applied.
- Native `build_replay_case(load_lateral_boundaries=False)` still loads the child `wrfinput_d02` split, which is not the CPU-WRF live-nest h0 split.
- CPU-WRF h0 `PB/MUB` are WRF base-formula values on the post-nest blended h0 terrain, but that terrain/base surface is produced by WRF parent interpolation plus `blend_terrain` before `start_domain_em`.
- Validation-only h0-HGT formula residuals: `PB` patch max `0.04889917548280209`, `MUB` patch max `0.044447155625675805` Pa.
- Simplified bilinear+blend reconstruction is rejected: `PB` patch max `796.2565574348409`, `MUB` patch max `798.7609739865584` Pa.

## Blocker

`build_replay_case` needs the WRF live-nest parent-interpolated/blended `HGT/MUB/PHB` surface, not CPU-WRF `wrfout_h0`. The exact next hook is WRF `share/mediation_integrate.F` after `blend_terrain` and `dyn_em/start_em.F` after base-state recomputation.

Full field tables are in `proofs/v014/base_state_split_fix.json`.
