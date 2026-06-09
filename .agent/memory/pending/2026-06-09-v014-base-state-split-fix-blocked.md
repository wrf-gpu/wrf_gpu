# V0.14 Base-State Split Fix Blocked

Date: 2026-06-09

Status: pending stable-memory review.

Fact:

- `proofs/v014/base_state_split_fix.json` verdict is
  `BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL`.
- The bad `PB/MUB` split cannot be fixed by a local `build_replay_case`
  formula tweak alone.
- CPU-WRF h0 `PB/MUB` are WRF base-formula values on post-nest blended h0
  terrain; formula on h0 HGT matches within about `0.06` Pa.
- The missing production input is WRF's parent-interpolated/blended
  `HGT/MUB/PHB` surface produced before `start_domain_em` base recomputation.
- Simplified bilinear+blend is not acceptable (`PB`/`MUB` patch residuals near
  800 Pa).

Operational consequence:

- Next sprint must instrument or port WRF live-nest initialization:
  `med_interp_domain` parent interpolation, `blend_terrain`, and
  `start_domain_em` base recomputation.
- Do not patch `PB/MUB` alone and do not use CPU-WRF `wrfout_h0` as normal
  production input.
