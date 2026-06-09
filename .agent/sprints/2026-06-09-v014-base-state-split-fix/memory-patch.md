# Memory Patch

Scope:

Project-memory update for the v0.14 base-state split fix attempt.

Evidence:

- `proofs/v014/base_state_split_fix.json` verdict is
  `BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL`.
- No production source patch was applied.
- CPU-WRF h0 `PB/MUB` are generated from a post-nest blended terrain/base
  surface; formula on h0 HGT matches within about `0.06` Pa.
- Simplified bilinear+blend reconstruction remains far outside tolerance:
  `PB` patch max `796.2565574348409` Pa and `MUB` patch max
  `798.7609739865584` Pa.
- Exact WRF chain named: `med_interp_domain`, generated
  `nest_interpdown_interp.inc`, `interp_fcn_sint`/`sint.F`,
  `blend_terrain`, and `start_domain_em`.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-base-state-split-fix-blocked.md`.
After the WRF live-nest hook/port sprint lands, condense into stable memory with
the actual native implementation path.

Reviewer Status:

Pending. Do not promote to stable memory until the next hook/port sprint
produces a real oracle or native implementation.
