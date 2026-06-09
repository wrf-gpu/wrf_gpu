# Manager Closeout

## Outcome

Verdict: `BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL`.

No source patch was applied. The evidence shows CPU-WRF h0 `PB/MUB` are base
formula values on a post-nest blended terrain/base surface. The current JAX
loader only has the child `wrfinput_d02` split and does not have WRF's
parent-interpolated/blended `HGT/MUB/PHB` fields.

## Proof Objects

- `proofs/v014/base_state_split_fix.py`
- `proofs/v014/base_state_split_fix.json`
- `proofs/v014/base_state_split_fix.md`
- `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`

## Merge Decision

Merge Decision:

Accept and land the blocked proof. It prevents an unsafe shortcut and names the
real next hook.

## Scope Changes

The next scope is not a `build_replay_case` formula tweak. It is a WRF
live-nest initialization hook/port sprint for parent interpolation,
`blend_terrain`, and `start_domain_em` base recomputation.

## Lessons

The WRF h0 base state can be reproduced from h0 HGT within about 0.06 Pa, but
the missing production input is the h0 blended terrain/base surface itself.
Simplified bilinear+blend remains hundreds of Pa off and is not acceptable.

## Next Sprint

Open `.agent/sprints/2026-06-09-v014-live-nest-base-hook/` to capture or port
WRF's post-blend/pre-start-domain and post-start-domain base fields.
