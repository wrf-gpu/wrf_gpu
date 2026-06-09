# Worker Report

## Summary

Summary:

Verdict: `BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL`.

The worker did not patch production source. The required WRF h0 `PB/MUB` split
depends on live-nest parent interpolation, `blend_terrain`, and
`start_domain_em` base recomputation. A normal production dependency on
CPU-WRF `wrfout_h0` was rejected as a shortcut.

## Files Changed

- `proofs/v014/base_state_split_fix.py`
- `proofs/v014/base_state_split_fix.json`
- `proofs/v014/base_state_split_fix.md`
- `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`

No production `src/` files were edited.

## Commands Run

- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/base_state_split_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/base_state_split_fix.py`
- `python -m json.tool proofs/v014/base_state_split_fix.json >/tmp/base_state_split_fix.validated.json`

## Proof Objects

- `proofs/v014/base_state_split_fix.json`
- `proofs/v014/base_state_split_fix.md`
- `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`

## Risks

- No native source fix landed.
- A production fix must reproduce WRF parent-interpolated/blended
  `HGT/MUB/PHB` and recompute `PB/MUB/PHB/theta_base` consistently.
- The h0-HGT formula check is validation-only and cannot be used as production
  logic.

## Handoff

Open a WRF live-nest base hook sprint. Capture or reproduce
`med_interp_domain` parent-interpolated `HGT/MUB/PHB`, `blend_terrain`, and
`start_domain_em` base recomputation before patching `build_replay_case`.
