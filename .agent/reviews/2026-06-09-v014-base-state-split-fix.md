# Review: V0.14 Base-State Split Fix

verdict: `BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL`

objective: fix or precisely block the native live-nested d02 base-state split mismatch.

files changed:
- `proofs/v014/base_state_split_fix.py`
- `proofs/v014/base_state_split_fix.json`
- `proofs/v014/base_state_split_fix.md`
- `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`

commands run:
- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/base_state_split_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/base_state_split_fix.py`
- `python -m json.tool proofs/v014/base_state_split_fix.json >/tmp/base_state_split_fix.validated.json`

proof objects produced:
- `proofs/v014/base_state_split_fix.json`
- `proofs/v014/base_state_split_fix.md`
- `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`

unresolved risks:
- No native source fix was landed; live-nested child init still needs WRF parent interpolation/blend parity.
- The h0-HGT formula check is validation-only and cannot be used as a production dependency.
- A production fix will also need terrain/metrics/BaseState consistency, not PB/MUB replacement alone.

next decision needed: Instrument or port WRF live-nest initialization: capture/reproduce med_interp_domain parent-interpolated HGT/MUB/PHB, blend_terrain, and start_domain_em base recomputation before changing build_replay_case.
