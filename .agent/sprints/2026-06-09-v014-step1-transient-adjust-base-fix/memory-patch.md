# Memory Patch: V0.14 Step-1 Transient Adjust-Base Fix

Date: 2026-06-09

Reviewer Status: Pending. Opening sprint only.

Reason:

- The current-MUB split proof showed that `adjust_tempqv` needs transient
  post-`blend_terrain`/pre-`start_domain` current `MUB`, while final BaseState
  must stay post-`start_domain`.
- This sprint may change `src/gpuwrf/integration/d02_replay.py`; record exact
  proof result and next validation gate after close.

Expected memory after close:

- Whether Step-1 theta/QV residual closed, improved, had no effect, or blocked.
- Exact transient adjust-base and final BaseState guard values.
- Next manager decision for grid-parity chain.
