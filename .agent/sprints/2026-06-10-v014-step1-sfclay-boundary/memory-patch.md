# Memory Patch

Summary: No memory-policy change required.

Reviewer Status: ACCEPT_NO_MEMORY_PATCH

The production change adds scalar first-step predicates and reuses existing
surface arrays. It does not introduce new resident state, new full-domain
temporaries, host/device transfers in timestep loops, or GPU validation claims.

Manager note: keep memory/FP32 long-lane decisions blocked behind the remaining
fp64 grid-parity frontier.

No roadmap memory item is closed or opened by this sprint. The only memory
follow-up is to rerun the exact-branch memory preflight after the TSK/ZNT parity
candidate lands, because this surface change is semantically active but not
expected to change peak allocation.
