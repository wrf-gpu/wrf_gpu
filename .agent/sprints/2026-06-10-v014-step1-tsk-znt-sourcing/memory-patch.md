# Memory Patch

Summary: No memory-policy change required.

Reviewer Status: ACCEPT_NO_MEMORY_PATCH

The production change replaces an inaccurate roughness/moisture-availability
surrogate with static WRF table lookups by `LU_INDEX`. It does not introduce new
resident prognostic state, new full-column or full-domain temporaries beyond
the existing lower-boundary arrays, host/device transfers in timestep loops, or
new GPU validation claims.

The memory/FP32 lane remains unchanged:

- proof-backed memory fixes are already merged;
- exact-branch memory preflight must be rerun on the final parity candidate;
- mixed FP32 remains blocked behind the open fp64 grid-parity frontier.

No roadmap memory item is closed or opened by this sprint. The next work is
correctness-only: localize the non-surface thermodynamic column inputs entering
`sfclay_mynn`.
