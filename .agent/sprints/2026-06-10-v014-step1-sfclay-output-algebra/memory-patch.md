# Memory Patch

Summary: No memory-policy change required.

Reviewer Status: ACCEPT_NO_MEMORY_PATCH

The production change adds WRF-faithful scalar/column formulas to the existing
MYNN surface path and reconstructs WRF `phy_prep` density from already-resident
state and grid metrics. It does not add resident state, a new carry leaf,
dynamic-shape arrays, host/device transfers in the timestep loop, or a new GPU
memory claim.

The memory/FP32 lane remains unchanged:

- exact-branch memory preflight should be rerun only after the grid-parity
  candidate stabilizes;
- mixed FP32 R1/R2 remains blocked behind the fp64 correctness frontier;
- no new memory roadmap item is closed or opened by this sprint.
