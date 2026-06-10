# Memory Patch

Summary: No memory-policy change required.

Reviewer Status: ACCEPT_NO_MEMORY_PATCH

The production change adds WRF-faithful scalar/column formulas to the
grid-backed surface-column view. It does not introduce a new resident state
contract, timestep-loop host/device transfers, dynamic-shape arrays, or a GPU
memory claim. The hydrostatic pressure reconstruction is a small fixed vertical
loop over existing state and grid metrics; it is a correctness change on the
surface path, not a broad memory optimization.

The memory/FP32 lane remains unchanged:

- exact-branch memory preflight should be rerun only after the grid-parity
  candidate stabilizes;
- mixed FP32 R1/R2 remains blocked behind the fp64 correctness frontier.

No memory roadmap item is closed or opened by this sprint.
