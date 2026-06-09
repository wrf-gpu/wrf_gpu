# Reviewer Report

Decision:

Accept as a read-only memory research integration proof.

What the proof establishes:

- RRTMG column/band/optics tiling remains the only known memory blocker already
  fixed in the lineage.
- No remaining non-radiation memory item should block long validation after
  grid parity closes.
- The smallest safe memory-only source sprint is WDM6 `slmsk` shape-only
  cleanup, but it is opt-in and small (`0.075119 GiB` fp64 at 641x321x50).
- The only material bit-identical cleanup is moisture transport velocity reuse
  for active moisture advection, estimated at `0.237621-0.620881 GiB`
  source-static recoverable.

What remains measurement-first:

- MYNN BouLac dense matrices.
- Non-radiation whole-domain column tiling.
- Post-physics donated/sparse merge.
- Moisture limiter workspace.

What remains deferred:

- Acoustic carry split and FP32 acoustic until grid parity no longer confounds
  dycore debugging.
- PBL/surface semantic diagnostic threading unless grid attribution points
  there.
