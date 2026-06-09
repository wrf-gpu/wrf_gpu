# Memory Patch

Reviewer Status: no stable-memory or rule change required from this sprint.

This was a CPU-only grid-parity localization sprint. It did not change
production model source, memory behavior, precision behavior, GPU launch
mechanics, or manager operating rules. The relevant durable knowledge is already
recorded in the proof and manager closeout:

- WRF `first_rk_step_part2` creates the material Step-1 `T_TENDF` residual via
  active raw dry physics source leaves in `update_phy_ten`.
- The current JAX dry source bundle is missing equivalent `RTHRATEN`/`RTHBLTEN`
  source leaves.
- Aggregate post-physics state-delta is rejected as a narrow replacement.

This should be carried forward in the v0.14 handoff/roadmap, not in stable
memory, until the implementation sprint proves the production fix.
