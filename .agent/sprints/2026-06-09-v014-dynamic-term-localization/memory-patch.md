# Memory Patch

Reviewer Status: no stable memory edit required from this sprint.

This sprint produced dynamic localization proof objects only. It did not change
runtime model code, memory behavior, RRTMG tiling, FP32 policy, validation
policy, or GPU runbook behavior. The durable project state is captured by the
roadmap/handoff updates:

- `PROJECT_PLAN.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Memory-relevant note for future managers: do not resume long validation runs
because this proof exists. The proof narrows the correctness search but leaves
the grid-field divergence unresolved. TOST, Switzerland equivalence, FP32
source landing, and additional memory optimization runs remain gated behind the
grid-parity/root-cause track.

If a future memory pending entry is needed, it should summarize only the
post-RK refresh lesson after the next sprint produces a green WRF surface or a
named JAX mismatch.
