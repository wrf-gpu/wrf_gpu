# Manager Closeout

Merge Decision: accept and land the empirical/static memory map.

Objective:

Produce an implementation-ready memory map for the remaining non-radiation
memory risks without source changes.

Accepted verdict:

`NO_REMAINING_NON_RADIATION_MEMORY_FIX_SHOULD_BLOCK_LONG_VALIDATION_AFTER_GRID_PARITY`.

Accepted evidence:

- `proofs/v014/empirical_memory_map.py`
- `proofs/v014/empirical_memory_map.json`
- `proofs/v014/empirical_memory_map.md`
- `.agent/reviews/2026-06-09-v014-empirical-memory-map.md`

Manager validation:

- Python compilation.
- CPU-only proof rerun.
- JSON validation.
- `git diff --check`.

Roadmap effect:

The v0.14 long-validation memory gate is now: after grid parity closes, rerun
the exact selected branch memory preflight and proceed if it fits. Do not block
that validation on broad memory rewrites. If a tiny safe memory-only source
sprint is desired before then, WDM6 `slmsk` shape-only cleanup is the smallest
candidate. Moisture velocity reuse is the only material bit-identical cleanup,
but only matters when active moisture advection is enabled.

Next decision:

Keep current engineering focus on grid-cell parity and the Pre-RK input-boundary
truth sprint. Defer MYNN/PBL/acoustic semantic memory changes until measurement
or grid attribution makes one binding.
