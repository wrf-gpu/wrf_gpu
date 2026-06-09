# Memory Patch

Scope:

Project-memory update for v0.14 memory planning after the empirical/static
memory map sprint.

Evidence:

- `proofs/v014/empirical_memory_map.json` reports
  `NO_REMAINING_NON_RADIATION_MEMORY_FIX_SHOULD_BLOCK_LONG_VALIDATION_AFTER_GRID_PARITY`.
- `validation.source_patterns_ok` is `true`.
- No candidate has `blocks_v014_long_validation_after_grid_parity=true`.
- Smallest safe memory-only source sprint is WDM6 `slmsk` shape-only cleanup.
- Only material bit-identical cleanup is moisture transport velocity reuse when
  active moisture advection is enabled.
- MYNN BouLac, non-radiation column tiling, post-physics merge, moisture
  limiter workspace, acoustic carry split, and FP32 acoustic remain
  measurement-first or grid-parity-gated.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-empirical-memory-map.md`. After
the next exact-branch memory preflight and grid-parity closure, condense into
`.agent/memory/stable/project-facts.md` or `.agent/memory/stable/recurring-gotchas.md`.

Reviewer Status:

Pending. Do not apply to stable memory until a post-grid-parity memory preflight
confirms this remains true on the validation branch.
