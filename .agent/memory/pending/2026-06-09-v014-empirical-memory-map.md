# Pending Memory Patch: V0.14 Empirical Memory Map

Scope:

Project-memory update for v0.14 memory planning after the empirical/static
memory map sprint.

Evidence:

- `proofs/v014/empirical_memory_map.json` reports
  `NO_REMAINING_NON_RADIATION_MEMORY_FIX_SHOULD_BLOCK_LONG_VALIDATION_AFTER_GRID_PARITY`.
- `validation.source_patterns_ok` is `true`.
- No candidate has `blocks_v014_long_validation_after_grid_parity=true`.
- The smallest safe memory-only source sprint is WDM6 `slmsk` shape-only
  cleanup, but it is opt-in and small.
- The only material bit-identical cleanup is moisture transport velocity reuse
  when active moisture advection is enabled.
- MYNN BouLac dense matrices, non-radiation column tiling, post-physics merge,
  moisture limiter workspace, acoustic carry split, and FP32 acoustic remain
  measurement-first or grid-parity-gated.

Proposed destination:

After grid parity closes and the selected long-validation branch memory
preflight passes, add a concise stable-memory entry:

- Do not hold long validation for broad non-radiation memory rewrites after
  RRTMG column/band/optics tiling is present. Rerun the exact-branch memory
  preflight; if it fits, proceed. Treat WDM6 `slmsk` and moisture velocity reuse
  as optional bit-identical cleanups, and require measurement before MYNN/PBL,
  post-physics merge, limiter, acoustic carry, or FP32 memory work.

Reviewer Status:

Pending. Do not apply to stable memory until a post-grid-parity memory preflight
confirms this remains true on the validation branch.
