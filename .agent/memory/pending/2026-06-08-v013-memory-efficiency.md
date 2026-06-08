# Memory Patch Proposal

## Scope

Project-memory update for v0.13 memory efficiency management after the 2026-06-08
641x321x50 VRAM probe and manager handoff. This patch is about recurring
operating behavior, not a one-off implementation detail.

## Evidence

- `.agent/decisions/V0130-ROADMAP.md` records the 2026-06-08 ~19:18 probe:
  the target 641x321x50 fp64 geometry is approximately 90 GiB and does not fit
  on a 32 GiB RTX 5090 because RRTMG materializes full-column-batch g-point
  transients.
- `proofs/v013/target_1km_vram_probe.json` identifies resident state as about
  2 GiB and the all-column RRTMG transient as the dominant memory consumer.
- The prior GPT analytic report at
  `.claude/worktrees/gpt-mem-map/.agent/reviews/2026-06-08-gpt-analytic-memory-map.md`
  classifies RRTMG column tiling as the immediate fix and lists additional
  recoverable memory issues by module.
- The principal elevated core-module memory efficiency to a v0.13 blocker before
  restarting TOST; TOST was stopped at 2/15 durable cases and must resume on
  memory-fixed code.

## Proposed Destination

`.agent/memory/stable/approved-patterns.md` and
`.agent/memory/stable/recurring-gotchas.md` after independent review approves
the generality and wording.

## Patch

Proposed addition to `approved-patterns.md`:

- For core GPU modules, memory efficiency is part of correctness-of-release:
  if a proof shows a full-domain temporary dominates VRAM and a bit-identical
  tiling/reordering fix exists, classify it before long validation campaigns
  and run TOST on the fixed code.

Proposed addition to `recurring-gotchas.md`:

- Column physics that batches every horizontal column can silently materialize
  full-domain spectral or vertical transients; compare against WRF's
  column-loop memory shape before treating VRAM OOM as a hardware limit.

## Reviewer Status

Reviewer Status: pending. Do not apply to stable memory until the RRTMG
column-tiling proof and refreshed memory-map report are reviewed.
