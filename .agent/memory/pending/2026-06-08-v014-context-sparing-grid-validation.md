# Memory Patch Proposal

## Scope

Project-memory update for manager/agent communication and v0.14 validation:
grid-cell comparison is the primary scientific validation artifact, while
human-facing reports and handoffs must stay compact to preserve context window
for long manager runs.

## Evidence

- The principal explicitly instructed that future validation should prioritize
  per-cell, per-time CPU-WRF-vs-GPU-WRF similarity over Canary station TOST as
  the expert-facing evidence.
- The principal explicitly requested context-window-sparing handoffs and
  top-level outputs while the manager coordinates many future sprints.
- `proofs/v014/grid_cell_envelope.md` and
  `proofs/v014/wind_mass_divergence_probe.md` show station summaries can hide
  broad grid-field divergence.

## Proposed Destination

`.agent/memory/stable/approved-patterns.md` after review.

## Patch

Proposed addition:

- For validation campaigns, make direct wrfout grid-cell comparison the primary
  scientific artifact and station TOST a complementary final gate. Human
  summaries and agent handoffs must stay compact: top-level verdict, top
  failures, proof paths, unresolved risks, and next action; detailed field tables
  belong in JSON/CSV artifacts.

## Reviewer Status

Reviewer Status: pending. Do not apply to stable memory until the v0.14 grid
comparison framework sprint is reviewed.
