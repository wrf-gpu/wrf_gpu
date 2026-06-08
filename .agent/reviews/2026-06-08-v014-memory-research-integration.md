# v0.14 Memory Research Integration

Date: 2026-06-08
Analyst: GPT-5.5 xhigh
GPU use: none
Source edits: none

## Objective

Integrate the v0.13/v0.14 memory reports into a v0.14 memory-fix roadmap before
long validation runs. Separate already-proven bit-identical memory-layout work
from dycore/physics semantic work, and identify what must block validation.

## Integrated Conclusion

The current memory blocker is resolved: RRTMG leading-column tiling is landed
and GPU-proven. The earlier `target_1km_vram_probe` result showed the
641x321x50 target at about 89.3 GiB after band/taumol chunking but before
column tiling. The later column-tiling proof changes that decision surface:
LW untiled OOMs on a 32.11 GiB allocation, while LW tiled peaks at 5374.84 MiB;
SW tiled peaks at 1619.54 MiB.

The remaining memory items are not reasons to run long validation on stale code,
but they also are not blockers for the next grid-parity-first validation branch.
They should be sequenced as measured follow-ups after an exact-branch memory
preflight.

## Stale-Branch Reconciliation

The report `.agent/reviews/2026-06-08-gpt-memory-refresh.md` correctly warned
that if RRTMG column tiling was still pending, TOST should wait. That condition
is now closed by:

- `.agent/reviews/2026-06-08-gpt-rrtmg-column-tile.md`
- `proofs/v013/rrtmg_column_tile.json`
- `proofs/v013/rrtmg_column_tile_vram_suite.json`
- `PROJECT_PLAN.md` status text recording the fix as landed

Therefore the v0.14 roadmap treats RRTMG column tiling as already fixed, not as
pending work.

## Ranked Findings

1. **RRTMG full-column radiation transient** - fixed. This was the only true
   memory blocker. Keep the bit-identical proofs and run exact-branch preflight
   before any long validation branch.
2. **MYNN BouLac dense `(C,K,K)` arrays** - largest unmeasured static risk.
   Measure before rewriting.
3. **Acoustic scan carry and FP32 acoustic** - valuable but dycore-sensitive.
   Keep separate from grid-parity diagnosis and memory cleanup.
4. **Post-physics merge and non-radiation whole-column physics batches** -
   likely multi-GiB recoverable, but they need per-scheme proof gates.
5. **Moisture velocity reuse, WDM6 `slmsk`, dry masks, pad helpers** - small or
   default-inert cleanup. Do not delay validation for them.
6. **PBL/surface diagnostics threading** - memory benefit is secondary to a
   correctness/coupling question. Dispatch only as a PBL/surface correctness
   sprint.

## Files Changed

- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- `.agent/reviews/2026-06-08-v014-memory-research-integration.md`

No `src/` files were modified.

## Commands Run

- Read project controls and skills with `sed -n`:
  `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `PROJECT_PLAN.md`,
  `.agent/skills/maintaining-memory/SKILL.md`,
  `.agent/skills/updating-docs-minimally/SKILL.md`,
  `.agent/skills/managing-sprints/SKILL.md`.
- Read current v0.14 contracts:
  `.agent/sprints/2026-06-08-v014-fp32-acoustic-derisk/sprint-contract.md`,
  `.agent/sprints/2026-06-08-v014-grid-parity-attribution/sprint-contract.md`.
- Read memory and validation sources:
  `.agent/reviews/2026-06-08-gpt-memory-refresh.md`,
  `.agent/memory/pending/2026-06-08-v013-memory-efficiency.md`,
  `.claude/worktrees/gpt-mem-map/.agent/reviews/2026-06-08-gpt-analytic-memory-map.md`,
  `.codex/worktrees/v013-memory-refresh/.agent/reviews/2026-06-08-gpt-memory-refresh.md`,
  `.agent/reviews/2026-06-08-opus-1km-target-vram-measurement.md`,
  `.agent/reviews/2026-06-08-gpt-rrtmg-column-tile.md`,
  `.agent/reviews/2026-06-08-gpt-fp32-roi-and-v013-decision.md`,
  `.agent/reviews/2026-06-08-gpt-v014-fp32-status-freeze.md`,
  `.agent/reviews/2026-06-08-gpt-v013-impl-review.md`,
  `.agent/reviews/2026-06-08-opus-v013-oracle-integrity-audit.md`,
  `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`,
  `.agent/decisions/V0140-VALIDATION-PLAN.md`,
  `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`.
- Used `find` and `rg` to locate `.claude/.codex` reports and memory-related
  proof files.
- Used `jq` on:
  `proofs/v013/target_1km_vram_probe.json`,
  `proofs/v013/rrtmg_column_tile_vram_suite.json`,
  `proofs/v013/rrtmg_column_tile.json`,
  `proofs/v013/gpoint_chunk_rrtmg.json`,
  `proofs/v013/optics_taumol_chunk.json`,
  `proofs/v013/twoway_vram.json`,
  `proofs/v0120/nested_oom_fix.json`,
  `proofs/v013/pd_moisture.json`,
  `proofs/v013/moisture_advection_wiring.json`,
  `proofs/v014/fp32_acoustic_probes.json`.
- Used `git status --short`, `git log -1`, `git show --stat`, and final diff
  checks.

## Proof Objects Produced

- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- This integration review

No numerical proof, GPU run, profiler artifact, or source test was produced in
this report-only task.

## Unresolved Risks

- There is not yet a full 641x321x50 end-to-end forecast memory artifact after
  RRTMG column tiling; current confidence comes from targeted GPU VRAM suites.
- MYNN BouLac and non-radiation physics peaks remain static-source risks until
  an empirical memory map catches them in a real compiled step.
- Runtime impact of RRTMG column tiling in full forecasts still needs profiler
  evidence before performance claims.
- FP32 acoustic memory gain is arithmetic and CPU-probe evidence only, not a
  measured mixed-mode forecast result.
- The working tree has unrelated dirty files; this task intentionally ignored
  them.

## Next Implementation Sprints

1. Exact-branch memory preflight before any long validation campaign.
2. Empirical memory map for MYNN BouLac, non-radiation physics, post-physics
   merge, and moisture limiter liveness.
3. Small bit-identical cleanup only if it is adjacent and gated: moisture
   velocity reuse or WDM6 `slmsk`.
4. One non-radiation column-tiling pilot after measurement selects the scheme.
5. Separate FP32 acoustic R0/R1/R3 lane after grid-cell divergence is no longer
   the active root-cause blocker.
