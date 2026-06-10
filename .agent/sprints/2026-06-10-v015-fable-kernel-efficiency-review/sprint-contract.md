# Sprint Contract: V0.15 Fable Kernel Efficiency Review

Date: 2026-06-10 WEST
Owner: Fable/Mythos in tmux `0:1`
Manager: `worker/gpt/v013-close-manager`
Status: PREPARED ONLY. Do not dispatch until TOST and Switzerland compute are
running or complete, and the current v0.14 Fable Step-1 closure is finished.

## Objective

Perform a thorough read-only kernel efficiency review of wrf_gpu2 and produce a
ranked v0.15 action list for memory and compute gains.

This is an analysis sprint, not an implementation sprint. Do not change source
code. The endpoint is a complete, prioritized recommendation set for v0.15:
what major gains remain, why they exist, expected gain, complexity, risk, and
which proof gates would be required.

## Project Goal Anchor

The project goal is a möglichst vollständiger, möglichst schneller und
effizienter, möglichst WRF v4 treuer GPU rewrite. Optimizations are valuable
only when they preserve WRF-facing correctness, GPU scalability, and validation
honesty.

## Scope

Review every major kernel/module family:

- dynamics: RK3, acoustic small steps, pressure/geopotential/mass update,
  flux-form advection, limiters, boundary/halo paths
- radiation: RRTMG SW/LW tiling, optics, g-point loops, diagnostics
- PBL/surface/LSM: MYNN, BouLac, YSU/MYJ/MRF, surface-layer adapters,
  Noah/Noah-MP state handoff
- microphysics/cumulus: operational schemes and fail-closed/ref-only schemes
- coupling/runtime: `operational_mode.py`, scan adapters, post-physics merges,
  checkpoint/restart, wrfout writer
- validation/performance infrastructure: compile cache, AOT, transfer audits,
  Grid-Delta Atlas, TOST runners
- multi-GPU/sharding substrate

Use existing artifacts first:

- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
- `.agent/decisions/ADR-031-mixed-perturb-fp32-acoustic-DRAFT.md`
- `.agent/decisions/V0150-ROADMAP-DRAFT.md`
- `proofs/v014/empirical_memory_map.*`
- `proofs/v014/mythos_memory_fixes_260609.*`
- `proofs/v014/exact_branch_memory_preflight.json`
- `proofs/v013/rrtmg_column_tile_vram_suite.json`
- `proofs/v013/optics_taumol_chunk.json`
- `proofs/v013/gpoint_chunk_rrtmg.json`
- profiler or run artifacts under `proofs/` only as needed

## Required Scoring

For each candidate, record:

- module/family
- issue/inefficiency
- compute gain estimate: `none`, `low`, `medium`, `high`, or measured value
- memory gain estimate: `none`, `low`, `medium`, `high`, or measured GiB/MiB
- complexity: `low`, `medium`, `high`, `very_high`
- complexity drivers: files/contracts changed, proof gates, numerical risk,
  GPU/XLA risk, validation risk, chance that gain does not materialize
- correctness risk: how it could break WRF fidelity
- proof gates required before merge
- v0.15 recommendation: `do_first`, `do_if_measured`, `defer`, `reject`

Complexity is not just LOC. It includes:

- how many contracts and modules change
- whether restart, boundary, wrfout, or validation schemas change
- how much WRF oracle/savepoint evidence is needed
- whether the gain is already measured or only hypothesized
- whether the fix touches active high-risk numerics such as acoustic `P/PH/MU`
- whether XLA may already optimize the issue away

## Rules

- No source edits.
- No `git add`, no commit.
- No GPU jobs unless manager explicitly grants them after long validation
  compute is no longer at risk.
- Do not interrupt TOST, Switzerland, or Grid-Delta Atlas runs.
- Keep output context-sparing: dense tables, no long source excerpts.
- If a candidate is speculative, label it as speculative.
- Do not recommend weakening tolerances, clamping, CPU-WRF runtime dependency, or
  host/device transfers inside timestep loops.

## Deliverables

Write:

- `.agent/reviews/2026-06-10-v015-fable-kernel-efficiency-review.md`
- optional machine-readable table:
  `proofs/v015/kernel_efficiency_review.json`

The Markdown must include:

1. Verdict paragraph, max 120 words.
2. Ranked top action table, max 20 rows.
3. Rejected/low-value table, max 12 rows.
4. Suggested first 5 v0.15 sprints with proof gates.
5. Open measurements required before any source work.
6. Context-sparing handoff bullets for the manager.

Completion marker to manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'FABLE V015_KERNEL_EFFICIENCY_REVIEW DONE - see .agent/reviews/2026-06-10-v015-fable-kernel-efficiency-review.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
