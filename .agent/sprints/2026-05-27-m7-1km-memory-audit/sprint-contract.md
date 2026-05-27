# Sprint Contract — M7 1km Memory Audit

**Sprint ID**: `2026-05-27-m7-1km-memory-audit`
**Created**: 2026-05-27 (autonomous overnight loop)
**Status**: READY
**Predecessor**: `.agent/decisions/M7-PERF-MEASUREMENT-CLOSEOUT.md` (3km wall-clock and D2H invariant proven; 5.71s warm)

## Objective

Per M7 acceptance gate #8 (`MILESTONES.md` M7 + `.agent/milestones/M7-canary-operational-v0.md`), document the GPU memory footprint and operational gaps for the 1km Canary domain. The 3km mass grid is (44, 66, 159); the 1km equivalent is ~9× larger horizontally and the same vertical (~462 K cells → ~4.16 M cells per field). The RTX 5090 has 32 GB VRAM. The audit must determine: **does the 1km forecast fit? With what headroom? What are the bottlenecks?**

This sprint is **measurement-only**. Optimization, kernel-fusion, or precision-downcast decisions belong to a follow-up sprint guided by the audit findings.

## Acceptance

- **AC1 — Static memory model**: build a per-field memory accounting table covering all 45 state fields in `src/gpuwrf/contracts/state.py` × the 1km grid shape. Use the precision matrix (FP64/FP32) per field. Emit `.agent/sprints/2026-05-27-m7-1km-memory-audit/static_memory_model.json` with: field name, dtype, shape, bytes, running total. Include sanity check: total ≤ device VRAM.
- **AC2 — 1km grid shape derivation**: read the existing 3km Canary domain definition and derive the 1km grid shape. The 1km Canary nest already exists in the Gen2 backfill — read `wrfout_d04` or `wrfout_d05` shapes from one existing run (`/mnt/data/canairy_meteo/runs/wrf_l2/20260520_18z_l2_72h_20260521T045847Z/` per `cpu-wrf-baseline.md`). Emit `.agent/sprints/2026-05-27-m7-1km-memory-audit/grid_shape_1km.json` with: nx, ny, nz, mass shape, staggered shapes, source wrfout path, comparison vs 3km.
- **AC3 — Live VRAM probe (if 1km fits)**: attempt to construct a State at 1km shape via `gpuwrf.integration.d02_replay.build_replay_case` adapted for the 1km grid. Use `jax.devices()[0].memory_stats()` and `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` before/after construction. Emit `.agent/sprints/2026-05-27-m7-1km-memory-audit/live_vram_probe.json`. If construction OOMs, capture the failure point and emit `BLOCKED_OOM` with field-where-it-failed.
- **AC4 — Forecast feasibility probe (if AC3 passes)**: attempt one warm 1-RK-step forecast at 1km. Do NOT run the full 1h forecast. Measure: peak VRAM during step, step wall-time, transient buffer estimate (peak - resident). Emit `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`.
- **AC5 — Operational gaps**: write `.agent/sprints/2026-05-27-m7-1km-memory-audit/operational_gaps.md` listing in plain English: (a) does 1km fit yes/no; (b) headroom percentage; (c) bottleneck fields (top 5 by VRAM consumption); (d) downcast candidates with operational-impact reasoning (per `PRECISION_POLICY.md` + `feedback_validation_philosophy.md`); (e) kernel-fusion candidates that would reduce transients; (f) explicit "what would have to change" list to make 1km operational.
- **AC6 — Worker report** with verdict `FITS / FITS_WITH_HEADROOM / FITS_TIGHT / BLOCKED_OOM` and the operational-gap recommendation.

## Files Worker May Modify

- `scripts/m7_1km_memory_audit.py` (NEW — measurement orchestrator)
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/**`
- `tests/test_m7_1km_memory_audit.py` (NEW, OPTIONAL — pin the static model logic only; do NOT add tests that depend on hardware OOM behavior)

## Files Worker Must Not Modify

- `src/gpuwrf/**` — measurement-only; no model code change
- `src/gpuwrf/contracts/state.py`, `src/gpuwrf/contracts/precision.py` — these are the source of truth for the static model
- governance files
- `/mnt/data/canairy_meteo/**`
- existing 3km wall-clock measurement artifacts

## Hard Rules

1. **No model code changes.** Audit + measurement only.
2. **No assumption that 1km MUST fit** — produce honest BLOCKED_OOM if it doesn't, with field-level diagnosis.
3. **CPU pinning**: `taskset -c 0-3` for any Python process.
4. **Do not interfere with tmux `0:1`** (nightly WRF on cores 4-31).
5. **No remote push.** Local commit on `worker/gpt/m7-1km-memory-audit` only.
6. **Do not modify the 3km forecast path.** The 1km probe must construct its own State, not reuse the 3km operational_mode entry point unchanged. If the entry point requires a generalization, document it but do not implement the generalization in this sprint.
7. **GPU concurrency**: the nightly WRF doesn't use GPU. Run the probe on the RTX 5090 freely; tear it down between AC3 and AC4 to get clean baselines.

## Dependencies

- M7 perf-measurement step complete (commit `b7d9fe7`)
- RTX 5090 + driver matching `project_target_hardware.md`
- Gen2 1km wrfout files present at `/mnt/data/canairy_meteo/runs/wrf_l2/`

## Proof Objects

- `.agent/sprints/2026-05-27-m7-1km-memory-audit/static_memory_model.json` (AC1)
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/grid_shape_1km.json` (AC2)
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/live_vram_probe.json` (AC3)
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json` (AC4)
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/operational_gaps.md` (AC5)
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/worker-report.md` (AC6)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-6 h
- Branch: `worker/gpt/m7-1km-memory-audit`
- Worktree: `/tmp/wrf_gpu2_1kmaudit`
