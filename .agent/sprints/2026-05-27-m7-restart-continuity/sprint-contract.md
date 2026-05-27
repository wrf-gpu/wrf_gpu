# Sprint Contract — M7 Restart-Continuity Test

**Sprint ID**: `2026-05-27-m7-restart-continuity`
**Created**: 2026-05-27 (autonomous overnight loop)
**Status**: READY
**Predecessor**: `.agent/decisions/M7-PERF-MEASUREMENT-CLOSEOUT.md` (M7 perf done); 1km audit FITS_WITH_HEADROOM

## Objective

Per M7 acceptance gate #3 (`.agent/milestones/M7-canary-operational-v0.md`): **N-step → checkpoint → restart → N-step compare within Tier-1 tolerance.** Operational daily-run capability requires a robust restart story. Debug crashes can't lose a forecast.

The test: run N RK steps, serialize the full State + namelist + grid + RNG-like reproducibility state to a checkpoint file, kill the JAX process, restart from the checkpoint, run another N RK steps, and compare against a reference forecast that ran 2N steps in a single process. Match within Tier-1 tolerance (1e-12 absolute or bitwise where possible).

This sprint implements the checkpoint + restart path. It does NOT need to produce WRF-format wrfrst — that's the NetCDF-writer sprint's territory. A plain `.npz` or `.pkl` checkpoint is sufficient for this gate.

## Acceptance

- **AC1 — Checkpoint writer**: implement `gpuwrf.runtime.checkpoint.write_checkpoint(state, namelist, grid, step_index, path)`. Stores all 47 State fields + namelist parameters + grid metadata + step counter. Use `pickle` for the State pytree (round-trip via `jax.tree.map(np.asarray, state)` if needed). Must round-trip: `read(write(x)) == x` bitwise on a synthetic State.

- **AC2 — Checkpoint reader**: implement `gpuwrf.runtime.checkpoint.read_checkpoint(path) → (state, namelist, grid, step_index)`. Reconstructs State as a JAX pytree on the device after host-side load.

- **AC3 — Restart-continuity probe**: write `scripts/m7_restart_continuity_probe.py` that:
  1. Loads one of the 3 V3 ICs (20260521 by default).
  2. Runs 2N RK steps in process A; saves final state as REFERENCE.
  3. Restarts the script via subprocess: process B1 runs N steps, writes checkpoint, exits cleanly.
  4. Subprocess B2 reads checkpoint, runs N more steps, writes final state.
  5. Compares B2_final vs REFERENCE field-by-field; emits PASS if max delta ≤ 1e-12 for FP64 fields and ≤ 1e-6 for FP32 fields.
  Use N=10 for the standard probe; provide a `--n-steps` override.

- **AC4 — Wall-time overhead**: measure checkpoint write + read overhead. Emit `.agent/sprints/2026-05-27-m7-restart-continuity/restart_overhead.json` with: write time, read time, total restart overhead, % of N-step forecast time. The overhead is **informational**, not a gate (no fixed threshold).

- **AC5 — Proof object**: emit `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json` with: N, IC, REFERENCE wall-time, B1+B2 wall-times, per-field max delta + threshold, verdict (PASS / FAIL / BLOCKED).

- **AC6 — Tests**: add `tests/test_m7_restart_checkpoint_roundtrip.py` with unit tests for the writer/reader round-trip on synthetic state (no GPU forecast).

- **AC7 — Worker report** with verdict.

## Files Worker May Modify

- `src/gpuwrf/runtime/checkpoint.py` (NEW)
- `src/gpuwrf/runtime/__init__.py` (export checkpoint helpers if appropriate)
- `scripts/m7_restart_continuity_probe.py` (NEW)
- `tests/test_m7_restart_checkpoint_roundtrip.py` (NEW)
- `.agent/sprints/2026-05-27-m7-restart-continuity/**`

## Files Worker Must Not Modify

- `src/gpuwrf/runtime/operational_mode.py` — checkpoint is a sibling module, not a fork of operational mode
- `src/gpuwrf/contracts/state.py` — State schema unchanged
- `src/gpuwrf/dynamics/**`, `src/gpuwrf/physics/**`, `src/gpuwrf/coupling/**` — no model code change
- governance files
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **No model code changes.** Checkpoint + restart only.
2. **GPU use**: 1km audit JUST finished; only the NetCDF writer sprint runs in parallel (CPU-only). This sprint can use GPU freely.
3. **CPU pinning**: `taskset -c 0-3` for any Python process.
4. **Do not interfere with tmux `0:1`** (nightly WRF).
5. **No remote push.** Local commit on `worker/gpt/m7-restart-continuity` only.
6. **Tolerance gates**: 1e-12 for FP64 fields, 1e-6 for FP32 fields. Stricter than M6 Tier-1; deterministic restart should be ~bitwise unless JAX/XLA scheduler non-determinism intrudes. Document any non-bitwise deltas with reasoning.
7. **No WRF-format wrfrst** — that is the NetCDF writer sprint. A plain pickle/npz checkpoint is sufficient.

## Dependencies

- M7 perf-measurement step complete (commit `b7d9fe7`)
- 1km audit confirmed VRAM headroom (commit `7907d7b`)
- 20260521 IC (V3 pinned set) + Gen2 backfill present

## Proof Objects

- `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json` (AC5 — gate)
- `.agent/sprints/2026-05-27-m7-restart-continuity/restart_overhead.json` (AC4 — informational)
- `.agent/sprints/2026-05-27-m7-restart-continuity/worker-report.md` (AC7)
- `tests/test_m7_restart_checkpoint_roundtrip.py` (AC6)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 3-6 h
- Branch: `worker/gpt/m7-restart-continuity`
- Worktree: `/tmp/wrf_gpu2_restart`
- GPU usage: YES (forecast runs)
