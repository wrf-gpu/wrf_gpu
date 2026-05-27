# Sprint Contract — M7 Corpus Bridge (bounded Option A + run-dir rebind)

**Sprint ID**: `2026-05-27-m7-corpus-bridge`
**Created**: 2026-05-27 (autonomous overnight loop, post M7 close)
**Status**: READY — small surgical follow-up after M7-OPERATIONALLY-CLOSED
**Predecessors**:
- `.agent/decisions/MILESTONE-M7-CLOSEOUT.md` (M7 operationally closed; gates #1+#7 partial)
- `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/recommendation.md` (Option A bridge + DEFAULT_M6_GEN2_RUN_DIR rebind)

## Objective

Apply the bounded Option A bridge recommended by the Gen2 corpus scout: lower the M7 Tier-4 harness floor from N=10 to N=5 with a probationary `--non-operational` tag. Also rebind `gen2_accessor.DEFAULT_M6_GEN2_RUN_DIR` from the stripped (wrfout-empty) cycle `20260520_18z_l3_24h_20260521T045847Z` to a surviving complete cycle.

This unblocks M7 gates #1 + #7 on probationary tolerance for THIS WEEK while operator-level Option D (Gen2 retention flip + 5-7 targeted replays) grows the corpus over the next 2-4 nights. Small, surgical, low-risk.

## Acceptance

- **AC1 — Tier-4 harness floor lowered to N=5 probationary**: in `scripts/m7_run_tier4_rmse_harness.py` and `src/gpuwrf/validation/tier4_rmse_harness.py`:
  - Add a `--non-operational` flag (default false) that, when true, lowers the corpus floor from 10 to 5.
  - When triggered, the harness emits `status="PASS_PROBATIONARY"` (instead of `PASS`) and stamps `"corpus_size_class": "bounded"` + `"M7_close_class": "probationary"` in the artifact. Default behavior (no flag) preserves the original `BLOCKED_CORPUS` outcome on small corpora.
  - Update existing pinned tests so they still pass; add a new test that exercises the `--non-operational` path with a fixture of N=5 synthetic members.

- **AC2 — DEFAULT_M6_GEN2_RUN_DIR rebind**: in `src/gpuwrf/io/gen2_accessor.py`:
  - Find the constant `DEFAULT_M6_GEN2_RUN_DIR` currently pointing to `20260520_18z_l3_24h_20260521T045847Z`.
  - Verify that target is wrfout-empty (per scout: it is) by listing the dir.
  - Rebind to `20260521_18z_l3_24h_20260522T133443Z` (one of the 3 pinned-grid-complete cycles per `2026-05-27-m7-gen2-corpus-scout/full_gen2_inventory.json`). Confirm by reading the dir lists wrfout_d02 hourly files.
  - Update any pinned reference in tests to the new constant.

- **AC3 — Regression test for the rebind**: add `tests/test_m7_default_gen2_run_dir.py` that asserts `DEFAULT_M6_GEN2_RUN_DIR` points to a path with **non-zero `wrfout_d02_*` files** when run on the host (auto-skip when `/mnt/data/canairy_meteo` absent, matching the corpus-scout test pattern).

- **AC4 — Probationary smoke**: execute the harness with `--non-operational` against the existing N=3 corpus and confirm `PASS_PROBATIONARY` + non-empty member_split + finite rmse_records skeleton. With N=3 < N=5, this should still emit `BLOCKED_CORPUS` (the floor of 5 is not yet met by surviving runs), and a `PASS_PROBATIONARY_PENDING` status with a clear "needs +2 members" message. Verify behavior; emit `.agent/sprints/2026-05-27-m7-corpus-bridge/probationary_smoke.json`.

- **AC5 — All existing tests pass**: full M7 test suite (`tests/test_m7_*`) must continue green. Use `taskset -c 0-3 pytest -q tests/test_m7_*.py`.

- **AC6 — Worker report** with verdict `BRIDGE_READY` / `PARTIAL` / `BLOCKED`.

## Files Worker May Modify

- `src/gpuwrf/validation/tier4_rmse_harness.py` (add `--non-operational` mode)
- `src/gpuwrf/io/gen2_accessor.py` (rebind DEFAULT_M6_GEN2_RUN_DIR)
- `scripts/m7_run_tier4_rmse_harness.py` (forward the flag)
- `tests/test_m7_tier4_rmse_harness.py` (add bounded-mode test; do not remove existing)
- `tests/test_m7_default_gen2_run_dir.py` (NEW)
- `.agent/sprints/2026-05-27-m7-corpus-bridge/**`

## Files Worker Must Not Modify

- `src/gpuwrf/validation/data_quality.py` (the M6.5-D1 RMSE adapter is FROZEN — don't touch)
- `src/gpuwrf/dynamics/**`, `src/gpuwrf/physics/**`, `src/gpuwrf/coupling/**`
- `src/gpuwrf/runtime/**` — pipeline + checkpoint + operational_mode are frozen post-M7
- `src/gpuwrf/io/wrfout_writer.py` — frozen
- `src/gpuwrf/validation/forecast_vs_obs.py` — frozen
- governance files
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **No model code changes.** Harness flag + accessor constant + tests only.
2. **Preserve operational-default behavior**: without `--non-operational`, the harness must produce the same artifact as before (BLOCKED_CORPUS on small corpora).
3. **CPU pinning**: `taskset -c 0-3`.
4. **No GPU runtime.** Pure schema + flag + tests.
5. **Do not interfere with tmux `0:1`** (nightly WRF).
6. **No remote push.** Local commit on `worker/gpt/m7-corpus-bridge` only.

## Proof Objects

- `.agent/sprints/2026-05-27-m7-corpus-bridge/probationary_smoke.json` (AC4)
- `.agent/sprints/2026-05-27-m7-corpus-bridge/worker-report.md` (AC6)
- `tests/test_m7_default_gen2_run_dir.py` (AC3)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 1-2 h (small surgical sprint)
- Branch: `worker/gpt/m7-corpus-bridge`
- Worktree: `/tmp/wrf_gpu2_bridge`
- GPU usage: NONE

## What this enables

After this sprint:
- M7 gates #1 + #7 elevated from PARTIAL to PROBATIONARY-PASS at N=5 (when corpus grows to ≥5 in-window pinned-grid-complete members)
- Operator-level Option D becomes additive (corpus grows; bridge mode optional)
- Latent `DEFAULT_M6_GEN2_RUN_DIR` bug eliminated
- M7 close becomes upgradable from OPERATIONALLY-CLOSED to FULLY-CLOSED once 2 more cycles backfill
