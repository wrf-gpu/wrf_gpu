# Sprint Contract — RRTMG OOM / XLA Autotune Triage

**Sprint ID**: `2026-05-28-rrtmg-oom-autotune-triage`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/rrtmg-oom-triage`
**Worktree**: `/tmp/wrf_gpu2_rrtmg`
**Wall-time**: 4-12 h (target ≤ 1 day)
**GPU usage**: YES
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint

The diagnostic harness 1h run requires radiation disabled to avoid OOM during XLA autotune. M13 worker also hit OOM during full pipeline 24h rerun. The OOM blocks every full-radiation 1h/24h validation run and therefore blocks every M12/M13/M14 close. The OOM is reproducible: ~922 MiB allocation request during RRTMG autotune.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs

1. `proofs/m13/radiation_diagnostic_augmentation.json` — M13 worker's note on RRTMG path
2. `proofs/m17/diagnostic_report_after_fix.json` — confirms 1h harness with radiation works only when cadence disabled
3. `src/gpuwrf/coupling/physics_couplers.py` — RRTMG adapter (post-M13)
4. `scripts/run_diagnostic_harness.py` — driver
5. JAX/XLA memory tuning env vars (XLA_PYTHON_CLIENT_PREALLOCATE, XLA_PYTHON_CLIENT_MEM_FRACTION, TF_GPU_ALLOCATOR=cuda_malloc_async, etc.)

## Acceptance

### AC1 — Root cause identified

`.agent/sprints/2026-05-28-rrtmg-oom-autotune-triage/root_cause.md`:
- Identify whether the OOM is (a) absolute VRAM shortfall, (b) XLA preallocation conflict, (c) autotune memory transient, (d) RRTMG kernel allocation pattern, or other
- Quantify: peak VRAM during autotune vs steady state; specific XLA pass triggering the allocation

### AC2 — Fix applied (one of):
- Reduce RRTMG kernel memory footprint (e.g. recompute intermediates instead of caching)
- Add autotune workspace cap via XLA flags
- Reshape RRTMG batching to fit
- Document a runbook for setting env vars at process start that fully avoids the OOM

### AC3 — Full-radiation 1h harness now runs

`taskset -c 0-3 python scripts/run_diagnostic_harness.py --hours 1 --jax-platform cuda --output proofs/rrtmg_triage/diagnostic_report_1h_full_radiation.json` SUCCEEDS. Output shows `rrtmg = ACTIVE` (radiation now firing within 1h).

### AC4 — Full-radiation 24h pipeline now runs

`taskset -c 0-3 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 24 --output-dir /tmp/rrtmg_triage_24h --proof-dir proofs/rrtmg_triage --run-root /mnt/data/canairy_meteo/runs/wrf_l3 --domain d02` SUCCEEDS (no OOM). 24 wrfouts produced.

### AC5 — 100-step parity preserved

`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES.

### AC6 — Speedup non-regression

Speedup measured during AC4 ≥ 14× vs M12 baseline.

### AC7 — Worker report

Standard format. Verdict `RRTMG_OOM_RESOLVED` if AC1-AC6 all pass; `RRTMG_OOM_PARTIAL` with explicit gaps otherwise.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--sandbox danger-full-access`. Coordinate with parallel M11.2 + M14 workers.
3. **Files writable**: `src/gpuwrf/coupling/physics_couplers.py` (RRTMG memory tuning only), `scripts/run_diagnostic_harness.py` (env var setup only), `proofs/rrtmg_triage/**`, `.agent/sprints/2026-05-28-rrtmg-oom-autotune-triage/**`.
4. **Files NOT writable**: dycore (M11.2 territory), surface/MYNN, BC, state contracts, governance, anything outside RRTMG memory tuning.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: rrtmg-triage DONE exit=$?" Enter`.
8. **End with verdict**: `RRTMG_OOM_RESOLVED` / `RRTMG_OOM_PARTIAL` + headline (full-rad-1h status).
