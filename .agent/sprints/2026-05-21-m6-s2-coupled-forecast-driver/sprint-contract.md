# Sprint Contract — M6-S2 Coupled Forecast Driver

**Sprint ID**: `2026-05-21-m6-s2-coupled-forecast-driver`
**Created**: 2026-05-21 12:35 by manager (Claude Opus 4.7 1M-context)
**Status**: STUB — dispatch after M6-S2a Opus ACCEPT
**Trigger**: M6-S1 closed UNBLOCKED-WITH-DEBT; M6-S2a (Gen2 accessor + d02 boundary replay + shared I/O) provides infrastructure. M6 plan critic ratified with amendments.

## Objective

First real coupled forecast driver on the GPU-resident state. Runs dycore + Thompson + MYNN + RRTMG + surface stub via M6-S1 coupling adapters, ingests IC from Gen2 `wrfinput_d02` and lateral BCs from M6-S2a boundary replay, produces 1h smoke → 6h → 24h forecast on the pinned d02 domain, with zero post-init host/device transfers and machine-readable spacetime budget.

## Acceptance (pre-M6-S3..S8 dispatch gate)

- **AC1 — Driver runs on real domain**. `scripts/m6_run_coupled_forecast.py` ingests d02 IC + d02 BC replay, runs forecast on 160×67×45 d02 domain, JAX `@jit`-compiled with `lax.scan` time-step.
- **AC2 — 1h smoke + 6h forecast + 24h forecast**. Three artifacts: `artifacts/m6/forecast_smoke_1h.json`, `forecast_6h_summary.json`, `forecast_24h_summary.json` (per `proof_schemas.ForecastSmoke` and `Forecast24h`). Output `wrfout_gpu_d02_*` files at +1h, +6h, +12h, +18h, +24h.
- **AC3 — Zero post-init transfer**. Transfer audit: `host_to_device_bytes_post_init = 0`, `device_to_host_bytes_post_init = 0`. Per-step temporary bytes properly measured (not literal — closes M6-S1 R-7 follow-up).
- **AC4 — Real GridSpec metrics (closes M6-S1 R-5)**. Replace `DEFAULT_DZ_M=100.0` in `physics_couplers.py:29` with real terrain-following `dz` array threaded from `GridSpec.eta_levels`. Adapter signature widened or `GridSpec` carried in state.
- **AC5 — Boundary-forcing State extension (closes M6-S1 R-13 / plan-critic amendment-3)**. Add `u_bdy, v_bdy, theta_bdy, qv_bdy` (and `ph_bdy` for vertical) State leaves; lateral-BC application at every time step from M6-S2a replay fixture. Per-step relaxation zone or specified-boundary nudge per WRF convention.
- **AC6 — FP32 storage policy ratified (closes M6-S1 R-3)**. Worker chooses ONE:
  - Path A: KEEP M6-S1's FP32 storage; M6-S2 forecast claims "operational-fitness-gated-on-M6-S7-RMSE" explicitly in worker report.
  - Path B: REVERT to FP64 storage for u/v/theta/qv; flip to FP32 only after M6-S7 RMSE gates pass. Worker amends ADR-010 §Decision.
  - **Recommendation**: Path A (simpler; preserves M6-S1 choice). Decision must be in worker-report §0.
- **AC7 — Radiation cadence robust (closes M6-S1 R-9)**. Forecast driver handles any total-step count, not just multiples of 10. Trailing-radiation-step decision documented.
- **AC8 — Honest accounting**. No `min(raw, cap)` fudge. Real temporary-bytes measurement (closes M6-S1 R-7). Debug-vs-stripped HLO diff = 0.
- **AC9 — Spacetime budget per d02**. `artifacts/m6/spacetime_budget_d02.json` with per-kernel + per-step wall + extrapolated 24h wall. Per-schema `Forecast24h.spacetime_budget`.
- **AC10 — ADR-010 amended**. Add §"M6-S2 amendments": Path A/B ratification, boundary-forcing leaves added, GridSpec threading.

## Inputs (from M6-S1 + M6-S2a)

- `src/gpuwrf/contracts/state.py` — extend with boundary-forcing leaves (AC5)
- `src/gpuwrf/coupling/physics_couplers.py` — thread GridSpec (AC4)
- `src/gpuwrf/io/gen2_accessor.py` — IC ingest from `wrfinput_d02`
- `src/gpuwrf/io/boundary_replay.py` — d02 BC fixture
- `src/gpuwrf/io/validation.py` — shared loaders
- `src/gpuwrf/io/proof_schemas.py` — `Forecast24h`, `ForecastSmoke`

## Files Worker May Modify

- `src/gpuwrf/contracts/state.py` (extend with boundary leaves, AC5)
- `src/gpuwrf/coupling/physics_couplers.py` (GridSpec threading, AC4)
- `src/gpuwrf/coupling/driver.py` (NEW — main forecast driver function)
- `src/gpuwrf/coupling/boundary_apply.py` (NEW — lateral BC application)
- `src/gpuwrf/profiling/budget.py` (real temporary-bytes measurement, R-7)
- `scripts/m6_run_coupled_forecast.py` (NEW — driver entry script)
- `tests/test_m6_forecast_smoke.py`, `test_m6_forecast_24h.py`, `test_m6_boundary_apply.py` (NEW)
- `.agent/decisions/ADR-010-coupled-state-extension.md` (amend with M6-S2 ratifications)
- Worker report

## Files Worker Must NOT Modify

- `src/gpuwrf/physics/**` — physics kernels FROZEN (M5-S1.y/S2.x/S3.x/S3.y closed)
- `src/gpuwrf/dynamics/**` — M4 frozen
- `src/gpuwrf/io/**` — M6-S2a OWNS; you USE
- Any other ADR or governance file
- `/mnt/data/canairy_meteo/**` — READ-ONLY

## Dispatch

- Primary worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory per sprint-lifecycle)
- Wall-time: **24-36 hours**
- Worktree: `/tmp/wrf_gpu2_m6s2` (NEW)
- Branch: `worker/codex/m6-s2-coupled-forecast-driver`

## Hard rules

- 24h forecast on REAL d02 domain (160×67×45) with REAL Gen2 IC + REAL boundary replay. No closed-domain shortcuts.
- Lateral-BC application physically correct per WRF convention (specified-boundary with relaxation zone).
- Real `dz` threading (no `DEFAULT_DZ_M=100`).
- FP32/FP64 policy explicit in worker report §0.
- ZERO post-init transfer (HARD CHECK).
- Use `gpuwrf.io.validation` for ALL Gen2 reads (per ADR-011).
- Use `proof_schemas` for ALL output JSON.

## Sequencing impact

M6-S3..S8 BLOCKED on this close.

## End-goal context

This is the first sprint where GPU-resident-coupled-physics produces a forecast that can be compared against the Gen2 CPU baseline. The wall-clock + transfer-audit numbers here feed M6-S5 ADR-007 4× verdict. The forecast quality feeds M6-S8 operational RMSE gate. Land this correctly.
