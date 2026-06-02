# v0.2.0 EQUIVALENCE TOST — Consolidation, End-Gate & Methodology Findings

**Branch:** `worker/opus/v020-tost` (off `worker/opus/v020-integration` @ d6ce779)
**Author:** Opus MAX consolidation+validation engineer
**Date:** 2026-06-02
**Status:** consolidation DONE, end-gate 2/2 PASS, harness BUILT + pointed at the
parquet corpus, smoke run in flight. **STOP before full campaign — manager
reviews methodology + smoke first.**

---

## 1. CONSOLIDATION

Fresh branch `worker/opus/v020-tost` from `worker/opus/v020-integration` @ d6ce779.
Merged the two FORECAST-AFFECTING validated branches:

| Branch | Tip | Forecast code touched | Merge |
|---|---|---|---|
| `worker/opus/l1-rad-time` | 5afd098 | `src/gpuwrf/runtime/operational_mode.py` (+58) | clean, no conflict |
| `worker/opus/p1-5-thompson` | 30a7850 | `src/gpuwrf/physics/thompson_column.py` (+108), `thompson_constants.py` (+10) | clean, no conflict |

File-disjoint as predicted (operational_mode.py vs thompson_*.py). `import gpuwrf`
clean from the worktree src. KF / nesting / conservation / P0-5 / boundary /
surface-w **EXCLUDED** (do not affect the d02/d03 replay forecast, or are
unsmoked/failed smoke).

## 2. GPU END-GATE — 2/2 PASS (no dycore regression)

Idealized close gate (`tests/idealized/test_dycore_close_gate.py`, `-m close_gate`,
fp64, taskset 0-3, detached on the free GPU, 142 s) on the consolidated branch:

| Case | Verdict | Sub-checks |
|---|---|---|
| warm bubble (Skamarock) | **PASS** | all 6 pass (finite, drift, max|w|, mass drift, thermal rise, theta') |
| density current (Straka) | **PASS** | all 6 pass (finite, front pos, max|w|, mass drift, rotor proxy, theta'min) |

Proof: `proofs/sprintU/close_gate/{warm_bubble,density_current}_verdict.json`
(byte-identical to the committed deterministic verdict → reproducibility confirmed;
the L1+Thompson merge did NOT perturb the dycore).

## 3. TOST METHODOLOGY (predeclared before scoring)

### Truth source — the 31-case point-shadow PARQUET corpus
`/mnt/data/canairy_meteo/artifacts/datasets/wrf_case_bank/_all_cases_point_shadows.parquet`
- **32 case_ids, 4 nests (d02, d03, d04, d05), 278 stations, 73 lead-h (d02) /
  25 lead-h (nests), 0% NaN on `wrf_phase14c_{t2_k,u10,v10}`.**
- CPU-WRF values are `wrf_phase14c_*`, interpolated to stations by
  `interpolation_method == nearest_grid_cell`, with the CPU run's
  `nearest_grid_iy`/`nearest_grid_ix` stored per row.
- This is the CPU-WRF station SHADOW. It is NOT the purged full-grid wrfout and
  NOT a 3-case manifest. It is the real CPU-WRF truth at the stations.

### Pairing (non-gameable, ADR-029 structure)
- The GPU forecast (consolidated dycore+physics, same `_advance_chunk` loop as the
  v0.1.0 d02 validator) emits a full-grid wrfout per lead; the scorer samples
  T2/U10/V10 at the **SAME `(nearest_grid_iy, nearest_grid_ix)` cell the parquet
  CPU extraction used** → identical nearest-grid-cell rule on both sides → the
  delta is purely model-vs-model.
- Complete-pair on `(station_id, valid_time_utc, var)`; complete-pair deletion
  (both sides finite). Per case × domain × **lead-block** (0-24/24-48/48-72 h) ×
  variable; blocks with `< 30` pairs are EXCLUDED (ADR-029 threshold).

### Statistic + TOST
- Per (case, block, var): `mean_bias = mean(GPU-CPU)`, `repro_rmse =
  sqrt(mean((GPU-CPU)^2))`.
- Per-case TOST delta for a variable = lead-block-averaged `repro_rmse`.
- TOST (α=0.05) on the per-case `repro_rmse` against the **FROZEN ADR-029
  margins** — equivalence iff the per-case reproduction error is statistically
  below the predeclared band for **all three** variables.

### Margins — FROZEN, ADR-029, not loosened
| Var | Margin | Source |
|---|---|---|
| T2 | ±0.2148692978020805 K | ADR-029 (10% of CPU-WRF RMSE 2.1487 K) |
| U10 | ±0.23064713972582307 m/s | ADR-029 (10% of 2.3065 m/s) |
| V10 | ±0.2752320537920854 m/s | ADR-029 (10% of 2.7523 m/s) |

### Why DIRECT GPU-vs-CPU, not RMSE-vs-obs
ADR-029's nominal delta is `RMSE_GPU_vs_obs − RMSE_CPU_vs_obs`, needing a common
OBS reference. The parquet has NO obs (obs live in the AEMET station parquets,
largely **daily** for the high-quality pool; only ~106 of the 278 stations have
hourly obs). The parquet's value IS the CPU-WRF point shadow. The defensible,
lossless, full-278-station equivalence question the parquet answers is the
**direct** one: *does the GPU reproduce CPU-WRF at the stations within the
ADR-029 band?* We apply the frozen ADR-029 margins to the direct GPU−CPU paired
statistic and keep the ADR-029 paired STRUCTURE. No self-compare (the CPU side is
real CPU-WRF, the GPU side is the independent JAX port), no synthetic truth, no
margin fudging.

## 4. THE LOAD-BEARING n-CAP FINDING (manager decision needed)

**The parquet truth is n=31, but the GPU REPLAY forecast cannot be initialised
for most of them.** `gpuwrf.integration.d02_replay.build_replay_case` needs the
**t=0 `wrfout` snapshot** (initial condition) **+ the parent hourly `wrfout`
boundary history** (≥2 frames for the lateral replay). The nightly pipeline
PURGED wrfout from all but:
- **d02 (wrf_l2): 3 cases** — `20260509` (73 frames), `20260521` (20 frames),
  `20260530` (73 frames).
- **d03 (wrf_l3): 8 dirs retain wrfout_d03**, but only `0509`/`0521`/`0530` have
  the full 25 frames; the rest are 2–9 frame partials.

So the **scoreable GPU-side n is init-capped at ~3 full d02 cases** (+ a few
partial d03), NOT 31 — UNLESS CPU-WRF is re-run from the retained `met_em`
forcing (present in all dirs) to regenerate wrfout. This is the same gap the
corpus memory flagged ("re-run ~12 cases for powered n≥15").

**This means: the parquet upgrades the TRUTH SOURCE quality (real 278-station
CPU-WRF shadow, 0% NaN) but does NOT by itself unlock a powered n=31 TOST.** The
honest v0.2.0 claim at GPU-runnable n≈3 is a **single-season MAM point estimate /
descriptive reproduction**, NOT a powered (n≥15) seasonal equivalence. To reach
the ADR-029 powered target the manager must schedule the CPU-WRF re-runs from
met_em (principal CPU decision, ~nights), then re-run this harness with the
regenerated wrfout — the harness already scores against the full 31-case parquet.

Additional honesty note on the runnable set:
- `0509` d02 carries only **40 parquet stations** ("old config" per corpus memory)
  vs `0530`'s full 278; both still clear the ≥30-pair-per-block floor.
- `0521`'s parquet `case_id` is the `_l2rerun` variant whose `wrf_workdir` does
  **NOT** match the retained `wrf_l2` dir → `(iy,ix)` alignment unverified →
  EXCLUDED from the clean campaign by default (override only after a grid re-check).

## 5. SMOKE (2 cases) — see `proofs/m20/tost_run/parquet_smoke/tost_parquet.json`

Cases `20260509_18z__d02` + `20260530_18z__d02` (both `parquet_workdir_match=True`,
72 scoreable leads each). *[numbers filled in by the smoke run; see the JSON]*

## 6. EXACT FULL-RUN COMMAND (manager launches after approval)

The full campaign is GPU-init-capped (see §4). Until CPU-WRF backfill, the
"full run" over the runnable d02 set is:
```
PYTHONPATH=src OMP_NUM_THREADS=4 JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  taskset -c 0-3 python3 proofs/m20/tost_parquet_runner.py --execute \
    --units 20260509_18z__d02 20260530_18z__d02 \
    --out-dir proofs/m20/tost_run/parquet_campaign
```
(0521 omitted — workdir/grid alignment unverified.) For the POWERED n≥15 campaign,
backfill wrfout via CPU-WRF re-run from met_em, register the new run dirs in
`RUNNABLE_UNITS`, then re-run the same command with the expanded `--units`.
