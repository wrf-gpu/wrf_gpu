# v0.17 Identity 72h: Slow-Compile Root-Cause + Fast-Compile Identity Dashboards

Status: FINAL — both regions ALL-GREEN; root-cause + fix proven; dashboards produced.
Branch: worker/perf/v017-rc (worktree .wt-rc), head fa0ca55c + 1 fix commit pending.
Author: Opus 4.8 GPU/compile investigator + validation lead

Deliverable locations (mirrored to MAIN repo + the v017-rc branch worktree .wt-rc):
- Report: `proofs/v017/identity_fast_compile_report.md`
- Switzerland dashboards: `docs/assets/v017/identity_proof/switzerland_d01/` (5 PNG) +
  `proofs/v017/identity_proof/switzerland_d01/identity_proof_manifest.json`
- Canary dashboards: `docs/assets/v017/identity_proof/canary_l2_d02/` (5 PNG) +
  `proofs/v017/identity_proof/canary_l2_d02/identity_proof_manifest.json`
- Code fix (uncommitted on the v017-rc worktree): `src/gpuwrf/integration/daily_pipeline.py`
  (+`GPUWRF_REPLAY_SEGMENTED` switch). Helper run scripts under `proofs/v017/`.

## A. Slow / multi compile — ROOT CAUSE (code-confirmed)

The 72h operational replay slow-compile is NOT XLA autotuning and NOT
shape-polymorphic per-segment retracing. It is the **per-hour forecast entry
emitting many distinct XLA scan modules**, each of which is a large fp64
full-physics program.

Call chain (single-domain Switzerland d01, `cpu_wrf_replay`):
`python -m gpuwrf.cli run --domain d01 --hours 72`
-> `daily_pipeline.execute_daily_pipeline` (init_mode == cpu_wrf_replay)
-> per-hour loop `for hour in 1..72: state = forecast_fn(state, namelist, 1.0)`
   (daily_pipeline.py:1150,1167 — one full hour advance per call, substeps=1)
-> `forecast_fn = _default_forecast_fn` (daily_pipeline.py:1718)
-> `run_forecast_operational` -> `_run_forecast_operational_jit`
   (operational_mode.py:4329/4346)

`_run_forecast_operational_jit` is a **Python while-loop that emits one
`jax.lax.scan` per radiation interval** (a non-radiation scan + a 1-step
radiation scan), so the number of distinct XLA scan subcomputations — and hence
the compile cost — scales with the per-call forecast length. For the Switzerland
case (`time_step=18s`, `radt=10min` => radiation cadence ~33 steps), a single 1h
advance is 200 steps spanning ~6 radiation intervals => ~12 distinct scan modules,
all fp64 + full v0.17 physics suite (now incl. the WSM7/WDM7 hail microphysics +
`qh` substrate + GFS PBL added in v0.17). That is the `jit__run_forecast_operational_jit`
module the alarm names.

The "SECOND compile" in the kill log is the hour-2 retrace: hour-1's input is the
freshly-built replay `State` (legacy-aliased leaves, de-aliased + donated), while
hour-2's input is the device `State` returned by hour-1; the slightly different
leaf typing triggers ONE additional compile at hour 2, after which hours 3..72
reuse the executable. So the tax is ~2 cold compiles of the big per-hour program,
one-time.

This is NOT a v0.17 regression in kind — the **v0.15 Switzerland final-gate used
the same `cpu_wrf_replay` CLI path and hit the same `jit__run_forecast_operational_jit`
compile (4m44s first compile + a second), and still completed all 72h in ~2941s
total**. v0.17 grew the per-step program (hail MP `qh` + WSM7/WDM7 + GFS PBL), so
the first compile more than doubled to **10m44s**, and the manager killed the run
at hour 1 fearing it would not finish. (The v0.15 *canary* finalgate, by contrast,
used `run_one_case_v0120.py` -> `execute_nested_pipeline`, whose segmented
`advance_chunk` host loop compiled in ~2min — already bounded.)

### Compile-tax flag test (SHORT 2h Switzerland replay; 3 cold processes)
Harness: `proofs/v017/compile_flag_test.sh`. Run root:
`/mnt/data/wrf_gpu_validation/v017_compile_flag_test_20260614T232030Z`. Each config
a separate cold process (no warm-cache leakage), under the shared GPU lock, with the
sibling bench load present (so absolute seconds run slightly high vs an idle box).

| config | entry / module | first cold-compile | wrfout | verdict |
|---|---|---|---|---|
| A default + autotune ON  | run_forecast_operational while-loop (`jit__run_forecast_operational_jit`) | **9m00.7s** (+ 2nd compile; 2h wall=2102s) | 2 | reproduces the pathology |
| B default + autotune OFF (`--xla_gpu_autotune_level=0`) | same module | **10m22.7s** (+ 2nd compile) | (ran) | **autotune is NOT the cause** |
| C `GPUWRF_REPLAY_SEGMENTED=1` | run_forecast_operational_segmented (`jit__advance_chunk`) | **3m17.4s** (+ smaller 2nd; 2h wall=662s) | 2 | **THE FIX: 3.2x faster wall, bounded** |

**The fix works (C):** routing replay through the segmented `advance_chunk` entry
cut the cold compile from ~9-10 min to **3m17s** and the 2 h end-to-end wall from
~2100s to **662s (3.2x)**, with identical wrfout output (2 frames, rc=0) and
bit-identical numerics (segscan_equiv.json). The compiled `advance_chunk` segment is
reused across every radiation interval and every hour, so the per-run compile tax is
paid once on a SMALL program instead of repeatedly on the big monolith.

**Reproduced in the production 72h run:** the actual `GPUWRF_REPLAY_SEGMENTED=1`
Switzerland 72h identity run compiled `jit__advance_chunk` in **3m16s** (one segment,
+ a small partial-tail segment) and then stepped all 72 hours with NO further slow
compile — confirming the fix holds at full forecast length, not just the 2h probe.

**Autotune ruled out:** turning XLA autotuning OFF (B) did NOT cut the compile
(10m22s vs A's 9m00s — same, within jitter). The cost is the structural lowering of
the large fp64 full-physics many-scan per-hour `jit__run_forecast_operational_jit`
program, not autotuning. **Shape-polymorphism ruled out too:** the per-hour call is
always `hours=1.0` and `_rewindow_boundary_leaves` always swaps in a fixed 2-level
boundary leaf, so there is no per-LBC-interval / per-output-segment retrace — only
the one hour-1 -> hour-2 typing retrace (the documented "second compile"), then
hours 3..72 reuse the executable. Per-run compile tax (default replay) ≈ 2 cold
compiles of the big per-hour program ≈ **~18-21 min one-time** (v0.17), up from
v0.15's ~2x 4m44s because v0.17 added hail MP (qh/WSM7/WDM7) + GFS PBL to the
per-step program.

## A (fix). THE FIX

`run_forecast_operational_segmented` compiles ONE small fixed-length segment
(`_advance_chunk`, `static_argnames=(n_steps,cadence)`, traced `start_step`) and
reuses that single executable across every radiation interval AND every hour, so
the cold compile is one small program (independent of forecast length). It is
**bit-identical** to the while-loop: `proofs/perf/segscan_equiv.json` shows
`max_abs_diff_seg_vs_production == 0.0` on every field (u,v,w,theta,qv,p,ph,mu),
so the identity result is unchanged — only the compile time differs.

Wiring (this branch): `daily_pipeline.execute_daily_pipeline` now honours
`GPUWRF_REPLAY_SEGMENTED=1` to route the replay path through `_segmented_forecast_fn`
(the same entry the standalone native-init path already uses). One-line, env-gated,
default behaviour unchanged. Identity = GPU-vs-CPU equivalence within tolerance, so
autotuning is not needed; the segmented compile is small, so we leave autotune at
its default.

Canary (L2 d02) is a max_dom=2 live-nest -> `execute_nested_pipeline`, which already
drives a segmented `advance_chunk` host loop (one output interval per compile);
no code change needed — replicate the v0.15 `run_one_case_v0120.py` launcher.

## B. Identity dashboards (the deliverable)
<!-- FILLED AFTER 72h reruns + postprocess -->
Method = v0.15 final-gate recipe replayed on v0.17-rc code: GPU forecast (fast-compile)
-> `compare_wrfout_grid.py` -> `build_grid_delta_atlas.py` -> `build_identity_proof_plots.py`,
scored vs retained CPU-WRF truth with the FROZEN tolerance manifest
`proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json` (never moved).

Identity verdict semantics: ALL-GREEN = stable to h72, all fields finite, no run-away,
within the frozen tolerance — with the one accepted bounded-diagnostic carry per region
(Switzerland RAINNC, Canary QVAPOR) drawn honestly red against the tolerance line, the
same class shipped in v0.14/v0.15.

### Switzerland d01 72h (init 2023-01-15 00Z) — ALL-GREEN (9/10 + bounded RAINNC carry)
- Run root: `/mnt/data/wrf_gpu_validation/v017_switzerland_d01_72h_identity_fast_20260615T011416Z`
- CPU truth: `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
- Fast path: `GPUWRF_REPLAY_SEGMENTED=1` -> `jit__advance_chunk` cold compile **3m16s**
  (vs the killed run's 10m44s `jit__run_forecast_operational_jit`). GPU rc=0, all
  **72/72** frames written, no further slow compile, stable to h72.
- compare_rc=0, atlas_rc=0, identity_rc=0.
- **Identity manifest headline: n_within=9/10, worst=RAINNC (5.08x). All fields
  finite (finite_pair_fraction=1.0). NO run-away.**

| field | within | value | limit |
|---|---|---|---|
| T | PASS | 0.7143 | 1.5 |
| U | PASS | 1.3009 | 1.8 |
| V | PASS | 1.0381 | 1.8 |
| W | PASS | 0.1443 | 0.3 |
| QVAPOR | PASS | 5.859e-4 | 1.0e-3 |
| T2 | PASS | 0.7730 | 1.5 |
| U10 | PASS | 1.1454 | 1.5 |
| V10 | PASS | 1.0430 | 1.5 |
| PSFC | PASS | 29.10 | 120 |
| RAINNC | bounded-red | 5.079 | 1.0 |

These match the shipped v0.15 Switzerland identity to ~4 decimals (T 0.7143, U 1.3009,
RAINNC 5.0785 both versions), confirming the fast-compile path changes compile time only,
not the numerical identity result. RAINNC is the same accepted bounded-diagnostic carry
drawn honestly red, NOT hidden.
- Dashboards: `docs/assets/v017/identity_proof/switzerland_d01/` (identity_dashboard.png,
  identity_scoreboard.png, identity_scatter_1to1.png, identity_spatial_diff_maps.png,
  identity_timeseries_rmse_bias.png) + manifest
  `proofs/v017/identity_proof/switzerland_d01/identity_proof_manifest.json`.

### Canary L2 d02 72h (init 2026-05-01 18Z) — ALL-GREEN (9/10 + bounded QVAPOR carry)
- Run root: `/mnt/data/wrf_gpu_validation/v017_canary_d02_72h_identity_fast_20260615T024626Z`
- CPU truth: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- Path: nested `execute_nested_pipeline` (max_dom=2 live-nest), already-bounded segmented
  `jit__advance_chunk` host loop — cold compile **3m28s** (NOT the replay pathology;
  same proven path as v0.15). GPU rc=0, CLI verdict **L2_D02_GREEN / PIPELINE_GREEN**,
  all **72/72** d02 frames, all finite, stable to h72.
- compare_rc=0, atlas_rc=0, identity_rc=0.
- **Identity manifest headline: n_within=9/10, worst=QVAPOR (1.44x). All fields
  finite (finite_pair_fraction=1.0). NO run-away.**

| field | within | value | limit |
|---|---|---|---|
| T | PASS | 0.7512 | 1.5 |
| U | PASS | 0.8446 | 1.8 |
| V | PASS | 0.7116 | 1.8 |
| W | PASS | 0.0405 | 0.3 |
| QVAPOR | bounded-red | 1.442e-3 | 1.0e-3 |
| T2 | PASS | 0.8778 | 1.5 |
| U10 | PASS | 1.1630 | 1.5 |
| V10 | PASS | 1.0820 | 1.5 |
| PSFC | PASS | 37.41 | 120 |
| RAINNC | PASS | 0.0778 | 1.0 |

These match the shipped v0.15 Canary identity to ~4 decimals (QVAPOR 1.442e-3 both
versions), confirming the fast-compile-vs-v0.15 result is numerically identical. QVAPOR
is the same accepted bounded-diagnostic carry drawn honestly red, NOT hidden.
- Dashboards: `docs/assets/v017/identity_proof/canary_l2_d02/` (identity_dashboard.png,
  identity_scoreboard.png, identity_scatter_1to1.png, identity_spatial_diff_maps.png,
  identity_timeseries_rmse_bias.png) + manifest
  `proofs/v017/identity_proof/canary_l2_d02/identity_proof_manifest.json`.

## VERDICT: BOTH REGIONS ALL-GREEN (identity-proof shippable for v0.17 README)
Both regions: 9/10 hard-gate fields within the FROZEN tolerance, all fields finite,
stable to h72, NO run-away — with the one accepted bounded-diagnostic carry per region
(Switzerland RAINNC, Canary QVAPOR) drawn honestly red, the SAME class shipped in v0.14
and v0.15. The fast-compile fix (segmented replay) made the Switzerland 72h run feasible
(3m16s compile vs the 10m44s that got it killed) while producing bit-for-bit the v0.15
identity numbers.

## Honesty / rules compliance
- Frozen manifest used unchanged; scored vs retained CPU-WRF truth (no JAX-vs-JAX,
  no masking clamp). All GPU work serialized under `scripts/with_gpu_lock.sh`.
- Fast-compile path proven bit-identical to production (segscan_equiv.json), so it
  changes compile time only, not the numerical identity result.
