# v0.15 FINAL 72h Field-Parity Gates + Identity-Proof Dashboards

Date: 2026-06-13 UTC
Branch/head: `worker/opus/v015-final-gates` from `worker/gpt/v013-close-manager` @ `8dc6b466`
Defaults exercised: `niter=16` (MYNN condensation), `GPUWRF_THOMPSON_COLD_COLLECTION=1`,
dense BouLac (byte-identical to v0.14).
Frozen tolerance manifest (NEVER moved): `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`
Method = exact v0.14 recipe replayed on v0.15 code: GPU forecast (cpu_wrf_replay /
nested-pipeline) -> `compare_wrfout_grid.py` -> `build_grid_delta_atlas.py` ->
`build_identity_proof_plots.py`, all scored against retained CPU-WRF truth.

## SHIP verdict: SHIP-READY for the identity story

Both regions: **9/10 hard-gate within frozen tolerance, stable to h72, all fields
finite (finite_pair_fraction=1.0)**. The single out-of-envelope field per region is a
bounded diagnostic shown honestly red, NOT hidden — identical class to the accepted
v0.14 release. v0.15 is in fact **cleaner** than v0.14 at the atlas level (1 tolerance
failure per region vs v0.14's 3), because the v0.15 writer now emits Switzerland
DZS/ZS paired and the Canary MUB/PB boundary-frame seam is fixed.

## Switzerland d01 72h (init 2023-01-15 00Z)

- Run root: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z`
- CPU truth: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
- GPU rc=0, compare rc=0, atlas rc=0, identity rc=0. 72/72 frames, all finite, stable to h72. **9/10**.

| field | verdict | overall rmse | limit | h72 rmse | ratio |
|---|---|---|---|---|---|
| T | PASS | 0.7143 | 1.5 | 0.5318 | 0.48 |
| U | PASS | 1.3009 | 1.8 | 1.1163 | 0.72 |
| V | PASS | 1.0381 | 1.8 | 0.9069 | 0.58 |
| W | PASS | 0.1443 | 0.3 | 0.0938 | 0.48 |
| QVAPOR | PASS | 5.858e-4 | 1.0e-3 | 4.314e-4 | 0.59 |
| T2 | PASS | 0.7730 | 1.5 | 1.0194 | 0.52 |
| U10 | PASS | 1.1454 | 1.5 | 1.1243 | 0.76 |
| V10 | PASS | 1.0430 | 1.5 | 0.9321 | 0.70 |
| PSFC | PASS | 29.10 | 120 | 28.87 | 0.24 |
| RAINNC | **FAIL** | **5.0785** | 1.0 | 6.617 | **5.08** |

- **RAINNC = 5.08 mm** vs the 1.0 mm bound. Cold-collection moved it **toward** the
  bound vs v0.14 (5.99 mm -> 5.08 mm, -15%) but it is **still above** the bound (5.08x).
  Honest number: 5.08 mm, not below 1.0. Accepted bounded-diagnostic (precip),
  drawn red in the dashboard.
- Worst non-RAINNC hard field: U10 at 0.76x its limit. All dynamics/thermo comfortable.
- Atlas verdict FAIL_TOLERANCE with **1 tolerance failure (RAINNC only)**.

### Benchmark (Switzerland)
- GPU total pipeline wall: 2941.3 s (CLI proof; incl IC + compile + 72h forecast + writes + scoring).
- GPU forecast-only stepping (h01->h72 frame span): ~2414 s.
- CPU truth (24-rank dmpar): total_wall 2906.3 s, mainloop 2887.6 s.
- Speedup: total-wall **~0.99x (parity, GPU ~1% slower)**; forecast-only-vs-mainloop **~1.20x**.
  The heavier v0.15 operational compile (~8-12 min, one-time) is included in total wall.
- Peak GPU memory: **22885 MiB** (v0.14 was 20474 MiB; +niter16 unrolled temps).

## Canary L2 d02 72h (init 2026-05-01 18Z, nested d01+d02)

- Run root: `<DATA_ROOT>/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z`
- CPU truth: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- GPU rc=0 (CLI verdict L2_D02_GREEN), compare rc=0, atlas rc=0, identity rc=0.
  72/72 d02 frames, all finite, stable to h72. **9/10**.

| field | verdict | overall rmse | limit | h72 rmse | ratio |
|---|---|---|---|---|---|
| T | PASS | 0.7509 | 1.5 | 0.8304 | 0.50 |
| U | PASS | 0.8444 | 1.8 | 0.9879 | 0.47 |
| V | PASS | 0.7108 | 1.8 | 0.7363 | 0.40 |
| W | PASS | 0.0405 | 0.3 | 0.0658 | 0.13 |
| QVAPOR | **FAIL** | **1.4422e-3** | 1.0e-3 | 1.7095e-3 | **1.44** |
| T2 | PASS | 0.8752 | 1.5 | 0.7892 | 0.58 |
| U10 | PASS | 1.1617 | 1.5 | 1.2619 | 0.77 |
| V10 | PASS | 1.0805 | 1.5 | 1.0349 | 0.72 |
| PSFC | PASS | 37.46 | 120 | 16.79 | 0.31 |
| RAINNC | PASS | 0.0778 | 1.0 | 0.0898 | 0.08 |

- **QVAPOR = 1.4422e-3** vs the 1.0e-3 bound = the carried ~1.44e-3 envelope.
  **No regression** from cold-collection/niter (v0.14 1.452e-3 -> v0.15 1.442e-3).
  Lone bounded miss, drawn red in the dashboard.
- Worst non-QVAPOR hard field: U10 at 0.77x its limit. All dynamics/thermo comfortable.
- Atlas verdict FAIL_TOLERANCE with **1 tolerance failure (QVAPOR only)** —
  MUB/PB boundary-frame statics that v0.14 flagged are now fixed.

### Benchmark (Canary)
- GPU total wall: 8413.6 s (CLI proof); forecast-only 8332.5 s.
- CPU truth (28-rank backfill, retained wrfout timestamp span): ~8713 s (honest
  approximate denominator; no rank-0 timing file exists for the historical backfill).
- Speedup: total-wall **~1.04x**; forecast-only **~1.05x** (same parity-class as v0.14's 1.06x).
- Peak GPU memory: **29782 MiB** (v0.14 was 21108 MiB; +niter16 unrolled temps, larger nested arena).

## Identity-proof dashboards (the deliverable)

Regenerated honestly (green only where truly within the frozen tolerance; the bounded
carry drawn red against the tolerance line, not hidden). 5 plots/region + manifest:

- Switzerland: `docs/assets/v015/identity_proof/switzerland_d01/` —
  manifest headline `n_within=9/10, worst=RAINNC (5.08x)`.
- Canary: `docs/assets/v015/identity_proof/canary_l2_d02/` —
  manifest headline `n_within=9/10, worst=QVAPOR (1.44x)`.
- Files each: `identity_dashboard.png`, `identity_scoreboard.png`,
  `identity_scatter_1to1.png`, `identity_spatial_diff_maps.png`,
  `identity_timeseries_rmse_bias.png`, `identity_proof_manifest.json`.

## Long-horizon divergence verdict on the two carries (ADDED, does not move the gate)

The two strict-tolerance carries were tested against the principal's long-horizon
non-escalating-divergence criterion (`.agent/decisions/REDUCED-PRECISION-EQUIVALENCE-AND-FP32-RIGOR.md §3`).
Measured result: **both are BOUNDED / non-escalating over 72 h — NOT run-aways.**
Switzerland RAINNC saturates (early→late slope ratio +0.05) at ~1.1× the precip field's
own spatial spread; Canary QVAPOR saturates (ratio −0.04) at 0.47× the moisture field's
spread. All other 9 fields per region are non-escalating too. So each carry is a
tight-per-cell-tolerance miss carried to 0.16, not a stability failure. The strict
frozen-tolerance 9/10 (red on the carry) is unchanged.

- Verdict: `proofs/v015/long_horizon_divergence_verdict.json` + `proofs/v015/LONG_HORIZON_DIVERGENCE_VERDICT.md`
- Added dashboard panel per region (adjacent to the unchanged identity dashboards):
  `docs/assets/v015/identity_proof/{switzerland_d01,canary_l2_d02}/long_horizon_divergence_panel.png`

## Honesty / rules compliance

- Frozen manifest used unchanged; no tolerance moved, no masking clamp, no JAX-vs-JAX
  (both scored vs retained CPU-WRF truth). All GPU work ran under `with_gpu_lock.sh`.
- The two out-of-envelope fields (Switzerland RAINNC, Canary QVAPOR) are the same
  bounded-diagnostic class as the shipped v0.14 release, reported at true value. The
  long-horizon divergence test above confirms they are bounded/non-escalating, ADDING a
  second criterion without moving or loosening the strict per-cell tolerance.
