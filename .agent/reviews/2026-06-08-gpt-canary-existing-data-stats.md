# GPT Canary Existing-Data Stats Review

Date: 2026-06-08
Branch: `worker/gpt/v013-canary-stats`
Base verified: `c93ffdf6 [v013] Agent F merged + WDM5-unroutable FIXED + Tiedtke debug lane (GPT-tmux win2)`

## Verdict

Existing CPU-WRF data is good enough to avoid new CPU runs for the **D02 powered TOST n=15** campaign. The corpus at `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output` has 15 complete CPU-WRF cases with D01/D02 hourly wrfout through 72 h.

Existing GPU/JAX outputs are **not** good enough to make a robust statistical GPU ~= CPU-WRF equivalence claim. They are either incomplete, old/narrow station-score outputs, grid-mismatched, proof-only summaries, or lack matching CPU-WRF truth. They can support first-order risk assessment and script smokes only.

For D03/nested 1 km equivalence, the existing raw corpus is **not sufficient**. Most MAM D03 CPU wrfouts are purged; surviving D03 CPU truth is sparse, and the 2026-05-31 GPU gate outputs have only one matching CPU-WRF lead on disk.

## Inventory Proof

Generated artifacts:

- `proofs/canary_stats/2026-06-08_existing_data/inventory.json`
- `proofs/canary_stats/2026-06-08_existing_data/inventory.csv`
- `proofs/canary_stats/2026-06-08_existing_data/inventory_table.md`
- `proofs/canary_stats/2026-06-08_existing_data/pairability.json`
- `proofs/canary_stats/2026-06-08_existing_data/pairability.csv`
- `proofs/canary_stats/2026-06-08_existing_data/pairability_table.md`
- `proofs/canary_stats/2026-06-08_existing_data/summary.md`

Inventory counts:

| class | records | wrfout files | usability |
| --- | ---: | ---: | --- |
| CPU-WRF | 37 | 3438 | primary truth where matching GPU exists |
| GPU/JAX GPUWRF full writer | 6 | 300 | mostly gate outputs; limited CPU truth overlap |
| GPU/JAX proof-output narrow writer | 4 | 154 | M20 T2/U10/V10-only station-score outputs |
| forcing/input-only | 88 | 0 | useful for future GPU init, not truth stats |
| unknown | 26 | 0 | no usable forecast output |

Corpus facts verified:

- `wrf_l2_backfill_output`: 15 complete D01/D02 CPU-WRF truth cases, 73 hourly frames per domain, 72 h coverage.
- `/mnt/data/canairy_meteo/runs/wrf_l2`: 32 dirs scanned, only 2 still have wrfout.
- `/mnt/data/canairy_meteo/runs/wrf_l3`: 41 dirs scanned, only 8 still have wrfout.
- `campaign_l2` and `campaign_l3`: no useful files found in this scan.
- Switzerland CPU truth exists under `/mnt/data/wrf_gpu_switzerland*`; noted, not pooled with Canary.

## Raw Pairability

Strict raw pairability found no robust complete same-grid CPU/GPU ensemble.

| domain | robust same-grid cases | exploratory same-grid pairs | station-only/grid-mismatch pairs |
| --- | ---: | ---: | ---: |
| d01 | 0 | 5 pairs / 2 unique init times | 0 |
| d02 | 0 | 5 pairs / 2 unique init times | 2 |
| d03 | 0 | 2 pairs / 1 unique init time | 0 |

Pairability classes:

- `exploratory_incomplete_leads`: 12 pairs. These include the partial 2026-04-29 D01/D02 GPU output with 16 raw common lead times and 2026-05-31 gate outputs with only 1 matching CPU-WRF lead.
- `pairable_station_only_grid_mismatch`: 2 D02 pairs from `proofs/m20/tost_run/gpu_wrfout` (`case2_L2`, `case3_L3`). They can be station-interpolated, but are not same-grid raw equivalence evidence and only contain T2/U10/V10 plus coordinates/time.

## Smoke Results

Commands were run CPU-only with `taskset -c 29-31 env JAX_PLATFORMS=cpu`.

Gridded D02 partial pair, CPU `wrf_l2_backfill_output/20260429...` vs GPU `/tmp/v0120_powered_tost_runs/...`, D02, max lead 15, spin-up lead 1 excluded, scored leads 2-15, 5-cell boundary trim:

| variable | frames | RMSE | bias | mean frame r |
| --- | ---: | ---: | ---: | ---: |
| T2 | 14 | 1.289986 K | 0.464460 K | 0.826150 |
| U10 | 14 | 2.411389 m/s | -0.583356 m/s | 0.405856 |
| V10 | 14 | 3.829848 m/s | -1.258346 m/s | 0.337678 |
| PSFC | 14 | 177.909952 Pa | 101.654601 Pa | 0.999066 |
| T | 14 | 4.892850 K | -0.601649 K | 0.997593 |
| U | 14 | 3.071638 m/s | -0.441404 m/s | 0.959338 |
| V | 14 | 8.690808 m/s | 0.476252 m/s | -0.004549 |
| W | 14 | 0.090217 m/s | 0.000167 m/s | 0.268341 |

This is exploratory and does not support same-solution equivalence.

Gridded D03 one-lead pair, CPU `wrf_l3/20260531...` vs GPU `gate_revalidate_gwd8/out`, D03, lead 1 only, 5-cell boundary trim:

| variable | frames | RMSE | bias | mean frame r |
| --- | ---: | ---: | ---: | ---: |
| T2 | 1 | 0.675124 K | 0.132584 K | 0.945919 |
| U10 | 1 | 1.327797 m/s | -0.676813 m/s | 0.964037 |
| V10 | 1 | 1.182944 m/s | -0.504446 m/s | 0.971163 |
| PSFC | 1 | 212.009579 Pa | -211.163115 Pa | 0.999996 |

This is a smoke only. One lead cannot support statistical equivalence.

Station TOST smoke on two M20 raw pairs (`case2_L2`, `case3_L3`) scored successfully, but n=2 is underpowered and all variables were not equivalent. Aggregate mean RMSE deltas were T2 +1.071253 K, U10 +0.166817 m/s, V10 -0.040056 m/s. Stored M20/postfix n=3 proof also reports `NOT_EQUIVALENT_OR_UNDERPOWERED`; its T2 mean delta is +0.734324 K, outside the ADR-029 T2 margin.

## Usability By Statement

### Robust

- The 15-case D02 CPU-WRF truth corpus exists and can be reused for powered TOST. No new CPU-WRF D02 runs are needed for that campaign.
- Most historical L2/L3 MAM wrfouts were actually purged; raw D03/nested statistical reuse is not available at n~30.
- Existing raw GPU outputs do not add robust statistical power to ADR-029 n=15 D02 TOST.

### First-order / exploratory

- Partial same-grid CPU/GPU gridded comparisons can be made for 2026-04-29 D01/D02 through 15 leads.
- A one-lead 2026-05-31 D03 CPU/GPU comparison can be made for the gate outputs.
- M20 T2/U10/V10-only station-score outputs can be re-scored for two raw pairs still on disk and can be read from stored JSON for a few more proof-only units.
- Gate outputs support completion/finiteness statements for their own GPU runs, not CPU equivalence.

### Not usable for robust statistics

- Purged `wrf_l2` / `wrf_l3` dirs with only `wrfinput`, `wrfbdy`, namelists, or met_em forcing.
- M20 `case1_L2` / `case1_L3` raw GPU outputs as robust raw pairs: matching CPU raw wrfout referenced by the old sidecar is no longer on disk.
- D03/nested gate outputs as statistical equivalence evidence: matching CPU-WRF truth is missing except for one 2026-05-31 lead.
- Any pooled n~30 claim combining D02, D03, old narrow outputs, proof-only summaries, and current gate outputs. The domains, grids, variables, code versions, and available masks differ.

## Key Question

Can we validate, to first approximation and without new CPU runs, that GPU/JAX reaches essentially the same solution as CPU-WRF for implemented couplings?

**Partly, but only for the D02 powered-TOST path after new GPU runs are executed.** The existing 15-case D02 CPU truth is sufficient and should be reused. The missing piece is current GPU/JAX outputs for those same cases, domains, leads, and masks.

**No for a robust existing-output claim.** The GPU outputs already on disk do not establish GPU ~= WRF. The best raw same-grid D02 pair is partial and shows material surface/3D differences. The D03/nested GPU gate outputs have almost no matching CPU truth on disk. The M20 station-score outputs are narrow, grid-mismatched, n<=3, and their stored TOST results are not equivalent/underpowered.

## Relation To Powered TOST n=15

The existing D03/nested and M20 outputs do **not** add statistical power to the in-flight powered TOST n=15. They should not be pooled into the ADR-029 result.

They add useful coverage only as exploratory diagnostics:

- D03/nested gate completion and finiteness.
- old station-score behavior for T2/U10/V10.
- smoke-test coverage for the scripts.

The statistically defensible route remains:

1. Reuse `wrf_l2_backfill_output` as CPU truth.
2. Run current GPU/JAX D02 forecasts for the same 15 cases.
3. Score with ADR-029 complete-pair deletion and predeclared margins.
4. Report n=15 as underpowered for 10% RMSE if empirical sigma does not rescue power, per ADR-029.

## Minimal Additional Runs

- D02 TOST n=15: **GPU only**. Existing CPU-WRF truth is sufficient.
- D03/nested robust equivalence: **CPU needed unless purged CPU wrfouts can be restored**. Existing D03 CPU truth is too sparse for n>=15 or even a credible seasonal subset.
- 2026-05-31 D03 gate equivalence: **CPU needed** for the missing 24 h D03 truth if that exact gate is to be claimed against CPU-WRF.
- Additional exploratory station-score reruns: **GPU only** where CPU truth already exists, but these should not be promoted to robust equivalence without same-case coverage and n.

## Scripts Added

- `scripts/canary_stats/inventory_existing_runs.py`
- `scripts/canary_stats/assess_pairability.py`
- `scripts/canary_stats/score_grid_pairs.py`
- `scripts/canary_stats/score_station_tost_pairs.py`
- `scripts/canary_stats/make_summary_report.py`
- `scripts/canary_stats/common.py`
- `scripts/canary_stats/README.md`

## Commands Run

```bash
git log -1 --oneline
taskset -c 29-31 env JAX_PLATFORMS=cpu python scripts/canary_stats/inventory_existing_runs.py --out-dir proofs/canary_stats/2026-06-08_existing_data
taskset -c 29-31 env JAX_PLATFORMS=cpu python scripts/canary_stats/assess_pairability.py --inventory proofs/canary_stats/2026-06-08_existing_data/inventory.json --out-dir proofs/canary_stats/2026-06-08_existing_data
taskset -c 29-31 env JAX_PLATFORMS=cpu python scripts/canary_stats/score_grid_pairs.py --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260429_18z_l2_72h_20260524T204451Z --gpu-dir /tmp/v0120_powered_tost_runs/l2_d02_20260429_18z_l2_72h_20260524T204451Z --domain d02 --init 2026-04-29T18:00:00+00:00 --max-lead 15 --spinup-hours 1 --boundary-width 5 --vars T2 U10 V10 PSFC RAINNC T U V W --out-json proofs/canary_stats/2026-06-08_existing_data/smoke_grid_20260429_d02.json --out-csv proofs/canary_stats/2026-06-08_existing_data/smoke_grid_20260429_d02.csv
taskset -c 29-31 env JAX_PLATFORMS=cpu python scripts/canary_stats/score_grid_pairs.py --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z --gpu-dir /mnt/data/canairy_meteo/gate_revalidate_gwd8/out --domain d03 --init 2026-05-31T18:00:00+00:00 --max-lead 1 --spinup-hours 0 --boundary-width 5 --vars T2 U10 V10 PSFC RAINNC --out-json proofs/canary_stats/2026-06-08_existing_data/smoke_grid_20260531_d03_1lead.json --out-csv proofs/canary_stats/2026-06-08_existing_data/smoke_grid_20260531_d03_1lead.csv
taskset -c 29-31 env JAX_PLATFORMS=cpu python scripts/canary_stats/score_station_tost_pairs.py --pair '{"case_id":"smoke_20260429_d02_partial","cpu_dir":"/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260429_18z_l2_72h_20260524T204451Z","gpu_dir":"/tmp/v0120_powered_tost_runs/l2_d02_20260429_18z_l2_72h_20260524T204451Z","domain":"d02","init":"2026-04-29T18:00:00+00:00","fh":15}' --out proofs/canary_stats/2026-06-08_existing_data/smoke_station_tost_20260429_d02_partial.json
taskset -c 29-31 env JAX_PLATFORMS=cpu python scripts/canary_stats/score_station_tost_pairs.py --pair '{"case_id":"case2_L2_existing_raw","cpu_dir":"/mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z","gpu_dir":"proofs/m20/tost_run/gpu_wrfout/case2_L2","domain":"d02","init":"2026-05-09T18:00:00+00:00","fh":72}' --pair '{"case_id":"case3_L3_existing_raw","cpu_dir":"/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z","gpu_dir":"proofs/m20/tost_run/gpu_wrfout/case3_L3","domain":"d02","init":"2026-05-21T18:00:00+00:00","fh":24}' --out proofs/canary_stats/2026-06-08_existing_data/smoke_station_tost_m20_two_pairs.json
taskset -c 29-31 env JAX_PLATFORMS=cpu python scripts/canary_stats/make_summary_report.py --inventory proofs/canary_stats/2026-06-08_existing_data/inventory.json --pairability proofs/canary_stats/2026-06-08_existing_data/pairability.json --out-md proofs/canary_stats/2026-06-08_existing_data/summary.md --plot-dir proofs/canary_stats/2026-06-08_existing_data/plots
taskset -c 29-31 env JAX_PLATFORMS=cpu python -m py_compile scripts/canary_stats/*.py
```

## Handoff

- objective: determine reusable statistical value of existing Canary CPU/GPU outputs and prepare scripts.
- files changed: `scripts/canary_stats/*`, `proofs/canary_stats/2026-06-08_existing_data/*`, this report.
- proof objects produced: inventory JSON/CSV/Markdown, pairability JSON/CSV/Markdown, grid-score smoke JSON/CSV, station-score smoke JSON, generated summary/plots.
- unresolved risks: inventory is limited to roots listed in the script; unknown proof-only JSONs may contain historical summaries but cannot replace raw re-scoreable wrfouts.
- next decision needed: run current GPU/JAX D02 powered-TOST cases against the existing n=15 CPU truth, and decide whether D03 CPU truth should be restored/rerun or deferred.
