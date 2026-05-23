# Worker Report - M6.x Gen2 RMSE Baseline Characterization

Summary: Implemented the pure-analysis Gen2 RMSE diagnostic and ran the contract validation command against the read-only Gen2 backfill. The actual archive layout differs from the older reference: many `wrf_l2` runs no longer retain per-hour `wrfout_d02_*` files but do retain `thin_gridded_d02_v1.nc` products with hourly `T2`/`U10`/`V10`. The diagnostic adapts to that layout, uses Method A consecutive-day overlap, and produces real 24h and 72h d02 RMSE anchors from 17 same-grid sample pairs.

## Inventory Summary

- `wrf_l3`: 29 run directories; 6 retained d02 products; 3 complete advertised 24h d02 runs.
- `wrf_l3` longest consecutive usable d02 init-date span: `2026-05-21..2026-05-22` (2 days).
- `wrf_l2`: 23 run directories; 23 retained d02 products; 22 complete advertised 72h d02 runs.
- `wrf_l2` longest consecutive usable d02 init-date span: `2026-04-30..2026-05-21` (22 days).
- Product convention observed: full files use `<run-id>/wrfout_d02_YYYY-MM-DD_HH:MM:SS`; retained thin products use `<run-id>/thin_gridded_d02_v1.nc`.
- Output frequency: hourly for complete retained products.
- Sample full wrfout metadata: 375 variables. Target fields are present as `T2` K, `U10` m/s, and `V10` m/s.

## Method

Method A was used because it is the contract preference: compare consecutive daily cycles at the same valid time. The tool scans the requested `wrf_l3` root and the `wrf_l2` sibling needed for 72h leads. For each lead it picks the source with the most valid consecutive-day same-valid pairs. `wrf_l2` thin products produced the binding sample: 17 same-grid pairs after skipping 4 shape-mismatched early pairs.

For 24h, each pair compares lead 24h from day D against lead 0h from day D+1. For 72h, each pair compares lead 72h from day D against lead 48h from day D+1. This measures Gen2 forecast-to-forecast variance under no model-code change, with updated IC/BC timing.

## Numerical RMSE Table

| field | lead_hours | spatial_mean_rmse | p95_rmse | sample_pairs | units |
|---|---:|---:|---:|---:|---|
| T2 | 24 | 0.6284061313 | 1.76199533 | 17 | K |
| U10 | 24 | 1.456482265 | 2.954589341 | 17 | m/s |
| V10 | 24 | 1.590974439 | 3.556604337 | 17 | m/s |
| T2 | 72 | 0.2550553229 | 0.4131125664 | 17 | K |
| U10 | 72 | 0.8878892918 | 1.77950453 | 17 | m/s |
| V10 | 72 | 0.8699087447 | 1.742272248 | 17 | m/s |

Time variability from per-pair RMSE:

- `T2` 24h: mean 0.951439, std 0.0925262, min 0.720742, max 1.08338 K.
- `U10` 24h: mean 1.61366, std 0.238994, min 1.08048, max 2.08774 m/s.
- `V10` 24h: mean 1.7955, std 0.378548, min 1.07196, max 2.37773 m/s.
- `T2` 72h: mean 0.251543, std 0.0903708, min 0.127197, max 0.44387 K.
- `U10` 72h: mean 0.894837, std 0.434395, min 0.431311, max 1.65442 m/s.
- `V10` 72h: mean 0.89352, std 0.382921, min 0.378329, max 1.59241 m/s.

## Spatial Pattern

T2 24h, top5 boundary fraction 0.002, mean lat/lon `28.2304/-16.6029`:

```text
::--::.::..::::.........
::=::::::..::--:........
.:--:::-::..:=+=........
.:::::----:::-*=...:::::
::::.:*=++*-:-=:...:-::.
::::.:--*@#=:......::=-:
.:-:::::-=#=:.......::-:
.:-+-:.:::--.........:-:
..-#=...::.:..........::
..:+=..................:
...::.:................:
......::...............:
```

U10 24h, top5 boundary fraction 0.133, mean lat/lon `27.9943/-16.7350`:

```text
:-+-:===+==-=+#=....:--:
:-*+-+#+*+=-=#@#....:--:
:===-=*+**===+**:::.---:
-=--:-++**====+--::.-==-
-++-:-==+*#--+==-::::++:
:-*+---+=+*+:::::::::-=:
:-*#=-=+*=#+:::.:::::.-:
--+*==----+=-::..:::::-:
--=*---:---::::....:::::
-=+=-::::--::::........:
:-=-:::::::::::::......:
:::::::::::::::::::::...
```

V10 24h, top5 boundary fraction 0.021, mean lat/lon `28.0708/-16.7704`:

```text
-=---:::::--====::.:::::
-===-----:::=***:::::-::
-::--==+=--:-#%+::::--:.
=::--=+@#*=--=+-::::--:.
=-:::==@%#=:---::::::--:
==-::::**==--::.::.::--:
-=+=-:::::--:::...:::-::
-+*++-::.:::::::::::::::
-=*++::....::::::::::.::
::--:::....::::::::::.::
:::::::....::::::::::.:-
:::::::......::::::::...
```

T2 72h, top5 boundary fraction 0.088, mean lat/lon `28.1376/-16.0779`:

```text
-===---::::::-::..::.:-=
-=+*==-::::::+=:..::::-=
--++-==-:::::#*=:::-::-=
--==--=-=-:::+%+.:::--=+
------@+-*+::-=+...=-=++
-=-----=+#*:::::.::--+#+
:-==:::---*+-:...:-:-=+=
:-=+----::--::::::::--==
:-=+--:::::::::::::::---
::-*--:::::::::::::::--+
::::---:::::::::::::::-+
:::::--::::::::::::::::+
```

U10 72h, top5 boundary fraction 0.059, mean lat/lon `28.0705/-17.2800`:

```text
--:-=======--=-::..::::-
+=-+:*@**+==-+*--:.::---
*#==-+***======---.:--=-
#%+=-=**+=+*-:-:-:-:===-
*@*=----*==+=:-=++-:-=+-
-%@--=+=-:-+:::-+--:::==
--#+*=+=:::==::---+:..-=
--=-**+=---:-::-:--::::-
-=-:::::::::::--:::::::-
-----:::::::::---::::::=
------:::::::::---::::::
-:::--::::::::::::::::::
```

V10 72h, top5 boundary fraction 0.042, mean lat/lon `28.0675/-17.1025`:

```text
:+:--::::.:=-:-::-------
:--=#+-=-..:==--:::--=--
::.:*%+*=-::-+:+-::-:==-
-==-=%+@+-:=-=:=%=:-----
=+=::+=#*:-=+-:+=+--:=-=
-*=-:==+-::+=..:-+--::==
-+*-=#+:...=:..:::--:::=
--#-*+--::..:..::.:::.:-
:-=.==-:::.:::.......:::
.:::----::::.::......:::
::..:::::::::.::...::.::
....:..::......::..:..::
```

## Threshold Recommendation

Recommend `RMSE < 2.0x Gen2 noise floor` as the default Tier-4 rejection threshold, matching ADR-007's operational-RMSE focus for `U10`/`V10`/`T2` while allowing minor implementation differences. Spatial-mean gates from this run:

- `T2` 24h: `< 1.2568122626 K`
- `U10` 24h: `< 2.91296453 m/s`
- `V10` 24h: `< 3.181948878 m/s`
- `T2` 72h: `< 0.5101106458 K`
- `U10` 72h: `< 1.7757785836 m/s`
- `V10` 72h: `< 1.7398174894 m/s`

Keep the CSV's p95 values as secondary concentration checks; the manager can apply the same 2.0x multiplier there if the close gate wants a regional-hotspot guard.

## Files Changed

- `scripts/diagnostic_gen2_rmse_baseline.py`
- `data/fixtures/gen2_baseline/rmse_summary.csv`
- `.agent/sprints/2026-05-23-m6x-gen2-rmse-baseline-characterization/proof_rmse_baseline.txt`
- `.agent/sprints/2026-05-23-m6x-gen2-rmse-baseline-characterization/worker-report.md`

## Commands Run

```bash
python scripts/diagnostic_gen2_rmse_baseline.py --help
```

Output: CLI usage printed successfully with `--gen2-root`, `--output`, `--method`, `--domain`, `--fields`, and `--leads`.

```bash
python scripts/diagnostic_gen2_rmse_baseline.py \
  --gen2-root /mnt/data/canairy_meteo/runs/wrf_l3/ \
  --output data/fixtures/gen2_baseline/rmse_summary.csv \
  --method A | tee .agent/sprints/2026-05-23-m6x-gen2-rmse-baseline-characterization/proof_rmse_baseline.txt
```

Output: exit 0. Full stdout/stderr is captured in `proof_rmse_baseline.txt`. Key lines:

```text
wrf_l2 retained products: 23 runs, 22 complete, 22-day consecutive span
Lead 24h source root: /mnt/data/canairy_meteo/runs/wrf_l2
Lead 72h source root: /mnt/data/canairy_meteo/runs/wrf_l2
T2,24,0.6284061313,1.76199533,17,K
U10,24,1.456482265,2.954589341,17,m/s
V10,24,1.590974439,3.556604337,17,m/s
T2,72,0.2550553229,0.4131125664,17,K
U10,72,0.8878892918,1.77950453,17,m/s
V10,72,0.8699087447,1.742272248,17,m/s
```

## Proof Objects

- `scripts/diagnostic_gen2_rmse_baseline.py`
- `data/fixtures/gen2_baseline/rmse_summary.csv`
- `.agent/sprints/2026-05-23-m6x-gen2-rmse-baseline-characterization/proof_rmse_baseline.txt`
- `.agent/sprints/2026-05-23-m6x-gen2-rmse-baseline-characterization/worker-report.md`

## Risks

- The binding numbers come from retained `thin_gridded_d02_v1.nc` products where full `wrfout_d02_*` was not retained. They should be accepted as Gen2-derived forecast products, not as raw wrfout-file parity evidence.
- Four early l2 candidate pairs were skipped because grid shape changed across products; the CSV uses 17 same-grid pairs only.
- Results are for ADR-016 d02. If the manager wants d01 or another nest as the binding Tier-4 domain, rerun with `--domain`.
- Method A intentionally compares different lead ages at the same valid time (`24h` vs `0h`, `72h` vs `48h`), so it is an operational forecast-to-forecast variance floor, not an observation-error floor.

## Handoff

Objective: characterize Gen2 forecast-to-forecast RMSE noise floor for `T2`/`U10`/`V10` at 24h/72h without touching model code.

Files changed: listed above.

Commands run: listed above; required validation command exited 0 and proof transcript is on disk.

Proof objects produced: listed above.

Unresolved risks: thin-product source for most 72h evidence, four skipped shape-mismatched pairs, and d02-only scope.

Next decision needed: manager should decide whether to bind the provisional Tier-4 close gate to `2.0x` the CSV spatial-mean floors, and whether to add p95 hotspot gates using the same multiplier.
