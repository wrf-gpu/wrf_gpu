# Canary CPU Truth Inventory for v0.14 Field-Parity Gate

Date: 2026-06-10
Manager branch: `worker/gpt/v013-close-manager`

## Decision

Use the existing Canary **L2 d02 72 h** CPU-WRF truth as the mandatory v0.14
Canary field-parity gate. The selected primary case is:

- run id: `20260501_18z_l2_72h_20260519T173026Z`
- CPU truth: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- retained input/run dir: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z`
- selected domain: `d02`
- horizon: 72 h, 73 hourly frames from `2026-05-01_18:00:00` through
  `2026-05-04_18:00:00`
- grid: `159 x 66 x 44` mass grid, `DX=DY=3000 m`, `GRID_ID=2`,
  `PARENT_ID=1`, `PARENT_GRID_RATIO=3`, `USE_THETA_M=1`

This case is selected over a fresh CPU run because it is already complete,
WRF-returncode clean, and is the same case used by the current h1 field
falsifier. That gives the fastest rigorous path after the EOS/theta blocker is
settled: rerun the short h1 GPU falsifier, then extend the same case to 72 h.

## Inventory Summary

Command used to find retained d02/d03 truth:

```bash
find <DATA_ROOT>/canairy_meteo/runs -maxdepth 3 -type f \
  \( -name 'wrfout_d02_*' -o -name 'wrfout_d03_*' \) \
  | sed -E 's#/wrfout_d0[23]_[0-9:_-]+$##' \
  | sort | uniq -c | sort -nr | head -80
```

Findings:

| Class | Evidence | Gate decision |
|---|---:|---|
| L2 d02 72 h CPU truth | 15 complete backfill roots with 73 `wrfout_d02_*` frames each, plus one direct `wrf_l2` root with 73 frames | Mandatory v0.14 Canary gate |
| L3 d03 24 h CPU truth | complete 25-frame examples exist for 20260509 and 20260521 | Secondary/stretch only |
| L3 d03 72 h CPU truth | not found as retained `wrfout_d03_*` truth | Would require a new CPU-WRF baseline build/run |

Representative complete L2/d02 roots found:

- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260429_18z_l2_72h_20260524T204451Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260430_18z_l2_72h_20260520T191306Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260502_18z_l2_72h_20260520T103946Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260503_18z_l2_72h_20260518T205545Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260504_18z_l2_72h_20260515T061907Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260505_18z_l2_72h_20260518T074056Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260506_18z_l2_72h_20260513T222831Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260507_18z_l2_72h_20260513T124307Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260508_18z_l2_72h_20260512T161222Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260510_18z_l2_72h_20260511T124717Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260511_18z_l2_72h_20260512T045528Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260512_18z_l2_72h_20260513T014823Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260513_18z_l2_72h_20260514T054102Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260530_18z_l2_72h_20260531T161057Z`
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z`

The selected `20260501` backfill manifest reports:

```json
{
  "mode": "wrf-only",
  "ncores": 28,
  "returncodes": {"wrf": 0},
  "final_output_counts": {"wrfout_d01": 73, "wrfout_d02": 73},
  "run_dir": "<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z",
  "safe_output_dir": "<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z"
}
```

## d03 / Tenerife Assessment

Retained L3 examples are scientifically valuable because they cover 1 km
steep-island nests and are more complementary to the Switzerland/Gotthard
single-domain 3 km Alps gate. However, the retained complete L3 examples are
24 h, not 72 h:

- `<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260511T190519Z`
  has 25 frames for each `d01` through `d05`.
- `<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z`
  has 25 frames for `d02` and `d03`.

For the 20260509 L3 root:

- `namelist.input` has `run_hours = 24`, `max_dom = 5`, `restart = .false.`.
- no `wrfrst_*` files were found in the run root.
- `wrfbdy_d01` has `Time = 4`, consistent with 24 h boundary forcing
  (`18z`, `+6`, `+12`, `+18`).
- the `d03` grid is `72 x 72 x 44`, `DX=DY=1000 m`, `GRID_ID=3`,
  `PARENT_ID=2`, `PARENT_GRID_RATIO=3`, `USE_THETA_M=1`.

Therefore d03 is not a clean resume/extend from retained truth. A 72 h d03 gate
would be a new CPU-WRF truth campaign plus a more complex GPU run. It remains a
strong stretch or v0.15 validation target after the two required 72 h gates are
green.

## Next Commands After h1 Blocker Is Green

Short Canary h1 falsifier on the selected case:

```bash
RUN_ROOT=<DATA_ROOT>/wrf_gpu_validation/v014_short_field_falsifier_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_ROOT"/{gpu_output,proofs,resources}
GPUWRF_RESOURCE_LOG_DIR="$RUN_ROOT/resources" \
GPUWRF_RESOURCE_LABEL=v014_canary_h1_field_falsifier \
scripts/run_gpu_lowprio.sh -- python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
  --run-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2 \
  --cpu-truth-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output \
  --run-id 20260501_18z_l2_72h_20260519T173026Z \
  --hours 1 \
  --output-root "$RUN_ROOT/gpu_output" \
  --proof-dir "$RUN_ROOT/proofs"
```

If that short run is green, the mandatory Canary 72 h gate uses the same command
shape with `--hours 72` and a new immutable run root, then Grid-Delta Atlas over
every common numeric `wrfout_d02` field for leads 0-72.

