# In-run extract (v0.5.0)

**Status: new in v0.5.0, off by default.** The extract writes compact per-nest evaluation files while the
forecast runs, so a series of experiments does not need to keep or re-read full `wrfout` history. With the
extract on or off the forecast itself is byte-identical (checked on the GPU with nine nests). The extract
equals an offline recomputation from full history (checked on CPU, real cases).

## Run with an extract

```bash
gpuwrf run --input-dir CASE --output-dir OUT --domains-from-namelist --emit-initial-history \
  --extract-spec v1 --extract-config request.json --extract-dir OUT_EXTRACT
```

All three `--extract-*` options are needed together, plus `--emit-initial-history` (the lead-0 frame).
The extract needs the native driver (nested, or native single-domain d01), one member (B=1) and no branch.
`--extract-dir` must be a fresh directory.

`request.json` (schema `gpuwrf.extract.request.v1`; unknown keys are refused):

```json
{"schema": "gpuwrf.extract.request.v1",
 "run_id": "series01", "member": "m000", "issue_utc": "20260721_18z",
 "arm": "control", "seed": null, "domains": ["d02", "d03"],
 "points_csv": ["/abs/path/POINTS_V2.csv"], "max_distance_m": {"d03": 1500},
 "raw_history": "on"}
```

| Key | Meaning |
|---|---|
| `run_id`, `member`, `issue_utc` | required; identifiers `[A-Za-z0-9._-]`, issue as `YYYYmmdd_HHz` or ISO UTC |
| `arm`, `seed` | optional labels copied into the manifest |
| `domains` | optional subset of nests (default: all); other nests are `NOT_REQUESTED` |
| `points_csv` | optional absolute paths of point lists; their SHA-256 is bound to the run and checked again on resume |
| `max_distance_m` | optional per nest; default 1.5 grid lengths |
| `raw_history` | `on` (default) writes the normal `wrfout`; `off` writes only the extract (no `wrfout` files at all) |

Point lists are accepted only with one of two exact headers:
`id,lat,lon,height_m,network,name,kind,height_source,coordinate_source` (stations) or
`id,kind,nest,island,zone,y,x,lat,lon,hgt_m,dem_m,line_order` (fixed grid cells).

## What is written

Per nest `extract_<run>_<member>_dNN.nc` with dimensions (time, y, x) and, with point lists,
`profiles_<run>_<member>_dNN.nc` with (time, point, level), a `points.csv` with the mapping, and
`manifest.json`, written last. A manifest with status `READY` means every file was read back and checked.

- Surface: `U10 V10` (grid and earth-relative), wind speed and direction, `T2 Q2 PSFC PBLH HFX LH UST ZNT
  RAINC RAINNC SWDOWN GLW TSK`, `SST` (only when the lower-boundary SST is updated, else `NOT_AVAILABLE`),
  `WSPD10MAX` (only when a producer is on: AFWA or `nwp_diagnostics`; its averaging window is recorded).
- Cloud: column cloud-fraction maximum and low-cloud maximum, liquid and ice water path, cloud base
  (`QC+QI ≥ 1e-5`), inversion height and strength (300–3500 m, ≥ 0.002 K/m), each with a status field.
- Profiles at the nearest mass point inside the nest: height, θ, T, p, `QV QCLOUD QICE CLDFRA`, `QNCLOUD`
  (Thompson aerosol-aware only), winds.
- Every point gets a status per nest: `OK`, `NO_COORDINATE`, `OUTSIDE_DOMAIN`, `BEYOND_MAX_DISTANCE` or
  `GEOMETRY_MISMATCH`.

Quality checks fail the extract (never `READY`) on non-finite or fill values, on negative water species
below −1e-6, and on fields no producer supplies (a writer fallback value is reported as missing, not as data).

## Restart and extension

The extract request is bound to the checkpoint: a resume must use the same request, and a changed request is
refused before the model starts. Extending a completed run writes a new `segment_NN/` with only the new frames;
published files are never touched.

## Offline and legacy formats

- `python -m gpuwrf.extract.offline` recomputes the same extract from existing `wrfout` files (also CPU-WRF output;
  `ZNT` must be present).
- `python -m gpuwrf.extract.legacy {sfc,cloud,wt1}` converts an extract to the older `dNN_sfc.npz`,
  `d03_cloud.npz` and profile-parquet formats.

## Cost and limits

Host work per frame is about 0.2 s for the grid fields and 0.4 s for 1,751 points on a 267×117 nest (CPU
measurement); with `raw_history off` the GPU→host transfer drops to the extract inputs (d02: 80 MB instead of
211 MB per frame). GPU wall-clock cost and host memory on the GPU have not been measured separately.
Not supported: batched members (B>1), branches, a 15-minute d03 cadence different from the history interval.
