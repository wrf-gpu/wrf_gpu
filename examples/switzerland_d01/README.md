# Bundled example case — Switzerland 3 km (d01)

A small, self-contained real-data case so a fresh clone can run an end-to-end GPU
forecast with no external data download. This is the case behind the
**"Switzerland d01"** identity proof on the project front page.

## What's here

| File | Size | What it is |
| --- | --- | --- |
| `wrfinput_d01` | ~3.8 MB | WRF initial condition (one time level) |
| `wrfbdy_d01` | ~9.6 MB | Lateral boundary tendencies for the run window |
| `namelist.input` | ~4 KB | The WRF namelist (physics + domain) for this case |

Domain: **42 × 42 mass points, 44 vertical levels, Δx = Δy = 3 km**, single domain.
Init: **2023-01-15 00:00 UTC**. Boundary interval: 3-hourly.
Physics menu: `mp_physics=8` (Thompson), `ra_lw/sw=4` (RRTMG), `sf_surface=4`
(Noah-MP), `bl_pbl=5` / `sf_sfclay=5` (MYNN), `cu_physics=0`.

## Provenance & license

The fields are derived from **NCEP GFS** analysis (US Government, public domain)
through the standard WPS / `real.exe` preprocessing chain (`TITLE = OUTPUT FROM
REAL_EM V4.7.1 PREPROCESSOR`, `SIMULATION_INITIALIZATION_TYPE = REAL-DATA CASE`).
GFS products are public domain, so these derived inputs are freely redistributable.

## Run it

This case selects RRTMG radiation and Noah-MP, which read lookup tables from a
pristine WRF v4 install, so set `GPUWRF_WRF_ROOT` first (see the top-level
[README Quickstart](../../README.md#quickstart)):

```bash
export GPUWRF_WRF_ROOT=/path/to/your/WRF      # your pristine WRF v4 source/run tree
python -m gpuwrf.cli run \
    --input-dir   examples/switzerland_d01 \
    --output-dir  runs/switzerland_d01 \
    --domain      d01 \
    --hours       1 \
    --scratch-dir /tmp/gpuwrf_scratch         # any real (non-tmpfs) fast disk

ncdump -h runs/switzerland_d01/wrfout_d01_*
```

`--hours` can be raised (the namelist window covers 24 h). The first run pays a
one-time XLA cold compile (~½–2 min for this domain); later runs hit the JIT cache.
