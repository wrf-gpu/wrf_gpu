# `data/` — runtime fixtures and manifests

This directory holds the **runtime lookup tables** that the GPU physics kernels
load at import/first-use, plus a couple of small ingest/observation manifests.
Everything here is a **required runtime asset** (loaded by `src/gpuwrf/...`),
not dev-only scratch — the model will not run faithfully without it.

These tables are **bit-exact extracts of the pristine WRF v4 Fortran lookup
tables** (the same `.dat`/`.BIN` contents WRF itself loads), captured once by a
build/extract script and vendored so the port is reproducible offline and does
not recompute them at runtime. They are tracked in git (no Git-LFS) — consistent
with the prior public release, which already shipped a comparable ~28 MiB
Thompson table (`thompson-tables-v1.npz`).

## `data/fixtures/`

| File | Size | Purpose | Loaded by / built by |
|---|---:|---|---|
| `thompson-tables-v1.npz` | ~28 MiB | Thompson (`mp_physics=8`) microphysics lookup tables (rain/graupel gamma families, 4-D rain-freezing `t*_qrfz`, etc.). **Shipped since the prior public release.** | `src/gpuwrf/physics/thompson_tables.py` |
| `thompson-cold-collection-v1.npz` | ~70 MiB | **v0.15** cold-collection tables: rain-collecting-snow (`qr_acr_qs`), rain-collecting-graupel (`qr_acr_qg`), and Bigg rain-freezing (`freezeH2O`). Bit-exact extract of the pristine WRF `.dat` tables (the Fortran-computed contents, not a recomputation). Required for the cold-collection WRF-fidelity fix. | `src/gpuwrf/physics/thompson_tables.py` (`COLD_TABLE_ASSET`) / `proofs/v015/cold_collection_oracle/extract_collision_tables.py` |
| `thompson-aero-tables-v1.npz` | ~26 MiB | **v0.16** aerosol-aware Thompson (`mp_physics=28`) tables: CCN activation table (verbatim WRF `CCN_ACTIVATE.BIN`), droplet-evap number table (`tnc_wev`), heterogeneous cloud-freezing tables (`tpi/tni_qcfz`), and the variable-`nu_c` cloud gamma families. Required for the aerosol-aware Thompson "+1". | `src/gpuwrf/physics/thompson_aero_tables.py` / `scripts/build_thompson_aero_tables.py` |
| `rrtmg-tables-v1.npz` | ~4 MiB | RRTMG SW/LW radiation lookup tables. | RRTMG radiation kernels |
| `rrtmg-tables-v1.json` | ~1 KiB | RRTMG table manifest/metadata. | RRTMG radiation kernels |
| `rrtmg-intermediate-oracle-v1.npz` | ~0.25 MiB | RRTMG intermediate-stage oracle fixture (validation). | RRTMG oracle tests |
| `analytic-rrtmg-{sw,lw}-column-v1/full.npz` | ~10 KiB each | Analytic single-column RRTMG SW/LW oracle samples. | radiation oracle tests |
| `gen2_baseline/rmse_summary.csv` | small | Gen2 CPU-WRF baseline RMSE summary (validation reference). | validation harness |
| `m6/d02_boundary_replay_v1.zarr/` | small (zarr) | d02 boundary-replay slice for the replay path. | replay harness |
| `tier3_idealized/` | small | Tier-3 idealized-case fixtures. | idealized-case tests |

### Why the two largest tables are kept (not excluded)

`thompson-cold-collection-v1.npz` (70 MiB) and `thompson-aero-tables-v1.npz`
(26 MiB) are **required runtime fixtures** for the v0.15 cold-collection fix and
the v0.16 aerosol-aware Thompson (`mp_physics=28`) scheme respectively — the
kernels load them directly. They are bit-exact captures of the pristine WRF
Fortran tables, so they cannot be cheaply regenerated at runtime without a WRF
build. The prior public release already shipped a comparable ~28 MiB Thompson
table, so vendoring these in-tree (rather than Git-LFS or external release
assets) follows the established convention. They are intentionally **kept** in
the clean public tree.

## `data/manifests/`

Small JSON manifests for data ingest and station-observation sources
(`aifs_ingest_v0.json`, `station_obs_sources_v0.json`).
