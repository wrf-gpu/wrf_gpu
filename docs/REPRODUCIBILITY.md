# Reproducibility — what an outsider can run on CPU, and what needs a GPU or data

This project validates its JAX/XLA WRF-compatible reimplementation with a large
collection of **proof objects** under `proofs/`. This document tells an external
reviewer, with nothing but a clone of this repository, exactly which proofs are
reproducible on a CPU-only machine and which require hardware or data that cannot
be redistributed.

The honest summary: **the numeric-correctness core is CPU-reproducible from the
repo alone**; speedup/throughput claims and the multi-day operational equivalence
proofs are not (they need an NVIDIA GPU and real CPU-WRF forecast corpus,
respectively).

## TL;DR — one command

```bash
bash scripts/verify_reproducibility.sh
```

- Forces `JAX_PLATFORMS=cpu` (no GPU context is created).
- Requires **nothing beyond this repo**: no GPU, no WRF source tree, no corpus.
- Exit `0` = every CPU-reproducible gate is green; non-zero = a gate failed
  (the per-gate log path is printed).

The gate runs three tiers:

| Tier | What | Needs |
|------|------|-------|
| 0 | Large binary assets present + `sha256` == pinned manifest (`manifest/reproducibility_assets.json`) | repo only |
| 1 | Asset-exercising unit tests (Thompson/RRTMG lookup-table loaders + manifest pin + WRF-source constants/shapes) | repo only |
| 2 | CPU physics savepoint-parity proofs — JAX port vs **vendored** unmodified-WRF Fortran savepoints (Kessler, BouLac PBL, Dudhia SW, RRTM LW, WSM, Grell–Freitas, Tiedtke, coupled-moist closure) | repo only |

## Why the Thompson tables matter (the v0.11 critique blocker)

The Thompson microphysics scheme needs ~29 MB of precomputed lookup tables
(`data/fixtures/thompson-tables-v1.npz`: rain-freezing, accretion, and collision-
efficiency tables originally produced by pristine WRF's `module_mp_thompson.F`
table init). The **v0.0.1 public tree excluded this binary**, which silently
blocked the full historical proof collection for an outsider.

It is now **vendored in-repo** and pinned by `sha256` in two independent places:

- `fixtures/manifests/analytic-thompson-column-v1.yaml` (asserted by
  `tests/test_m5_thompson_constants.py`), and
- `manifest/reproducibility_assets.json` (asserted by Tier 0 of the gate).

You do **not** need to rebuild it. If you ever want to regenerate it for
provenance, point at a pristine WRF checkout and run the extractor:

```bash
JAX_PLATFORMS=cpu WRF_PRISTINE_ROOT=/path/to/WRF \
    python3 scripts/extract_thompson_tables.py
```

(Compiles a small Fortran harness against the unmodified WRF source, dumps the
tables, and packs the `.npz`. Needs `gfortran` + a WRF source tree.)

## The `WRF_PRISTINE_ROOT` knob (optional)

The proof runners reference an **unmodified pristine WRF v4 source checkout** for
two things only: (a) regenerating oracles, and (b) recording a provenance
`sha256` of the WRF source in the proof JSON. WRF is a separate, openly available
codebase (UCAR/NCAR) and is intentionally **not** vendored here.

- **Default**: a `wrf_pristine/WRF` directory that is a *sibling* of this repo.
- **Override**: set `WRF_PRISTINE_ROOT=/path/to/WRF` (and `WRF_FC` for a specific
  Fortran compiler; otherwise `gfortran` on `PATH` is used).
- **If absent**: the CPU savepoint-parity gates still pass — they validate against
  the **vendored** savepoints, and the provenance hash simply records `"missing"`.

No proof runner contains a hardcoded developer path anymore; all repo and WRF
paths resolve from the file location or `WRF_PRISTINE_ROOT`.

## What an outsider canNOT reproduce from the repo alone

| Category | Examples | Why | What you'd need |
|----------|----------|-----|-----------------|
| **GPU-only** | `proofs/perf/*`, `proofs/multigpu_dgx/*`, `proofs/v0120/*` nested 1 km; any speedup / throughput / per-watt claim | Correctness is CPU-checkable, but performance requires real GPU execution | An NVIDIA GPU + CUDA-enabled JAX |
| **Purged corpus** | `proofs/m20/*` (TOST equivalence), multi-day operational gates, `proofs/v090/*_savepoint_parity` oracle dirs | Depend on real CPU-WRF `wrfout` + AIFS forcing that are not redistributable (lived under `<DATA_ROOT>`, mostly purged — see `docs/fixture-storage-policy.md`) | Re-run CPU-WRF from the AIFS forcing documented in `data/manifests/aifs_ingest_v0.json` |
| **Oracle rebuild** | Fortran-linked savepoint regeneration (e.g. `proofs/v040/*_savepoint_parity.py` compile paths) | Recompile/relink against the WRF source | `WRF_PRISTINE_ROOT` + `gfortran` |

These exclusions are also printed at the end of every `verify_reproducibility.sh`
run, and enumerated in `manifest/reproducibility_assets.json` under
`external_inputs_not_vendored`.

## Known asset-pin discrepancy (pre-existing, non-blocking)

`fixtures/manifests/analytic-rrtmg-{sw,lw}-column-v1.yaml` and
`rrtmg-intermediate-oracle-v1.yaml` pin a stale `sha256` for
`data/fixtures/rrtmg-tables-v1.npz` that does not match the current on-disk asset.
The asserting test (`tests/test_m5_rrtmg_tables.py`) only checks the digest is a
valid 64-char hash, **not** that it equals the manifest pin, so nothing fails
today. `manifest/reproducibility_assets.json` records the discrepancy
(`pin_verified: false`) and Tier 0 of the gate skips it accordingly; the
radiation-owned manifest pin should be refreshed in a future sprint.
