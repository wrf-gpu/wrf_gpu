# v0.18 AC1_FIT d03 (1 km nest) GPU OOM — Root-Cause (measurement only)

**Date:** 2026-06-18  **Device:** RTX 5090 (32 GiB, sm_120), fp64, JAX 0.10.0
**Case:** AC1_FIT 9/3/1 nest, d03 = 520×280×45 (144,801 mass columns × 44 layers).
**Scope:** measurement + localization only — no `src/` edits.

---

## TL;DR (the WHY answer)

1. **The OOM is REAL and was REPRODUCED.** Running the exact command
   (`python -m gpuwrf run --max-dom 3 --hours 1`) drove device usage to **31.8 GiB
   of 32 GiB** and failed with a driver-level **`Failed to allocate 12.72 GiB
   (13,659,682,048 bytes)`** — i.e. essentially the reported ~13.7 GiB failing
   allocation — plus recurring BFC **2.09 GiB** requests that could not find a
   contiguous block. (`real_run_oom.log`)

2. **It is a transient-WORKING-SET + FRAGMENTATION problem, not a fundamental
   need.** The 12.72 GiB request is XLA reserving one big *contiguous
   preallocated-temp scratch arena* for the d03 program; it does not fit on top
   of the resident multi-domain state + a ~5-GiB-class radiation transient +
   dynamics scratch + ~2.9 GiB held by other processes.

3. **The dominant single transient is RRTMG-LW**, and inside it the dominant
   buffer is the **McICA cloud-optics pair `cldfmc` + `taucmc`** — each
   `f64[tile, nlay, 16 bands, 16 g-points]` = **1.41 GiB at tile=16384 (2.81 GiB
   for the pair)** — built up front in `_lw_cldprmc_state` and held live across
   the entire 16-band flux scan. Secondary LW buffer: the McICA random mask
   `f64[6300, tile]` = **0.77 GiB**. (`lw_d03_buffer-assignment.txt`, allocation
   199 = the 4.89 GiB preallocated-temp arena.)

4. **The column tile-scan WORKS.** LW and SW per-call peak drop ~linearly with
   `tile_cols`. **The fix is mostly a smaller default tile (env knob, bit-identical).**

---

## Pre-flight blocker found (important, but not the OOM)

The committed **v0.18.1 working tree cannot run the command as-is** — it dies at
*import* before any GPU op:

- `data/fixtures/thompson-cold-collection-v1.npz` has 16 keys but
  `thompson_tables.COLD_TABLE_NAMES` expects 18 → `KeyError: 'tpi_qcfz ...'`.
- `data/fixtures/thompson-aero-tables-v1.npz` is **absent** →
  `THOMPSON_AERO_TABLES = load_thompson_aero_tables()` raises at import.

`gpuwrf/physics/__init__` imports Thompson eagerly, so even mp=8 runs and any
RRTMG import fail. **This means the production OOM happened on a tree where these
assets matched the code; the released tree as-checked-out cannot reproduce it
without the assets.** For measurement I used a PYTHONPATH `sitecustomize` shim
(`shim/`) that injects the 2 missing cold keys as zeros and redirects the aero
asset to a zero-stub. mp=8 never traces the aero path at runtime, so the only
effect is zeroing the freezeH2O cloud-water-freezing source term — irrelevant to
VRAM. **No `src/` file was modified.** (Manager: the fixture/code mismatch is a
real ship blocker that should be fixed separately.)

---

## The decisive experiment — d03 tile-scan sweep (standalone, fp64)

Faithful d03-scale (144,801 cols × 44 layers) calls into the public solvers,
device `peak_bytes_in_use` per process (BFC allocator — the production `platform`
allocator does **not** populate memory stats):

| component | tile=16384 | 8192 | 4096 | 2048 | floor (O(ncol)) | per-16384 slope term |
|-----------|-----------:|-----:|-----:|-----:|----------------:|---------------------:|
| **LW** transient GiB | **5.06** | 3.07 | 1.67 | **1.23** | 0.67 | 4.45 |
| **SW** transient GiB | **1.67** | 1.17 | 0.86 | **0.82** | 0.66 | 1.01 |
| **MYNN** transient GiB | **3.32** | 2.91 | 2.73 | **2.45** | **2.42** | 0.92 |

Linear fit `transient = floor + slope·tile_cols`:

- **LW**: tile-scaling term DOMINATES (4.45 of 5.06 GiB) → finer tiling cuts peak
  ~linearly. **Tiling works.**
- **SW**: small, mostly a fixed ~0.66 GiB floor.
- **MYNN**: dominated by a **2.42 GiB O(ncol) FLOOR that tiling does NOT remove.**

So the answer to "is there a residual O(ncol) buffer that bypasses the tile
scan?": **For LW/SW — no, the peak is O(tile_cols) and the scan is doing its
job.** **For MYNN — yes**, there is a tiling-immune full-column floor (see below),
but MYNN is not the OOM driver (LW is larger and fires on the same step).

---

## Named dominant buffers (from the XLA buffer-assignment)

### RRTMG-LW (`lw_d03_buffer-assignment.txt`, tile=16384)

`allocation 199: preallocated-temp = 4.89 GiB` holds the live transient set. Top
families inside it:

| shape | each | live | total | what it is |
|-------|-----:|-----:|------:|------------|
| `f64[16,45,16384,16]` | **1.41 GiB** | 2 | **2.81 GiB** | **McICA `cldfmc` + `taucmc`** — `(16 bands, nlay, tile, 16 g-points)`. Built in `_lw_cldprmc_state` (`rrtmg_lw.py:1920-1946`) BEFORE the band scan; every band reads `jnp.take(cldfmc/taucmc, band, axis=-2)` so the **full 16-band stack is pinned live across the whole scan.** |
| `f64[6300,16384]` | **0.77 GiB** | — | 0.77 | **McICA random cloud mask** `random_flat` from `lax.scan(length=140*nlay=6300)` (`_lw_mcica_random_cloud_mask`, `rrtmg_lw.py:1893-1913`). |
| `f64[16384,45,16]` | 0.09 GiB | (subset) | per-band | per-band `_lw_taumol_band` tau/frac + rtrnmc temporaries INSIDE the band scan — correctly freed by the scan carry barrier (O(tile), as designed). |

**Key insight:** the `_LW_TAUMOL_CHUNK` band-scan successfully avoids the 16-band
*taumol* stack, but the McICA *cloud* arrays (`cldfmc`/`taucmc`) are **not**
chunked the same way — they are the dominant residual LW buffer.

### RRTMG-SW (`sw_d03_after_optimizations_hlo.txt`)

Largest tile-scale buffer `f32[16384,45,14,12]` (14 bands × 12 g-points, **fp32**)
≈ 0.46 GiB each — the SW two-stream g-point stack. SW uses fp32, so it is ~half
LW per element and a smaller contributor.

### MYNN-PBL (the 2.42 GiB tiling-immune floor)

`_tiled_mynn_step` (`mynn_pbl.py:1810-1844`) pads **all ~30 state leaves** to
`(padded_ncol, nz)` for **both** the input `padded_state` **and** `init_state`
(`tree_map(zeros_like, …)`), and `lax.scan` scatters per-tile outputs into the
full padded carry. ~30 leaves × 147,456 × 45 × 8 B ≈ 1.5 GiB, ×2 (in+out) ≈
**2.4 GiB** — matching the measured floor. The dense BouLac `(tile, nz, nz)`
matrices (default; `GPUWRF_MYNN_BOULAC_ONZ=0`) are only the **slope** term
(~0.92 GiB at 16384), not the floor.

---

## The real-run OOM, decoded (`real_run_oom.log`)

- `cuda_executor: Failed to allocate 12.72 GiB (13,659,682,048 B)` — XLA reserving
  a **single contiguous whole-program temp arena** (same arena *class* as LW
  allocation 199, but for the larger nested/d03 program).
- `GPU_0_bfc ran out of memory trying to allocate 2.09 GiB` (repeated) — the
  recurring **per-step radiation/PBL transient** that can't find a contiguous hole
  once the pool is occupied; the BFC map `***___***___***` shows the fragmentation.
- **smi peaked at 31.8 GiB / 32 GiB**, with ~2.9 GiB held by *other* processes,
  i.e. effectively ~29 GiB available.

So the failure is: `resident multi-domain state + a ~5-GiB-class d03 LW transient
+ d03 dynamics/acoustic scratch + XLA arena padding ≈ 30 GiB`, at which point the
next big contiguous request (12.72 GiB arena, or 2.09 GiB block) cannot be
satisfied. **Not one fundamental 12.72 GiB buffer.** The shipped mitigation
(platform/cudaMalloc allocator + per-output-hour segmentation in
`nested_pipeline.py`) attacks fragmentation but leaves the ~5 GiB transient
working set unchanged — and that working set is what finer tiling shrinks.

> Limitation: a buffer-assignment for the *full nested* program could not be
> captured because that program OOMs in compile/first-exec. Dominant-buffer
> naming is from the standalone d03-scale programs, which are faithful (the
> coupler feeds the solvers flattened `(ncol, nz)` views; no extra full-ncol
> arrays are built outside the tiled solve — verified in `physics_couplers.py`).

---

## Achievable peak & whether it fits

| lever | type | effect on d03 LW transient | notes |
|-------|------|---------------------------:|-------|
| `GPUWRF_RRTMG_LW/SW_COLUMN_TILE_COLS=2048` | **env only, bit-identical** | LW 5.06→**1.23 GiB**, SW 1.67→0.82 GiB | proofs/v013 prove tiling is byte-exact |
| band-chunk `cldfmc`/`taucmc` inside the existing band scan | code change | 2.81→~0.18 GiB | the single biggest LW item; not yet chunked like taumol |
| MYNN buffer-donation / in-place scatter (alias in+out carry) | code change | floor 2.42→~1.5 GiB | the tiling-immune MYNN floor |

**Estimate (env-only, tile=2048):** resident (~2.5) + largest single transient
LW (~1.23) + dynamics (~0.7) ≈ **~4.5 GiB working set** — comfortably < 32 GiB and
not fragmentation-prone. **Fits 32 GiB with wide margin.**

**Estimate (env + code levers):** ~3–4 GiB. This **approaches CPU-WRF's ~10 GiB
total RSS** — in fact the GPU resident+per-tile working set is *smaller* than CPU
WRF's 12-rank halo+tile working set, because only one tile's transient is live at
a time. The OOM is purely that the **default tile (16384) sizes the live
radiation transient to ~5 GiB and lets the nested d03 program's contiguous arena
demand exceed free VRAM** — a tuning/efficiency issue, exactly as hypothesized.

**Bottom line:** the 1 km nest does NOT fundamentally need >32 GiB. A smaller
default radiation/PBL column tile (and, for the last few GiB, band-chunking the
McICA cloud arrays + donating the MYNN carry) brings d03 peak to ~3–4.5 GiB.

---

## Artifacts in this directory

- `sweep_results.jsonl` — the 12-point tile-scan (LW/SW/MYNN × {16384,8192,4096,2048}).
- `lw_d03_buffer-assignment.txt` / `…-values.txt` — the authoritative LW buffer dump.
- `sw_d03_after_optimizations_hlo.txt`, `mem_sw_d03.prof` — SW evidence.
- `real_run_oom.log` — the REPRODUCED 12.72 GiB OOM from the real `gpuwrf run`.
- `probe_standalone.py`, `dump_buffer_assignment.py`, `run_sweep.sh`,
  `parse_buffer_assignment.py`, `shim/` — the (throwaway) measurement harness.
- `rootcause.json` — machine-readable summary.
