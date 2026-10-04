# bigswiss GPU 13h benchmark — v0.17 release "benchmark compare"

**Date:** 2026-06-14 · **Branch:** `worker/perf/v017-rc` (worktree `.wt-rc`) ·
**Device:** NVIDIA GeForce RTX 5090 (GB202, 32607 MiB) · **Precision:** fp64 ·
`GPUWRF_MYNN_BOULAC_ONZ=1`

## Headline result

**BLOCKED — bigswiss does NOT fit a single 32 GB RTX 5090 in fp64.**
No measured warm 13h GPU wallclock and no CPU/GPU factor can be produced on one
GPU for this grid. The failure is a **VRAM out-of-memory**, **not** a NaN/CFL
instability — the forecast never advanced a step.

| Quantity | Value |
|---|---|
| Grid | 461×461×45, dx=3 km, **211,600 mass cols** |
| GPU warm 13h wallclock | **NOT MEASURABLE (OOM)** |
| CPU/GPU factor | **N/A (no GPU wall)** |
| GPU peak VRAM | hit single-op alloc of **22.28 GiB** (default) / **21.35 GiB** (max-tiled) → OOM |
| ms/step (cold/warm) | N/A — never reached the step loop |
| Stability / equivalence | not reached (0 wrfout; OOM before hour-1 output) |
| CPU reference (given, verified) | 12-rank, cores 4-15: **3.346 h** (h0→h13 mtime = 12046.7 s) |

## What was run (honest setup)

- **True forecast from IC+LBC** (`standalone_native_init`): IC from
  `wrfinput_d01`, LBC from `wrfbdy_d01`. **Not** a CPU-wrfout replay / self-compare.
  - The provided `run_cpu/` dir holds 16 hourly CPU `wrfout` files, which made the
    pipeline auto-select **`cpu_wrf_replay`** (≥2 wrfout ⇒ replay). To force a real
    standalone forecast I built an **init-only dir**
    `<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init/` (symlinks to
    `wrfinput_d01` + `wrfbdy_d01` + `namelist.input` + physics tables, **no**
    wrfout) → `detect_init_mode = standalone_native_init`. Confirmed before launch.
- **Physics identical to the CPU run, none silently omitted.**
  `validate_operational_namelist` → **PASS, zero warnings**: Thompson MP (mp=8),
  RRTMG-fast LW/SW (ra=4), MYNN surface layer (sf_sfclay=5), **Noah-MP** land
  (sf_surface=4), MYNN PBL (bl_pbl=5), no cumulus (cu=0). fp64 acoustic solve
  (force_fp64, as the operational path pins).
- GPU lock held for all GPU work (`scripts/with_gpu_lock.sh --label bigswiss-gpu-13h`).
  CPU pinned to cores 0-3, OMP=4.

## Two independent OOM runs (the ceiling is hard, not allocator/fragmentation)

| Run | Allocator | Column-tiling | Big single-op alloc | Outcome |
|---|---|---|---|---|
| 1 (default) | BFC (XLA default) | RRTMG/MYNN @16384; **MP un-tiled** | **22.28 GiB** | OOM @231 s |
| 2 (mitigation) | **platform** (cudaMalloc) | **MP=8192, MYNN=8192, RRTMG=8192** | **21.35 GiB** | OOM @213 s |

Resident "in use" peaked at only **~9 GiB**; the kill was a **single contiguous
transient** of 21-22 GiB. Maximally column-tiling every column-physics scheme
(MP, MYNN, RRTMG) only shaved 22.28 → 21.35 GiB. The residual is **not** one
un-chunked physics giant — it is the **sum of un-tiled full-grid `(ncol, nz)`
arrays**: the prognostic State (~30 leaves), the acoustic/dycore work arrays, and
the per-RK-stage tendencies. **No env knob caps these.**

## Corroboration: this is the already-documented v0.15 ceiling

`.wt-rc/proofs/perf/v015/km_bench/` measured peak VRAM as **LINEAR in ncol**
(0.1654 GiB / 1000 col, R² 0.997):

| Grid | ncol | Projected/measured peak | Fits 32 GB? |
|---|---|---|---|
| 560×280 (1km-unlock "small") | 156,800 | 24.4 GiB | **YES** |
| largest that ran (264×636) | 167,904 | 26.25 GiB (measured) | YES |
| first documented OOM (264×795) | 209,880 | 28.44 GiB single alloc | NO |
| **bigswiss 460×460** | **211,600** | **~33.5 GiB** | **NO** |

bigswiss (211,600 cols) is essentially the v0.15 "first-OOM" grid (209,880 cols).
The measured 22.28/21.35 GiB single-op OOM is exactly the predicted ceiling.
`vram_ceiling_findings.json` already flagged the fix (column-tile the MP adapter +
dycore work, ~4-5 GiB savings) as an **unimplemented, honestly-deferred** task —
and even that ~4-5 GiB is short of the ~5+ GiB bigswiss overshoot.

## Projection if it fit (NOT a measurement)

The v0.15 core throughput fit (a=19.58 ms + b=4.937 ms/1000col, R² 0.997;
**core = dycore+rad+MYNN, EXCLUDES boundary relaxation, Noah-MP, full Thompson MP,
output I/O**) gives:

- core ms/step ≈ **1064 ms**; core warm 13h wall ≈ **0.77 h**; core factor vs CPU
  3.346 h ≈ **4.35×**.

This is a **projection only**. The real warm wall with full physics + boundary +
I/O would be slower (smaller factor), and it **cannot be measured** on one GPU
because the grid OOMs. Do **not** put this number in a release headline as measured.

## Recommendation for the v0.17 headline

1. **Pick a grid that fits** — ≤ ~167k cols, e.g. **560×280 (156,800 cols)**, the
   proven single-GPU 1km-unlock fits-ceiling (24.4 GiB). Then a real warm
   wallclock + measured CPU/GPU factor is obtainable on this RTX 5090.
2. **Or** benchmark bigswiss on **multi-GPU** — per the project anchor, bigswiss-class
   value is **1km-capability + cluster weak-scaling**, not single-GPU speed.
3. **Or** land the deferred MP-adapter + dycore-work column-tiling (still likely
   short of the bigswiss overshoot; needs its own before/after VRAM gate).

## Artifacts

- This proof + machine JSON: `proofs/v017/bigswiss_gpu_benchmark.md` / `.json`
- Run logs: `<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_bench/bench.log`,
  `…/bench_mitig.log`
- Pipeline blocked record:
  `<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_bench/proofs/pipeline_run_20260521.json`
- Init-only run dir: `<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init/`
- Driver: `.wt-rc/scripts/v017_bigswiss_gpu_bench.py`
- v0.15 VRAM ceiling evidence:
  `.wt-rc/proofs/perf/v015/km_bench/{km_feasibility_verdict,vram_ceiling_findings,localize_18g_alloc}.json`
