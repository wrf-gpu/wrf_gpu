# CANARY NESTED TRAINING GRID — MEASURED CPU/GPU 24h ESTIMATES (2026-06-14)

**Author:** Opus 4.8 CPU/GPU benchmark lead (autonomous)
**Hardware:** AMD Ryzen 9 9950X **16 physical cores / 32 threads**; single RTX 5090 32 GiB (Blackwell).
**Status:** MEASURED. Supersedes the QUICK projection (`canary_nested_estimates_QUICK.txt`,
which used an over-subscribed 28-rank basis and area-scaling → ~46 h; that number is now retired).

---

## GRID (FINAL — AC1_FIT, not redesigned)

Nested 9 km (all) → 3 km (all) → 1 km (all-7-islands), one-way (feedback=0), fp64,
`GPUWRF_MYNN_BOULAC_ONZ=1`, center 28.535 N / −15.755 W, e_vert=45,
parent_grid_ratio 3,3, parent_time_step_ratio 3,3, time_step=18 s (→ d02 6 s, d03 2 s):

| dom | dx | e_we×e_sn | mass cols | cu | i/j_parent_start |
|---|---|---|---|---|---|
| d01 | 9 km | 92×64 | 91×63 = 5,733 | 1 | 1,1 |
| d02 | 3 km | 208×124 | 207×123 = 25,461 | 0 | 10,12 |
| d03 | 1 km | 520×280 | 519×279 = 144,801 | 0 | 18,16 |

Physics = operational suite: mp=8, bl=5, **sf_sfclay=5** (operational reference uses 5,
not 4), sf_surface=4 (Noah LSM), ra_lw=ra_sw=4, cu=1 (d01 only), gwd_opt=1.

---

## CPU — 12 MPI RANKS, `taskset -c 4-15` (MEASURED, the correct non-oversubscribed baseline)

Built fresh for THIS grid (operational nvhpc dmpar toolchain still present):
geogrid (3 dom) + metgrid (reused existing 2026-04-28_18z AIFS ungrib, 14 metgrid /
2 soil levels) + real.exe → wrfinput_d01/02/03 + wrfbdy_d01. Then a warm window of
`mpirun -np 12 --oversubscribe --bind-to none` wrf.exe pinned to physical cores 4-15
(`OMP_NUM_THREADS=1`). Steady step times read from `rsl.error.0000` (first 3 steps
dropped as warmup).

(One benchmark-only IC fix: 35 edge cells of d02 SKINTEMP came in as 0 K from the AIFS
metgrid mask → nearest-valid fill, ~287 K; does not change column count or step cost.)

**Measured warm per-domain step times (105 warm d03 steps; d03 std 0.083 s = 4.5%):**

| dom | cols | warm median | warm mean | per-col (own-solve) |
|---|---|---|---|---|
| d01 9 km | 5,733 | 19.044 s/step* | 19.145 s | 19.2 µs/col |
| d02 3 km | 25,461 | 6.279 s/step* | 6.317 s | 29.8 µs/col |
| d03 1 km | 144,801 | **1.837 s/step** | 1.862 s | 12.6 µs/col |

\* In WRF the parent step time is **recursive-inclusive** — the d01 step time already
contains its 3 d02 sub-steps, and each d02 step already contains its 3 d03 sub-steps.
Verified exactly: 3×d02 (18.75 s) + d01-own (0.11 s) = 18.86 s = measured d01 step;
3×d03 (5.49 s) + d02-own (0.76 s) = 6.25 s = measured d02 step.

**24 h CPU wall (12-rank) = 4,800 d01-steps × 19.04 s/step (recursive-inclusive)**

| | hours | share |
|---|---|---|
| **TOTAL 24 h** | **~25.4 h** | 100% |
| d03 1 km (43,200 steps × 1.84 s) | ~22.0 h | 88% |
| d02 3 km (own, 14,400 × 0.76 s) | ~3.0 h | 12% |
| d01 9 km (own, 4,800 × 0.11 s) | ~0.1 h | <1% |

**Confidence: HIGH.** Real 12-rank run of the exact AC1_FIT grid + operational physics
on this exact box; 105 steady warm d03 samples (low variance). Cross-check: the same
12-rank dmpar build measured ~17.5 µs/col on a 461×461 3 km single domain
(`v017_bigswiss_cpu16…`); d03 1 km here is 12.6 µs/col (1 km is lighter per-col than
3 km at this rank count — fewer halo/comm cells per interior cell). Consistent.

> Note: 28-rank OVERSUBSCRIBES a 16-core box (the retired QUICK number). 12 ranks on
> 12 physical cores (cores 4-15) is faster *and* the apples-to-apples baseline.

---

## GPU — WARM (MEASURED warm step; estimate by step×steps)

**Capability verdict (honest separation):**

| GPU capability | status | evidence |
|---|---|---|
| AC1_FIT d03 1 km step (145k cols), standalone | **MEASURED here** | 557 ms/step warm, **21.40 GiB peak**, finite, fp64, BouLac-ONZ |
| live-coupled 9/3/1 to 24 h completion (small oper nest) | **PROVEN** (one-way, GWD off) | `proofs/v0120/nested_24h_1km_gate_FINAL.json`: PIPELINE_GREEN, 4800 steps, ~2.0 h wall, ~7k-col d03 |
| live-coupled 9/3/1 **at AC1_FIT (145k-col d03) + GWD**, from IC | **NOT run / unlikely to fit** | AC1_FIT d03 alone = 21.4 GiB; live d01+d02+LBC+GWD on top would push toward the 32 GiB ceiling (GWD already exceeds fp64 VRAM at ~hr7 on the *small* nest). Realistic GPU mode for AC1_FIT = **standalone / parent-driven d03 1 km**, NOT full live-coupled-with-GWD. |

So the GPU number below is the **standalone AC1_FIT d03 1 km** path (the GPU's real value
here = 1 km all-islands capability), labelled honestly — NOT a measured live-coupled
9/3/1 run.

**MEASURED (under GPU lock, branch `worker/perf/v017-rc` @fa0ca55c / `.wt-rc`,
`GPUWRF_MYNN_BOULAC_ONZ=1`, fp64, harness `proofs/perf/v017/boulac_onz_bench.py`
tiled to the AC1_FIT d03 shape 525×279 = 146,475 cols):**

- warm = **556.99 ms/step** (best-of-3: 4.456 / 4.473 / 4.472 s per 8 steps)
- **peak VRAM = 21.40 GiB** → FITS 32 GiB, thin headroom (consistent with the
  prior 18.84 GiB measured-fit; the higher figure here reflects fresh-state tiling +
  warm reps and confirms the "thin-margin" caveat).
- cold/compile ~141 s (one-time JIT, excluded from warm rate).
- output finite. Result: `…/nested_canary_training/ac1fit_gpu_d03_bench.json`.

**24 h GPU estimate (d03 is the bottleneck; d01/d02 add <0.3 h):**

| d03 dt | steps/24h | GPU 24 h (d03) | note |
|---|---|---|---|
| 2 s (matched to CPU) | 43,200 | **~6.7 h** | apples-to-apples with the 12-rank CPU run |
| 3 s (operational 1 km) | 28,800 | **~4.5 h** | the validated 1 km production dt |
| 4 s (larger validated) | 21,600 | ~3.3 h | upper-dt |

**Confidence: MEDIUM-HIGH** on the step rate (measured warm on this card/branch at the
exact AC1_FIT d03 size, reproduces the prior 590 ms point). MEDIUM on 24 h: the bench
step omits GWD/Noah-MP/LBC (tiling helper), so the real standalone-d03 step is somewhat
heavier; and full live-coupled-9/3/1-at-AC1_FIT is not demonstrated (capability caveat above).

---

## FACTOR = 12-rank CPU 24 h / GPU warm 24 h

| comparison | CPU | GPU | **factor** |
|---|---|---|---|
| **matched dt (d03=2 s)** | 25.4 h | ~6.7 h | **~3.8×** |
| operational dt (d03=3 s) | 25.4 h | ~4.5 h | **~5.7×** |
| larger dt (d03=4 s) | 25.4 h | ~3.3 h | ~7.6× |

**Honest headline: ~3.8× at matched numerics, ~5-6× at operational 1 km dt** — over the
correct **12-rank** CPU baseline (not the retired 28-rank-oversubscribed ~46 h / ~7-10×
quick projection). The real GPU value here is **1 km all-islands capability** (one card,
one night, ~4.5-6.7 h) — covering Fuerteventura/Lanzarote that the operational 1 km nests
do not — and **cluster weak-scaling**, NOT a single-GPU fp64 raw-speed miracle.

---

## ARTIFACTS

- CPU case (geo_em / met_em / wrfinput / wrfbdy / namelist / rsl):
  `<DATA_ROOT>/wrf_downscale/nested_canary_training/ac1fit_20260614T220802Z/`
  (`run_cpu/rsl.error.0000` = the measured 12-rank timing; `run_cpu/namelist.input` = AC1_FIT operational namelist)
- GPU d03 bench: `<DATA_ROOT>/wrf_downscale/nested_canary_training/ac1fit_gpu_d03_bench.json`
- Predecessor GPU VRAM/step ladders (corroborating):
  `<DATA_ROOT>/wrf_downscale/grid_feasibility/final_canary_training_grid/benchmark_results.json`
- Live-nested 9/3/1 capability proof: `proofs/v0120/nested_24h_1km_gate_FINAL.json`
- Toolchain (still present, contra the "WRF build down" note):
  WRF dmpar `…/canairy_meteo/Gen2/artifacts/wrf_src/WRF/install_gen2_dmpar/run/{real,wrf}.exe`;
  WPS `…/WPS/install_gen2_dmpar/bin/{geogrid,metgrid,ungrib}.exe`;
  mpirun `…/Gen2/artifacts/envs/wrf-build/bin/mpirun`.
