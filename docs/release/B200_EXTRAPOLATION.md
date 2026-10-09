**Historical precision/performance record. Current v0.3.3 uses native fp32 dynamics, with double precision only in WRF DOUBLE islands; earlier fp64 figures below describe their named older source.**

# B200 numbers — extrapolated from B200 runs on an old version

Every B200 number in the v0.3 README/plots is **[I] extrapolated from B200 runs on an old version**. This version of
wrf_gpu was never run on a B200.

## Source used (newest paper version with the B200 runs)

- `git show 43379efb5:publication/paper.tex` — 2026-07-09 12:33, branch `worker/gpt/vnext-a1-finite-detector`
  (identical file on `manager/kimi-k3-policy`, `worker/gpt/standing-gpu-resource-policy`); PDF in the same commit.
  It is the last commit touching any `.tex` in the repos; the B200 ladder is table `tab:perf` (L522–540), same numbers
  as `7261ec46d` (2026-06-23) and `PAPER_DATA_INVENTORY.md` L99–126 in `43379efb5`. `main` only has
  `publication/draft/paper.md` (2026-07-21) without B200 numbers.
- Raw B200 evidence (`<DATA_ROOT>/wrf_gpu_b200_20260623/b200_swiss_scaling/`) no longer exists; the paper table is the
  surviving record [M-historical].
- Same-version RTX 5090 ladder: `proofs/v020/benchmark/T2T3_REPORT.md` L28–36 (on `main`).

| grid (44 lev) | B200 s/fc-h | RTX 5090 fp64 s/fc-h | B200/5090 | B200 board W |
|---|---|---|---|---|
| 128² | 12.12 | 32.6 | 2.69 | 453 |
| 256² | 31.95 | 100.8 | 3.15 | 582 |
| 384² | 65.84 | 237.5 | 3.61 | 617 |
| 512²–1024² | 112.8–476.0 | OOM | — | 653–685 |
| throughput ceiling R∞ | 3.74e7 cell-updates/s | 1.06e7 | 3.53 | |

Old version = v0.20 era (ladder measured 2026-06-23), fp64 everywhere, single domain (Swiss base state tiled),
lateral boundaries, GWDO and Noah-MP off, dt ≈ 10 s.

## Method

1. Today's RTX 5090 throughput at saturation (N parallel WN3 cases, stepping) is measured [M]:
   WN3 = 572 M cell-updates per case-hour (d01 120×70, d02 267×117, d03 111×93, 44 levels, dt 54/18/6 s).
   LW9 (prototype): N = 3 → 10.97 s per case-hour → 52 M cell-updates/s, i.e. **4.9× the old version's R∞ on the same
   card** although it now runs full physics and three nests.
2. B200 = RTX 5090 × r, with r the old-version measured B200/5090 ratio: central r = 3.53 (R∞), range 2.69 (small-grid
   ratio, latency-bound) … 4.46 (HBM bandwidth ratio 8.0/1.79 TB/s, cap).
   The transfer of this ratio is an assumption: the old code ran fp32 ≈ fp64 speed on the 5090, and the LW9 static
   bandwidth analysis placed it near a bandwidth floor (FINDINGS A54, including its acoustic-overcount correction).
   Today's final kernel mix and host cost may scale differently. The range is a set of scenarios, not a confidence interval.
3. Cases per B200 = 180 decimal GB / (measured per-case peak converted from GiB + 1 GiB extra growth margin).
   The measured process peak already includes its CUDA context and modules; the extra GiB allows for different B200
   allocations, and is conservative. At the prototype 6.19 GiB/case this gives **23 cases [I]**, not 25 from mixing GB and GiB.
   Cases needed to saturate =
   N_sat(5090) × r ≈ 8–13 (VRAM is not the limit on a B200; host cores are: ≥ 1 per case).
4. Energy = B200 board power at saturation on the old ladder (653–685 W, mean 674 W) × extrapolated s per case-hour.
   The whole-run projection leaves the measured host/startup cost per case-hour unscaled; only the steady rate gets r.

Hardware inputs: NVIDIA reports 1,440 GB of memory and 64 TB/s of HBM bandwidth across eight B200 GPUs, giving
180 GB and 8 TB/s per GPU. The capacity scenario interprets GB as decimal; actual driver-reported usable bytes remain
to be checked. [NVIDIA DGX B200](https://www.nvidia.com/en-eu/data-center/dgx-b200/).
The RTX 5090 bandwidth is 1,792 GB/s. [NVIDIA RTX 5090 specifications](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/).

## Uncertainty (state it with every number)

- r was measured on different code (fp64 dycore-heavy, no nesting); today's step has more small kernels per cell
  → host/launch effects favour the lower end. The ±range above spans 2.69–4.46 (−24 %/+26 % around 3.53).
- Host CPU: a DGX-class B200 node has slower single-thread cores than the 9950X; per-process dispatch could need
  more parallel cases to saturate the GPU [U].
- Board power only (no node/PSU); the old ladder's power is for a different kernel mix [I].
