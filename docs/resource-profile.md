# System requirements & resource profile

This page documents the real, measured resources a `wrf_gpu` forecast consumes,
so you can size a machine and set expectations before the first run. The
numbers below were operator-surprising in earlier releases (undocumented), so
they are stated up front and honestly here.

All figures are measured on the reference workstation unless labelled
**(projected)**.

## Reference workstation

| Component | Spec |
|---|---|
| GPU | 1 × NVIDIA RTX 5090 (Blackwell), 32 GiB VRAM |
| CPU baseline (for speedup) | AMD Ryzen 9 9950X, 28-rank CPU-WRF v4.7.1 |
| Scratch | local NVMe (non-tmpfs), 200+ GiB free |
| OS / toolchain | Linux, CUDA 13, JAX with the CUDA plugin |

A single consumer RTX 5090 is the *design target*. The port is single-GPU by
default; multi-GPU is an optional, not-yet-throughput-validated path (see the
roadmap in the README).

## Memory (VRAM)

- **MEASURED v0.18.2:** the AC1_FIT 9/3/1 nested 1 km all-island case (d03
  520x280x45, ~145k columns, fp64, mp8 Thompson / MYNN / Noah-MP / RRTMG) now
  fits one reference RTX 5090 at **~18.1 GiB peak VRAM**. Before the fix, the
  same path OOMed near **31.8/32 GiB** and failed a **12.72 GiB** contiguous-arena
  request plus recurring **2.09 GiB** allocations.
- The realized algorithmic lever is bit-identical column tiling: RRTMG radiation
  column-tile defaults **16384→2048** plus tiled MYNN cold-start BouLac
  initialization. The peak is transient working memory, not persistent State
  (persistent State is ~2.5 GiB), so the fix reduces the radiation/cold-start
  scratch peak rather than changing default numerics.
- Retained 72 h gate peaks remain **22.9 GiB** (Switzerland d01) and **29.8 GiB**
  (Canary L2 d02, nested). d01 9 km standalone peaks **≈ 4.7 GiB**; the 1 km
  single-domain gate fits in a fresh process at **18.25 GiB**.

If you hit an out-of-memory error, the first levers are: run a smaller domain,
reduce the forecast a single segment at a time, or (experimental) try the
gated-fp32 preview — note fp32 is **not** the validated operational mode and is
currently no faster on this memory-bound workload.

## Cold JIT compile (first run)

JAX/XLA compiles the timestep program the first time it runs. This is a
**one-time, up-front cost before any forecast integration begins**:

- **Cold compile ≈ 4 min 55 s** on the reference workstation before the first
  forecast hour is integrated. The process appears to "hang" with no output
  during this window — it is compiling, not stuck.
- `wrf_gpu` enables JAX's **persistent on-disk compilation cache** automatically
  (`src/gpuwrf/runtime/jax_cache.py`). After the first run, the identical XLA
  executable is read from disk, so **subsequent runs skip the ~5 min compile**
  and start integrating almost immediately. The cache changes nothing about the
  numerics — the cached executable is bit-for-bit the same program.
- Default cache dir: a portable per-user path (`$XDG_CACHE_HOME/gpuwrf/jit`, or
  `$HOME/.cache/gpuwrf/jit`) — never `/tmp`, so it survives a reboot and works
  out of the box on a fresh clone. Override with `GPUWRF_JAX_CACHE_DIR` (or the
  standard JAX `JAX_COMPILATION_CACHE_DIR`). Disable for a clean cold-compile
  benchmark with `GPUWRF_JAX_CACHE=0`.

**Implication for short jobs:** a 1 h smoke run is dominated by the cold compile
on its first invocation. Always warm the cache once before timing or before
running a naive-agent / CI gate, or the wall clock will be misread as "slow".

## Scratch directory

The forecast pipeline writes intermediate files and history output to a scratch
location. It needs a **real (non-tmpfs) directory with room for the wrfout
history** — a few GiB for a single-domain 24 h run, more for nested runs.

- Point it at a local NVMe path. **Do not use a tmpfs / RAM disk** — large
  history files plus working files can exhaust RAM and fail mid-run.
- The standalone CLI exposes a scratch location via `--scratch-dir`
  (and/or the `GPUWRF_SCRATCH` environment variable). The exact flag name is
  being finalized with the v0.12.0 standalone CLI; see the quickstart for the
  current invocation.

## Wall-clock & energy (measured, v0.14)

**v0.14 runs at parity with 28-rank CPU-WRF.** On the final 72 h GPU-vs-CPU
field-parity gates (fp64, reference RTX 5090 workstation):

- **Switzerland d01** 72 h: GPU **~2762 s** vs CPU **2906 s** — **~1.05×**.
- **Canary L2 d02** 72 h: GPU **~8200 s** vs CPU **8713 s** — **~1.06×**.
- Steady-state deep-kernel cost is **~173 ms/step** and is ~90% of the wall
  (`proofs/perf/v014_perf_regression_triage.json`).

v0.14 is a **memory + WRF-identity release, not a performance release**:
completing the fully WRF-faithful dycore + physics raised per-step compute to
parity (the earlier, faster per-forecast-hour numbers were measured on an
**incomplete dycore** and no longer reflect the shipped code). **No multi-×
speedup is claimed.** Performance recovery — the fp64-operational-state ADR is
the flagged highest-leverage lever — is the dedicated focus of **v0.15**.

- **Energy:** at the current parity wall-clock the GPU is **roughly at energy
  parity** with 28-rank CPU-WRF (no multiple is claimed). The card draws ~267 W;
  an honest GPU-vs-CPU energy-to-solution figure is **pending the v0.15
  performance re-measurement** and is not asserted for v0.14.

**Whole-Earth-at-1 km (PROJECTED).** The whole-Earth-at-1 km memory and
rack-scale figures in the README are explicitly labelled **projected**: the
memory is exact arithmetic, but any global wall-clock estimate is **contingent on
the v0.15 performance recovery and on real multi-GPU throughput** (the
domain-decomposition path is bit-identity-proven on a CPU fake mesh only;
real multi-GPU throughput is not yet shipped). It is a "where this is going"
note, not a near-term capability or a benchmark.

## Quick sizing checklist

- [ ] GPU sized to the case: **~18.1 GiB peak** for the measured v0.18.2
      1 km-NESTED AC1_FIT path, **29.8 GiB peak** for the retained Canary L2 d02
      72 h gate; RTX 5090 / 32 GiB remains the reference recommendation.
- [ ] **Local NVMe scratch**, non-tmpfs, several GiB free.
- [ ] Expect a **~5 min cold compile** on the first run; warm the cache before
      timing or gating.
- [ ] CUDA 13 + a JAX CUDA build that sees the GPU (`python -c "import jax;
      print(jax.devices())"` should list a `cuda` device).
