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

- **Peak VRAM during integration ≈ 24.6 GiB** on the validated 3 km Canary d02
  (159 × 66 × 44) forecast at fp64. This includes JAX/XLA working buffers, which
  are several times the bare prognostic-state footprint.
- A 32 GiB card (RTX 5090) runs d02 comfortably. Cards with less than ~26 GiB of
  free VRAM are likely to OOM on this domain at fp64.
- Nested d01→d02→d03 runs hold only one domain's transient scratch live at a
  time, but the peak is still set by the largest domain; budget for the 32 GiB
  card.

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
- Default cache dir: `/mnt/data/gpuwrf_jax_cache`. Override with
  `GPUWRF_JAX_CACHE_DIR` (or the standard JAX `JAX_COMPILATION_CACHE_DIR`).
  Disable for a clean cold-compile benchmark with `GPUWRF_JAX_CACHE=0`.

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

## Wall-clock & energy (measured, warmed)

On the 3 km Canary d02 fixture, warmed (compile excluded), fp64:

- **≈ 15.35 s per forecast-hour** on one RTX 5090 (≈ 42.6 ms/step at dt=10 s).
- **≈ 2.47× warm real-user wall-clock speedup** vs 28-rank CPU-WRF on the same
  workstation for the d02 single-domain path. The honest apples-to-apples band
  is **5–8× per-step**, with a strict dt-parity floor of **~3.2×**; the kernel /
  compute-only ceiling (~5.3×–7.84×) is a per-step number and is **not** the
  real-user wall-clock headline. Provenance: `proofs/perf/speedup_denominator.md`.
- **Energy:** GPU 267 W × 15.4 s/forecast-hour ≈ 4.1 kJ per forecast-hour vs CPU
  ~200 W × 83 s ≈ 16.6 kJ — **~4× less energy** to the same 24 h forecast.

**(Projected)** On large, GPU-saturating grids the energy-to-solution advantage
is projected to widen to 4–8×. The whole-Earth-at-1 km memory and rack-scale
figures in the README are explicitly labelled **projected** (memory is exact
arithmetic; wall-clock is a roofline projection of the not-yet-implemented
multi-GPU domain-decomposition path).

## Quick sizing checklist

- [ ] GPU with **≥ 26 GiB free VRAM** (RTX 5090 / 32 GiB recommended) for d02 fp64.
- [ ] **Local NVMe scratch**, non-tmpfs, several GiB free.
- [ ] Expect a **~5 min cold compile** on the first run; warm the cache before
      timing or gating.
- [ ] CUDA 13 + a JAX CUDA build that sees the GPU (`python -c "import jax;
      print(jax.devices())"` should list a `cuda` device).
