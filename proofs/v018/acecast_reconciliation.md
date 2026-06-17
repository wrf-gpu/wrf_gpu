# v0.18 Apples-to-Apples vs AceCAST — Reconciliation (EXPECTATION / PROJECTED)

**Status:** EXPECTATION / PROJECTED. **No head-to-head benchmark was run.** This
note records the like-for-like reasoning behind the single README sentence that
positions `wrf_gpu` relative to hand-tuned-CUDA / OpenACC WRF GPU ports such as
**AceCAST**. It is **not** an established competitive claim and must not be read
as one.

## What AceCAST is (the comparator)

AceCAST (TempoQuest) is a **commercial Fortran→GPU port of WRF** (OpenACC /
CUDA-Fortran directive-based acceleration of the existing WRF source tree). It is
a hand-tuned, single-codebase GPU acceleration of the same physics WRF runs on
CPU. Published / vendor-stated performance for directive-based GPU WRF ports of
this class sits in the **~5–7× vs multi-core CPU** band for suitable
configurations, on data-center GPUs.

`FahrenheitResearch/wrf-gpu-port` is a second, independent open GPU-WRF effort.
Both exist; `wrf_gpu` does **not** claim to be the first or only OSS GPU WRF port
(the "first OSS WRF-GPU" claim was explicitly dropped — see
`publication/PUBLICATION-GO-NOGO-2026-06-16.md` and commit `4358fc7d`).

## Why this is NOT a measured head-to-head

- **No AceCAST license / binary was run on this workstation.** We have no AceCAST
  wall-clock on our cases, our GPU, or our precision regime.
- AceCAST is **fp32-dominant directive-accelerated Fortran**; `wrf_gpu` runs the
  acoustic dynamical core in **fp64** by design (around the pressure-gradient /
  buoyancy cancellation). These are different precision regimes, so even a raw
  wall-clock side-by-side would not be apples-to-apples without normalizing for
  precision and GPU class.
- Our own measured single-card numbers are **geometry-specific**: on tiny-nest
  (~55 k-column) cases the GPU is launch/occupancy-bound and runs at ~parity with
  same-box CPU-WRF (opt-in fused fast-mode ~1.27–1.30×, MEASURED). A directive
  port's published 5–7× figures are typically on larger single grids on
  data-center GPUs — a different operating point.

## The honest like-for-like EXPECTATION

On a like-for-like basis — **same GPU class, same precision regime, same domain
size, same physics** — there is no architectural reason a whole-state
device-resident JAX/XLA implementation should be far off a hand-tuned directive
port. The dominant cost is the same memory-bound stencil + column-physics work;
XLA fuses the elementwise + reduction structure, and the project's kernel
characterization shows the fp64 core is **device-bound / near-roofline** on the
reference GPU (`proofs/perf/v015/kernel_characterization.md`,
`proofs/v017/hostgap_fix_opus.md`).

So the EXPECTATION we are willing to state — **clearly labeled PROJECTED, not
measured** — is: **on a like-for-like configuration we expect to land in the same
ballpark as hand-tuned-CUDA/OpenACC ports like AceCAST, not orders of magnitude
behind.** We do **not** claim parity, and we do **not** claim to beat AceCAST.

## What would turn this into a measured claim

1. Obtain an AceCAST (or other directive-GPU-WRF) build and license.
2. Run the **same case** (same domain, same physics, same init) on the **same GPU
   class**, normalizing precision (either both fp32, or document the fp32↔fp64
   gap explicitly).
3. Report command-to-finish wall-clock both ways with the compile/cache state
   disclosed.

Until that is done, the README sentence stays an **EXPECTATION / PROJECTED**
positioning note and cites this file.

## Provenance

- `publication/PUBLICATION-GO-NOGO-2026-06-16.md` — drops the "first OSS WRF-GPU"
  claim (AceCAST + FahrenheitResearch exist).
- `.agent/decisions/PAPER-STRATEGIC-FRAMING.md` — AceCAST characterized as a
  commercial Fortran-OpenACC port with a ~5–7× ceiling; `wrf_gpu` positioned as a
  different artifact (open, clean-slate, single-language JAX).
- `PROJECT_PLAN.md` §14 — AceCAST referenced as a shadow/counterfactual benchmark,
  never run as a primary gate.
- Measured single-card context: `proofs/v017/hostgap_fix_opus.md`,
  `proofs/perf/v015/kernel_characterization.md`,
  `proofs/v018/perf_neutrality_FINAL.md` (v0.18 perf-neutral vs v0.17).
