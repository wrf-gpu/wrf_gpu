# Thompson Microphysics Performance Analysis (Canary d02, RTX 5090)

**Author:** opus frontrunner (`worker/opus/thompson-perf`)
**Base commit:** `4e3a9ff` (manager HEAD)
**Date:** 2026-05-31
**Mission:** halve the Thompson phase (the coupled step's #1 phase, ~half the step) toward >=10x vs 28-rank CPU-WRF, without breaking microphysics correctness or the core.

---

## TL;DR

> **fp32 microphysics is a DEAD END for speed on this kernel (~1.0x).** Direct
> measurement refutes the premise that Thompson is the fp64-compute-bound fp32
> lever: the Thompson kernel is dominated (~85 %) by the **sedimentation substep
> loop**, which is **launch/bandwidth-bound** (64 sequential tiny dependent
> upwind passes x 4 species), NOT fp64-arithmetic-bound. Even when forced to run
> the entire rate/integration math in fp32, the kernel barely moves
> (11.0 -> 11.0 ms; the tiled 20748-col grid: 42.3 vs 42.9 ms). This is the SAME
> launch-bound finding as the dycore — fp32 only helps arithmetic-throughput-bound
> kernels and this kernel has none.
>
> **Shipped (safe, default ON, BIT-IDENTICAL): sedimentation scan-unroll = 2**,
> a pure launch-count reduction (~1.1x on the kernel: 33.8 -> 31.2 ms on the
> tiled grid). It changes nothing numerically (the unrolled scan inlines
> iterations in order). The full Thompson kernel and the WRF microphysics oracle
> are byte-for-byte unchanged from base.
>
> **Kept gated (default OFF): fp32 work-dtype** (`GPUWRF_THOMPSON_FP32=1`). It is
> oracle-faithful (perturbs moist outputs by <= ~1 fp32 ULP, rel <= 9e-7, at/below
> the WRF oracle's own fp32 storage granularity) but gives ~1.0x, so it is not a
> default.
>
> **The real >1.5x lever is an ALGORITHM CHANGE — implicit (backward-Euler)
> sedimentation — measured ~2.4x on the kernel (33.8 -> 13.9 ms).** It is NOT
> shipped: it is more numerically diffusive (smears the falling profile O(1)
> relative) although it conserves precip + column mass to ~1 %. Adopting it needs
> a **precipitating** WRF Thompson oracle (the current oracle savepoint is a
> dry/clear column — all fall speeds 0 — so it cannot discriminate this change)
> plus a 6-24h coupled precip/T2/U10/V10 skill comparison. GPT-5.5 xhigh
> independently concurred this is a candidate scheme change, not a faithful
> drop-in. **This is the honest path to halving Thompson — but it is a
> manager-gated / ADR scheme change, not a lane-local default flip.**

---

## 1. Where the Thompson time actually goes (the decomposition that re-shaped the plan)

Per-phase isolation on the real 5187-column d02 grid (`proofs/thompson_perf/kernel_lever_summary.json`):

| Sub-phase | fp64 ms | fp32 ms | share |
|---|---|---|---|
| source/sink process rates (warm-rain, ice, sat-adj, evap, melt/freeze) | 2.48 | 2.00 | ~15 % |
| **sedimentation (64 substeps x 4 species)** | **9.99** | **9.00** | **~85 %** |
| full kernel | 11.0 | 11.0 | 100 % |

The sedimentation substep loop is the kernel. It is `NSED_SUBSTEPS=64` sequential
`lax.scan` iterations per species, each a tiny upwind flux divergence
(2 multiplies, 2 concatenates, 2 divides, 2 adds, 2 maximums) — the textbook
launch-bound pattern. fp32 shaves ~10 % off it (bandwidth), not the expected ~2x.

**The premise (compute_cycle_analysis: "Thompson is the one compute-bound phase")
is refuted by direct measurement.** Thompson is launch/bandwidth-bound like the
rest of the model.

## 2. Levers measured (full kernel, 20748-col tiled d02 grid, median of 120 reps)

| Lever | median ms | vs base | shipped? | correctness |
|---|---|---|---|---|
| base (per-species, 64 substeps, fp64) | 33.8 | 1.00x | — | — |
| **sed scan unroll=2 (DEFAULT)** | **31.2** | **1.08x** | **YES (default)** | **BIT-IDENTICAL** |
| sed scan unroll=4 | 32.1 | 1.05x | knob | bit-identical |
| fp32 work-dtype (unroll=1) | 42.3 | 0.80x | — | <=1 fp32 ULP |
| fp32 work-dtype + unroll=4 | 34.1 | 0.99x | gated opt-in | <=1 fp32 ULP |
| 4-species batched scan (unroll=1) | 42.9 | 0.79x | REJECTED | bit-identical but SLOWER |
| **implicit backward-Euler sedimentation** | **13.9** | **2.44x** | **NO (scheme change)** | profile O(1) diff; precip/mass ~1% |

Notes:
- **Batching the 4 species into one scan is a REGRESSION** — XLA already overlaps
  the 4 independent per-species scans, and stacking serialises them. Rejected.
- **fp32 + no unroll is a regression (0.80x)** because the fp32 casts add
  elementwise passes to a launch-bound kernel without removing launches — exactly
  the dycore fp32 finding. fp32 only reaches parity once combined with unroll.

## 3. fp32 is oracle-faithful (it just doesn't help)

fp32-vs-fp64 raw output diff on a moist analytic column (the WRF Fortran-harness
fixture; the real oracle savepoint is dry, see below):

| field | max abs | max rel | interpretation |
|---|---|---|---|
| qv | 2.2e-10 | 5.0e-8 | < 1 fp32 ULP |
| qc | 5.6e-11 | 8.9e-7 | ~7 fp32 ULP |
| qr/qi/qs/qg | <=2.7e-11 | <=7.8e-7 | <= ~7 fp32 ULP |
| Ni/Nr | <=0.05 | <=6.7e-7 | < 1 fp32 ULP |
| T | 1.5e-7 | 5.5e-10 | essentially exact |

These are at/below the WRF oracle's own fp32 storage granularity (WRF stores
qX/Ni/Nr/theta in fp32), so fp32 microphysics is as faithful to WRF as fp64.
The WRF microphysics oracle (`run_oracle_parity_f64`) is **byte-identical** in
fp64-default, fp32, and base modes (8 mass/number fields exact; `th` within
fp32 storage = same `all_fields_within_float32_storage_precision=True` as base).

## 4. The honest ceiling for the coupled step

Sedimentation is a genuine accuracy/cost knob in BOTH schemes:
- Explicit sub-stepped upwind is **first-order convergent** — error halves as
  NSED doubles; NSED=64 still has ~2.6 % rel error in qr vs converged NSED=128.
  So reducing NSED trades real microphysical accuracy for speed (rejected as a
  default; gated behind `GPUWRF_THOMPSON_NSED`).
- Implicit backward-Euler does it in one unconditionally-stable sweep (~2.4x) but
  is more diffusive (smears the vertical profile) while conserving the integrated
  precip + column mass to ~1 %.

**For strict WRF-faithful explicit sedimentation, the ceiling is ~1.1-1.3x**
(scan unroll + minor fusion), which is what is shipped. **>=10x coupled is NOT
reachable by touching Thompson within the faithful-explicit constraint** — the
~1.1x Thompson win is a small fraction of the coupled step. The path to halving
Thompson exists (implicit sedimentation, ~2.4x on the kernel) but it is a
**scheme change** gated on a precipitating oracle + coupled skill, which is the
recommended next sprint (manager/ADR decision).

## 5. Files / provenance

- `src/gpuwrf/physics/thompson_column.py` — shipped: sed scan `unroll=_sed_unroll()`
  (default 2, bit-identical); gated fp32 work-dtype (`_work_dtype`, default fp64);
  `GPUWRF_THOMPSON_NSED` knob (default 64).
- `src/gpuwrf/physics/thompson_column_debug_stripped.py` — mirror updated to carry
  the same cast wrapper so the HLO-identity debug test holds in both modes.
- `proofs/thompson_perf/kernel_lever_summary.json` — all lever timings + verdicts.
- `proofs/thompson_perf/thompson_timing_{base,fp32}.json` — kernel warmed timing harness.
- `proofs/thompson_perf/oracle_{default,fp32}.json` — WRF microphysics oracle parity (both pass = base).
- `proofs/thompson_perf/fp32_vs_fp64_oracle_diff.json` — fp32 perturbation (dry oracle; moist diff in this doc §3).
- `proofs/thompson_perf/implicit_sedimentation_prototype.py` — the ~2.4x scheme-change prototype (NOT wired in).
- `proofs/thompson_perf/gpt55_sedimentation_review.md` — GPT-5.5 xhigh independent review.
