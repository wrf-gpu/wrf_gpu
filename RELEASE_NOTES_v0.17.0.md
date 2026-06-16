# wrf_gpu v0.17.0 — the PERFORMANCE release (honest, host-side-maxed)

> **v0.17 closes the GPU's *host-orchestration* holes for live-nested forecasts
> and answers the speedup question honestly.** The default configuration stays
> **bitwise-identical to v0.16 numerics** (the WRF-fidelity identity gates
> transfer unchanged). On top, an **opt-in fast-mode** lifts the canary all-7
> nested run from CPU-parity to **~1.27–1.30× vs a 12-rank CPU**, and the work
> determines — with a profiler and an independent cross-model critic — the honest
> ceiling: **for tiny-nest geometries the GPU is launch/occupancy-bound, ≥2× is
> not single-card reachable, and fp32 cannot move it.** The GPU's real value is
> **CAPABILITY** — **MEASURED:** a 1 km single domain fits one RTX 5090
> bit-identically, the all-7 1 km nested case runs end-to-end on one card, and the
> ~1.27–1.30× opt-in fast-mode (vs the 12-rank CPU) is real. **PROJECTED /
> UNMEASURED:** large-grid + **cluster / multi-GPU weak-scaling** and
> whole-Earth-at-1 km throughput. NOT single-card speedup on tiny nests.

All speedups are **MEASURED** on the reference RTX 5090 workstation against the
**same-box 12-rank CPU-WRF** baseline; capability statements are labelled
**PROJECTED** where not benchmarked.

---

## 1. Shipped DEFAULT-ON — bit-identical (the identity gates still hold)

These fix real wallclock defects in the live-nested path **without changing any
dispatched op or HLO**, so the default-config output is byte-for-byte the v0.16
forecast (the WRF-fidelity identity proofs transfer unchanged):

- **Nested compile-CHURN fix.** The all-7 island nest (`--max-dom 9`) previously
  **never reached a warm forecast on a cold cache** — it recompiled the per-domain
  timestep program ~2× per domain (≈18–20 modules, one every ~4–5 min, GPU ~99 %
  idle, **0 forecast output after >65 min**). Cause: the nested host loop seeded
  each domain with a *host/uncommitted* carry while the compiled chunk *returns
  device-committed* leaves, so the first and second advance keyed on different
  shardings → two compiles per domain. Fix: commit the seed carry once
  (`jax.device_put`; values bit-identical). **Result: ~9 compiles, the all-7 now
  forecasts at all.**
- **GPU-idle relaxation (`block_between` → root-async sync).** The production loop
  blocked the host after **every** domain advance (~5,000 `block_until_ready`/
  forecast-hour), draining the GPU queue. It now syncs once per root step
  (`GPUWRF_NESTED_SYNC_MODE=root`, default) — JAX device dataflow preserves the
  WRF parent→child ordering without the host block. A CPU test proves the
  advance/force/output schedule is byte-identical across sync modes.
- **Edge-only (ring-only) boundary interpolation** (`GPUWRF_EDGE_ONLY_BOUNDARY`,
  **default ON**). The child-boundary forcing interpolated the full child grid
  then sliced the width-5 ring; it now gathers only the ring (same per-cell
  bilinear formula on a weight subset). **Bit-identical** (23 CPU gates,
  `np.array_equal` on every boundary leaf, including under `jax.jit`).

→ **Default config = v0.16 numerics, bitwise.** No new schemes, no precision
change. (v0.18 work is intentionally **not** in this release.)

## 2. OPT-IN fast-mode — `GPUWRF_NESTED_FUSE=1` (tolerance-pass, NOT bitwise)

A **fused d02-substep cascade** compiles one parent-substep + its seven child
{force, advance} into a single GPU program, cutting host dispatch from ~47 to ~5
per root step. **MEASURED on the canary all-7 (9/3/1 km, fp64):**

| config | warm s/forecast-hr | vs 12-rank CPU (893 s) | GPU util | peak VRAM |
|---|---:|---:|---:|---:|
| default (eager, root sync) | 1005 | 0.89× | 56 % | 17.4 GB |
| `GPUWRF_NESTED_FUSE=1` (fused) | 702 | **1.27×** | **96 %** | 19.1 GB |
| fused + edge-only (default-on) | 689 | **1.30×** | 96 % | 19.2 GB |

**Caveats (read before enabling):**
- **NOT bit-identical to the default path.** Fusing lets XLA FMA-contract the
  boundary interpolation; in a *chaotic* forecast that last-bit difference grows
  (perturbation pressure P diverged ~1.3 → ~20 over 2 h) — **the forecast stays
  physical and PASSES the tolerance gate vs CPU-WRF**, but it is a different valid
  trajectory, not the bitwise-default one. **fp32 would diverge the same way,
  faster.**
- **~38 min one-time fused compile** (the fused HLO is large), **cached**
  thereafter — fine for multi-hour / repeated release runs, heavy on a fresh
  machine.
- **Stability is the operator's gate for their case:** treat it as an optional
  fast-mode and confirm it holds tolerance over *your* forecast length (e.g. 24 h
  for a 1 km nowcast) against WRF v4 before relying on it.

Related opt-in knobs: `GPUWRF_NESTED_SYNC_MODE` (`root` default / `advance`
legacy / `segment`), `GPUWRF_JIT_BOUNDARY` (jit the eager boundary builder,
default off), `GPUWRF_HOST_LEDGER` (per-phase host-time diagnostic).

## 3. The honest WHY (the speedup question, answered)

After the host holes are closed the all-7 is **GPU-COMPUTE-bound at ~674 s/
forecast-hour** (the host ledger shows `sync ≈ 0` — the GPU is never waiting on
the host). An **nsys** trace shows the top kernels are generic element-wise XLA
fusions with **no hot-spot** (top kernel 11 %) and **every kernel averages ~1.5 µs
across thousands of launches** — the signature of **many tiny kernels on
under-filled 1 km nests** (d06/d07 are 40×40), i.e. a **launch/occupancy limit**,
not a throughput limit. Consequences, independently confirmed by a GPT-5.5 critic:

- **fp32 cannot move it.** There is no large saturating kernel whose bytes/FLOPs
  fp32 would halve; broad fp32 is ~1.1× and corrupts the geopotential/PGF
  cancellation (the acoustic core is *deliberately* fp64). **fp32-physics islands
  project ~1.5–1.6× — still < 2× — and are deferred to v0.18 as an optional
  24h-stability fast-mode**, not shipped here.
- **≥2× and 3× are NOT single-card reachable for this tiny-nest geometry.** 3×
  needs ≤298 s/hr on 55 k tiny columns where a 12-rank CPU is genuinely
  competitive. The all-7 is **near worst-case** for single-card GPU speedup.

## 4. HEADLINE — capability, not single-card tiny-nest speedup

**MEASURED** value is **CAPABILITY**: a **1 km single-domain fp64 forecast fits on
one RTX 5090, bit-identically** (v0.16 chunked-BouLac unlock), the **all-7 1 km
island product now runs end-to-end on one card**, and the **~1.27–1.30× opt-in
fast-mode** vs the 12-rank CPU is real. **PROJECTED / UNMEASURED:** large single
grids + **cluster / multi-GPU weak-scaling** (where the *throughput* would live)
and the whole-Earth-at-1 km "fits one rack" figure (exact memory arithmetic; real
multi-GPU throughput is **not benchmarked**). Do not read cluster throughput as
measured. **No single-card multi-× speedup on tiny nests is claimed.**

---

## Validation & evidence

- Identity (default config): unchanged from v0.16 — the bit-identical default-on
  fixes touch only host-side device-placement / `block_until_ready`, proven by a
  CPU schedule-equivalence test + warm-cache re-run.
- Fast-mode A/B, host ledger, nsys grounding, fp32 verdict, and the full
  root-cause write-up: **`proofs/v017/hostgap_fix_opus.md`**.
- Release assembly + ready-to-tag checklist: `proofs/v017/v017_release_assembly.md`.
