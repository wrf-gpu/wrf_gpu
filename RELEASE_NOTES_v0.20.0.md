# v0.20.0 — correctness + stability + capability + reliability (modest measured nest speedup)

A **bit-identical-safe** release. The fp64 default path is byte-for-byte unchanged
from v0.19, and v0.20 adds a modest *measured* nest speedup, an opt-in fp32
capability mode, and a compile cache that just works across runs and across
forecast dates. v0.20.0 branches off **0.19.1** (0.19.2 was never shipped). No
breaking interface, namelist, or default-numerics change.

## What changed

### Faster all-7 nested fast path than v0.19 and CPU — byte-identically

The default fused all-7-island, 9-domain run measures **~668 s/forecast-hour warm**
(range 645–680) on the reference GPU against **713 s/forecast-hour for v0.19** and
the canonical **12-rank CPU-WRF baseline at 1020 s/forecast-hour** — **~1.07×
faster than v0.19** and **~1.53× faster than CPU**. The output is **byte-identical
to v0.19 (1926/1926 vars, maxΔ = 0.000e+00)**: the gain comes entirely from a
**numerics-free CUDA stream-ordered allocator** (`cuda_async`, now the default),
which carries essentially the whole improvement (~+44 s/forecast-hour); the
async-output and host-RAM event-tail-guard levers are stability/neutral by design.
Peak VRAM is **122 MiB leaner** than v0.19 and stays flat over 12 output groups
(1 km fit, no OOM). The nest is host-bound (~7–8% GPU duty) and the headline is a
**range, not a point** (measured under live CPU contention, ±2–3% noise); the
identity / fit / VRAM gates are robust regardless. Larger structural multipliers
(acoustic-substep config, fused megakernel, fp32-relaxed) are an explicit **future
wave, not in v0.20**.

*MEASURED: [`proofs/v020/lowhang/COMBINED_SPEEDUP.md`](proofs/v020/lowhang/COMBINED_SPEEDUP.md) §4–§8.*

### fp64 default stays byte-identical and same-speed

The `fp64_default` GPU all-7 9-domain output is **963/963 vars maxΔ = 0.000e+00,
byte-identical** across all 9 domain files; warm fp64 forecast-only time is
**HEAD ≈ baseline** (within noise). v0.20 adds no numerics risk on the default
path.

*MEASURED: [`proofs/v020/fp32_integration/FP32_INTEGRATION_REPORT.md`](proofs/v020/fp32_integration/FP32_INTEGRATION_REPORT.md) §4.1/§4.1b.*

### Opt-in fp32 mixed-precision mode for capability + VRAM (not speed)

`GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020` runs a
perturbation-authoritative fp32 acoustic path (fp64 totals). It cuts whole-run
VRAM **−14.4%** (aggressive) and extends full-physics cell capability **~1.16×**
(fits 700² where fp64 caps 650²; in a dynamics-only stress fp64 OOMs at 1M columns
where fp32 fits). On the single RTX 5090 it is **NOT a speedup** — the fp32/fp64
throughput-ceiling ratio is **≈0.91 (≈1, not ≈2)**, and there is no peak-VRAM win
on small single domains (radiation-transient-bounded below ~384²). **Honest
scope:** fp32 tolerance is checked at the **1 h lead** (19/19 fields green) — real
but **not stringent**; the **24–120 h skill gate is future work, out of v0.20
scope**. fp32 is strictly opt-in; `fp64_default` remains the default.

*MEASURED VRAM/capability + 1 h tolerance: [`FP32_INTEGRATION_REPORT.md`](proofs/v020/fp32_integration/FP32_INTEGRATION_REPORT.md) §4.2. INCONCLUSIVE single-card speed: [`proofs/v020/benchmark/T2T3_REPORT.md`](proofs/v020/benchmark/T2T3_REPORT.md) R∞ ratio ≈0.91.*

### Compile cache hits across forecast dates, zero config

The persistent per-user on-disk JIT cache (default on since v0.12.0) previously
baked the forecast date into the radiation/solar HLO, so every new date missed the
cache and recompiled. The date is now a **runtime argument** (`clock_base`), so the
**lowered HLO is identical across dates** (sha256 identical across 3 dates,
including a leap year) and re-running on a new or leap-year date is a **warm cache
hit with 0 new cache entries**. The **default RRTMG path stays bit-identical**
(traced vs baked clock: **64/64 State leaves byte-identical**, 3 steps, RRTMG
SW+LW). One documented non-default residual: GSFC SW (`ra_sw=2`) keeps a seasonal
ozone-band index, so its HLO still date-varies.

*MEASURED: [`proofs/v020/julday_cache/JULDAY_CACHE_FIX_REPORT.md`](proofs/v020/julday_cache/JULDAY_CACHE_FIX_REPORT.md).*

### Single-domain scaling re-certified honestly

A parametrized single-domain scaling study (Swiss base, tiled, dt = 10 s) shows
throughput saturating at a ceiling of **~9.6e6 cells/s (fp32) / ~1.06e7 cells/s
(fp64)** by ~384²; fp32 **fits 512² (11.5 M cells) at 16.2 GB where fp64 OOMs**
(capability), with **no single-domain speed win** from fp32 (t-ratio 0.86–1.07).
On a tiny single-domain 129² grid the GPU is **~2.3× slower** than 24-rank CPU-WRF
(host/launch-bound); GPU-vs-CPU identity at 1 h is tight (T2 RMSE 0.79 K, U10
0.56 m/s, PSFC corr 0.999996). The harness is parametrized to lift to H200/GB300
(those results are **PROJECTED**).

*MEASURED: [`proofs/v020/benchmark/T2T3_REPORT.md`](proofs/v020/benchmark/T2T3_REPORT.md) G-series + Swiss-CPU-match + identity.*

### All-7 24 h GPU-vs-CPU identity (fp64 vs CPU-WRF, 9 domains)

The dynamics/thermo **core is EXCELLENT** — **T corr 0.9999 (RMSE 0.69 K), PH and
PSFC corr 0.9997, U 0.991, V 0.968, QVAPOR 0.964** (mean correlation across the 9
domains). The **surface diagnostics are looser** (the most
parameterization-sensitive fields): **T2 0.944 (RMSE 0.78 K), TH2 0.878, U10 0.855,
V10 0.852** (mean corr); on the **inner 1 km Alpine nests d06/d07 the 10 m winds
spread to corr ~0.48–0.65 / RMSE ~4–5 m/s** — honestly the expected most-sensitive
field on complex terrain, **byte-identical to the validated v0.19 output (this is
NOT a v0.20 regression)**, with divergence growing with lead time. Logged as a
v0.20.1 characterization item.

*MEASURED: [`proofs/v020/validation/identity/identity_metrics.json`](proofs/v020/validation/identity/identity_metrics.json) — 24 h, init 2026-02-14 18Z, fp64 GPU vs CPU-WRF, 9 domains, 10 fields.*

## Opt-outs (explicit)

| Switch | Effect |
| --- | --- |
| `GPUWRF_BITWISE=1` or `GPUWRF_NESTED_FUSE=0` | Eager non-fused bitwise/debug path |
| `GPUWRF_ALLOCATOR=platform` | Restores the pre-v0.20 synchronous allocator |
| `GPUWRF_ACOUSTIC_PRECISION_MODE=fp64_default` (default) | Byte-identical fp64 precision mode |

## Compatibility

No breaking interface, namelist, or default-numerics change from v0.19. The
`fp64_default` path is byte-identical to v0.19. fp32 mixed-precision and the
non-default allocator are strictly opt-in.
