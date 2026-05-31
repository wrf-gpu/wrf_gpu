# Optimization Refutations — measured, not assumed

Five candidate per-step optimizations were each **implemented and measured**, and
each was refuted (or rejected on fidelity) by direct measurement. This is the core
"optimization homework": the ~5.3×/7.8× speedup is near-optimal *under the
WRF-fidelity constraint*, and the constraint does real work in every case.

Full narrative: `publish/runtime_optimization_analysis.md` §4.

| Lever | hypothesis | measured effect | fidelity verdict | decision | proof path |
|---|---|---|---|---|---|
| **fp32 dynamics** | fp64 = 1/64 of fp32 on Blackwell → ~2× | **~1.00× (0× gain)** | n/a (no speed) | rejected as speed lever | `proofs/perf/compute_cycle_analysis.md` |
| **CUDA command-buffer graph capture** | launch tax (~6 900 tiny kernels, 43–68 % idle) is what graph capture removes | dynamics-only **+1.71×**; but **operational coupled step REGRESSES 15–21 %** (0.826× / 0.866×, two A/B runs) | bitwise-identical in all 8 fields | **do NOT bake** the flag — slows the operational forecast | `proofs/perf/fusion_results.md`, `proofs/perf/fusion_confirm_results.md` |
| **fp32 Thompson microphysics** | Thompson is the one large compute phase (~half step) → fp32 lever | **~1.0×** (full kernel 11.0→11.0 ms; tiled 42.35 vs 33.83 fp64 = **0.80× regression** without unroll) | oracle-faithful (≤ ~1 fp32 ULP, rel ≤ 9e-7) | kept as gated opt-in `GPUWRF_THOMPSON_FP32`, **not default** (no speed) — Thompson is ~85 % sedimentation = launch/bandwidth-bound | `proofs/thompson_perf/kernel_lever_summary.json`, `proofs/thompson_perf/THOMPSON_PERF_ANALYSIS.md` |
| **Implicit (backward-Euler) sedimentation** | replace 64-substep explicit upwind with one unconditionally-stable implicit sweep | kernel **+2.25–2.44×** (33.8→13.9 ms) | **FAILS fidelity** — over-precipitates +47 % vs precipitating WRF Thompson oracle (nsub=1); nsub=4 still +26 % vs oracle (worse than explicit default's +13 %) at only 1.61× | **REJECTED as default**; gated experimental knob `GPUWRF_THOMPSON_IMPLICIT_SED` (OFF). GPT-5.5 xhigh concurred | `proofs/thompson_perf/implicit_sed_timing.json`, `proofs/thompson_perf/PRECIP_ORACLE_AND_IMPLICIT_SED.md` |
| **Thompson sed-scan unroll** (shipped safe win) | inline sedimentation `lax.scan` iters → pure launch-count cut | kernel **~1.08×** (33.8→31.2 ms); coupled step **~1.05×** (45.53→43.56 ms/step) | **bit-identical** to base (17/17 Thompson tests pass; oracle byte-for-byte) | **SHIPPED** (`GPUWRF_THOMPSON_SED_UNROLL=2`) — moves headline 5.06×/7.5× → 5.29×/7.84× | `proofs/thompson_perf/kernel_lever_summary.json`, `proofs/thompson_perf/coupled_timing_base_vs_opt.json` |
| **Acoustic-substep unroll** (opt-in win) | inline acoustic `lax.scan` → launch-count cut on dycore | dycore **1.225×** (44.76→36.53 ms/step) at unroll=4 | **NOT bitwise-identical** (cross-substep fp64 reassociation, rel ~1e-15: max abs diff u 6.5e-13, theta 9.1e-13, p 2.5e-10); passes both idealized close gates | shipped **env-gated, default OFF** (~7× cold-compile penalty; OOMs coupled path under shared-GPU pressure) | `proofs/perf/unroll_ab_verdict.json`, `proofs/perf/compute_cycle_analysis.md` |

## Why fp32 fails everywhere (the unifying finding)

Both fp32 levers give ~1.0× for the **same reason**: the model is launch/bandwidth-
bound, not fp64-compute-bound. fp32 only accelerates arithmetic-throughput-bound
kernels, of which this model has none (dycore AI 0.40 is 146× below the fp32
ridge; fp64 arithmetic is ~8 % of the step). fp32 *would* halve bytes on the
bandwidth-bound step, but the mandatory-fp64 acoustic island (EOS, calc_coef_w,
momentum updates, implicit w/φ solve, geopotential — fp64 for stability) forces
`convert(f32↔f64)` on entry/exit every RK stage/substep, adding HBM traffic and
kernels that cancel the saving while leaving the ~6 900-kernel launch count
unchanged. Source: `proofs/perf/compute_cycle_analysis.md`.

## The implicit-sedimentation refutation in numbers

Validated against a purpose-built **precipitating** WRF Thompson oracle (real
Fortran `mp_gt_driver`, rain/graupel/snow/ice all sedimenting — the pre-existing
dry/clear savepoint could not discriminate a sedimentation change):

| Scheme | surface precip vs WRF 0.347 mm | qr mean-rel vs WRF | kernel speedup |
|---|---:|---:|---:|
| faithful explicit (shipped default) | 0.393 mm (+13 %) | 1.53 % | 1.00× |
| implicit BE nsub=1 | 0.510 mm (+47 %) | 5.11 % | 2.25× |
| implicit BE nsub=2 | 0.466 mm (+34 %) | 3.68 % | 2.03× |
| implicit BE nsub=4 | 0.436 mm (+26 %) | 2.76 % | 1.61× |

No nsub is both ≥2× faster AND as faithful as the explicit default.
Source: `proofs/thompson_perf/PRECIP_ORACLE_AND_IMPLICIT_SED.md`.
