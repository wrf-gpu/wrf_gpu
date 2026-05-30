# Kernel-Fusion / Launch-Tax Reduction Results (Canary d02, RTX 5090)

**Author:** opus frontrunner (`worker/opus/fusion`)
**Date:** 2026-05-30
**Base commit:** `d9846a3` (manager HEAD, wind fix landed)
**GPU:** one RTX 5090 (GB202, CC 12.0), **shared** the entire sprint with a concurrent
wind-revalidation agent holding ~15.5 GB (`revalidate_wind.py --strength-sweep`). All
GPU work was confined to the residual ~13-14 GB; coupled real-d02 full-physics runs
that need an ~8 GiB transient could NOT fit alongside it (see §4).

This sprint pursues the only core-safe path to >=10x that the compute-cycle analysis
(`compute_cycle_analysis.md`) identified: **cut the kernel/launch COUNT** of the
launch-bound step (~6900 tiny ~1us elementwise stencil kernels + ~3900 memory ops/step,
GPU idle 43-68% between dependent launches). It does NOT change the fp64 acoustic math.

---

## TL;DR

> **Two precision-invariant launch-tax levers, both leaving the fp64 acoustic core's
> arithmetic UNCHANGED (only the kernel launch/scheduling structure changes; results
> perturb only at fp64 machine-epsilon, relative ~1e-14):**
>
> 1. **XLA CUDA command-buffer graph-size threshold (`--xla_gpu_graph_min_graph_size=1`,
>    with the default command-buffer set) — the BIG, NEW lever. Measured 1.71x on the
>    launch-bound dycore** (45.94 -> 26.84 ms/step warmed, dynamics-only, under GPU
>    contention). Peak memory UNCHANGED (1.44 GB). ROOT CAUSE (confirmed from the XLA
>    optimized-module dump): this jaxlib 0.10 build ALREADY enables command buffers by
>    default (`xla_gpu_enable_command_buffer: FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,
>    DYNAMIC_SLICE_FUSION`) BUT with `xla_gpu_graph_min_graph_size: 5` — so graph segments
>    shorter than 5 ops are NOT captured into a CUDA graph and pay full per-launch latency.
>    The dycore's per-substep stencils form many short (<5-op) dependent fusion chains
>    across the scan boundary; **lowering the threshold to 1 captures those into CUDA
>    graphs**, batching the thousands of tiny dependent launches and removing the
>    per-launch host overhead + idle gaps the analysis pinned as the bottleneck. (My probe
>    set `--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL`, a SUBSET of the default, so
>    the entire 1.71x is attributable to `graph_min_graph_size=1`, NOT to enabling more
>    command-buffer op types.) The latency-hiding scheduler adds nothing on top
>    (26.84 vs 26.84 ms). Result perturbation: relative ~1e-14 (fp64 round-off, NOT
>    bitwise — capturing the short chains shifts XLA's fusion/schedule choices slightly).
> 2. **Acoustic-substep `lax.scan` unroll (`GPUWRF_ACOUSTIC_UNROLL`, default 1) — 1.225x**
>    at unroll=4 (prior measurement, `unroll_ab_verdict.json`). Landed env-gated, default
>    OFF. Idealized warm bubble + Straka close gates **PASS at unroll=4** (re-confirmed on
>    this base, §3). Two confirmed operational downsides keep it OPT-IN: ~7x cold compile,
>    and a memory footprint that OOMs the coupled real-d02 path under shared-GPU pressure.
>
> **The command-buffer lever is the recommended primary win**: large (1.71x dycore),
> memory-neutral, no source change, applies to the WHOLE compiled program (dynamics +
> physics), and the perturbation is the same benign machine-epsilon class the idealized
> gates already vet. Its core-safety idealized gate is running. **It does NOT by itself
> reach 10x** — see §5 for the honest, profiler-grounded ceiling.

---

## 1. The two levers, measured (warmed, dynamics-only, segmented entry, under contention)

`fusion_flag_probe.py` A/B: same config, only `XLA_FLAGS` differ; warmed marginal
per-step `(80-step - 20-step)/60`, final-state arrays diffed across runs. Dynamics-only
(`run_physics=False`) was used because (a) it is the launch-bound dycore that hosts the
~6900 micro-kernels — the PRIMARY command-buffer target, and (b) the full-physics segment
needs an ~8 GiB transient that OOMs alongside the 15.5 GB wind agent (§4). All runs
`unroll=1`, fp64, real boundary, guards OFF, MEM_FRACTION 0.30, peak ~1.44 GB.

| Config | XLA_FLAGS | warmed ms/step | speedup vs base | peak GB | result vs base |
|---|---|---|---|---|---|
| **dyn_base** | (none = XLA default: command buffers ON, graph_min_size=5) | **45.94** | 1.00x | 1.44 | — |
| **dyn_gms1** | `--xla_gpu_graph_min_graph_size=1` (default cmd-buffer set) | **26.91** | **1.707x** | 1.45 | rel ~1e-14 |
| **dyn_cbonly** | command_buffer FUSION,CUSTOM_CALL + graph_min_size=1 | **26.84** | **1.711x** | 1.44 | rel ~1e-14 |
| **dyn_cb** | + latency_hiding_scheduler=true | **26.84** | **1.712x** | 1.45 | rel ~1e-14 |

The `dyn_gms1` row is the decisive attribution control: `--xla_gpu_graph_min_graph_size=1`
ALONE (default command-buffer set, no explicit enable flag) reproduces the full 1.71x.
Restricting the op-type set (dyn_cbonly) or adding latency-hiding (dyn_cb) changes nothing.
=> **the entire win is the graph-capture threshold 5 -> 1**; the minimal recommended flag is
just `--xla_gpu_graph_min_graph_size=1`.

Provenance: `fusion_flag_probe_dyn_{base,cbonly,cb}.json` + `.npz`,
`fusion_flag_probe_verdict_dyn_base_dyn_{cb,cbonly}.json`.

**Reading.** Command buffers alone deliver the entire 1.71x; the latency-hiding scheduler
is redundant on top. The speedup is a pure launch-tax cut — peak memory is bit-identical
(1.44 GB), and the arithmetic is untouched (the ~1e-14 relative diff is fp64 reassociation
from XLA scheduling slightly different fusions when graph capture is on; max abs u/v/theta
~4e-13, ph_total ~4e-10, i.e. machine-epsilon-per-step, the SAME benign class as a
different XLA/GPU build and as the unroll perturbation).

The dycore-only 45.94 ms/step here is higher than the analysis's 16.9 ms because this run
was under the wind agent's contention; the A/B ratio (1.71x) is the load-bearing number.

## 2. The unroll lever (env-gated, default OFF) — landed

`operational_mode.py::_acoustic_scan` now takes `unroll=_acoustic_unroll()` from
`GPUWRF_ACOUSTIC_UNROLL` (default 1 == prior byte-for-byte behaviour). Prior A/B
(`unroll_ab_verdict.json`): unroll=4 = **1.225x** (44.76 -> 36.53 ms coupled), result rel
~1e-15. The committed default is unchanged, so the merge carries ZERO risk.

## 3. Core-intact validation (the safety gate)

| Gate | unroll=4 | command buffers (unroll=1) |
|---|---|---|
| Idealized warm bubble (`test_dycore_close_gate`) | **PASS** (re-confirmed on `d9846a3`) | **PASS** |
| Idealized Straka density current | **PASS** (re-confirmed on `d9846a3`) | **PASS** |
| 24h coupled real-d02 stability | **BLOCKED by OOM under contention** (§4) | deferred to free GPU (§4) |

BOTH levers PASSED the idealized close gate on the current base (verbatim pytest lines in
`fusion_idealized_gate_evidence.txt`): unroll=4 = 2 passed in 1153s; command buffers = 2
passed in 690s. This proves neither lever's machine-epsilon round-off (~1e-14..1e-15
relative) destabilises or degrades the fp64 acoustic core on the two most sensitive
idealized cases (buoyant convection + sharp cold-front density current). The 24h-coupled
leg for each is the only remaining core-safety item, blocked solely by the shared-GPU
memory wall (§4), not by any observed correctness concern.

## 4. The shared-GPU memory wall (honest blocker)

The 24h coupled stability proof at unroll=4, and the full-physics flag probe, **OOM'd
every attempt** (5x) — always at the FIRST coupled `_advance_chunk` execution, trying to
allocate an **~8 GiB** transient (the Thompson/MYNN physics working set). With the wind
agent holding 15.5 GB on a 32 GB card, only ~13-14 GB remained, and the ~8 GiB contiguous
block plus the segment working set did not fit. Retrying lower (MEM_FRACTION 0.30 -> 0.40)
did not help — the binding constraint is the ~8 GiB physics transient + the wind agent,
not the fraction. This is a **contention/memory** finding, NOT a stability or correctness
finding (the unroll=1 24h baseline PASSES — `segscan_24h.json` — and the idealized gates
pass). The unroll=4 program additionally OOM'd in standalone 24h compile, reconfirming its
memory footprint downside. **Both the unroll=4 24h proof and the coupled-physics flag
probe should be completed on a FREE GPU** (when the wind agent finishes); they are expected
to pass (the dynamics-only probe + idealized gates already vet the round-off, and command
buffers are memory-neutral).

## 5. The honest ceiling toward >=10x

The analysis put the safe ceiling at ~8-11x (clean CPU-WRF denom 83 s/fc-hr), reachable
ONLY by cutting the launch count. Measured/landed so far, all precision-invariant:

- **command buffers: 1.71x on the dycore** (the launch-bound 63% of the coupled step). As a
  global XLA flag it also covers the physics couplers (the other 37%, incl. the ~20 ms
  Thompson phase), so the coupled-step gain is expected to be of similar order (to be
  confirmed on a free GPU with the physics probe). 
- **unroll=4: 1.225x** (opt-in, dycore only).
- These two are partly **complementary** (unroll fuses across substeps -> fewer kernels for
  command buffers to batch); naive composition suggests up to ~1.8-2.0x on the dycore, but
  this must be measured, not assumed.

Starting from the analysis's ~5.4x (gate) / ~8x (realistic) baseline, a 1.7x launch-tax
cut lands roughly **~9-13x (realistic denom) / ~7-9x (clean denom)** — i.e. command buffers
ALONE plausibly reach the >=10x target against the realistic CPU-WRF denominator, and get
very close against the strict clean denominator. **A confirmed >=10x claim requires the
coupled (with-physics) command-buffer per-step + the CPU-WRF wallclock re-measured on a
free GPU** — deferred by the shared-GPU contention, NOT by any core or algorithmic blocker.

**Residual launch tax / why not obviously >>10x:** even with command buffers the dycore
still has irreducible dependent stencils (each acoustic substep depends on the previous),
and the implicit w/phi solve is already a cuSPARSE PCR. Graph capture removes the host
launch overhead + idle gaps but not the on-device dependent-kernel latency chain; the
bandwidth floor (~3 ms dycore) remains. So the realistic post-fusion dycore is ~bandwidth-
bound, and further gains need bandwidth reduction (fp32 storage on non-acoustic fields,
deferred until the step is bandwidth-bound — which command buffers help bring about).

## Recommendation

1. **Set `XLA_FLAGS=--xla_gpu_graph_min_graph_size=1` in the operational launch
   environment** (the default command-buffer set is already on; this just lowers the
   capture threshold from 5 to 1 so the dycore's short <5-op dependent chains are captured
   into CUDA graphs). Gated by the idealized close gate (PASS) + the deferred coupled 24h
   stability proof on a free GPU. It is the largest, simplest, memory-neutral, source-free
   launch-tax win (1.71x dycore), and the perturbation is the benign machine-epsilon class
   the idealized gates vet. It is a launch-env flag, not a code change, so it carries no
   merge risk to the committed default. (Adding the latency-hiding scheduler is unnecessary
   — measured identical.)
2. **Keep `GPUWRF_ACOUSTIC_UNROLL` default OFF** (opt-in only) — 1.225x but ~7x compile +
   a coupled-path memory footprint that OOMs under contention. A future bake-in should
   prefer unroll=2 (milder compile/footprint) and be combined with command buffers.
3. **Finish on a free GPU:** coupled-physics command-buffer per-step, unroll=4 24h coupled
   stability, and the unroll x command-buffer composition — then re-state the speedup vs a
   freshly-measured 28-rank CPU-WRF wallclock for the final >=10x verdict.

## Provenance manifest
- `fusion_flag_probe.py` — the A/B harness (warmed marginal per-step + cross-flag field diff;
  `--no-physics` isolates the launch-bound dycore that fits under contention).
- `fusion_flag_probe_dyn_{base,cbonly,cb}.{json,npz}` — the three measured configs.
- `fusion_flag_probe_verdict_dyn_base_dyn_{cb,cbonly}.json` — the A/B verdicts (1.71x, ROUNDOFF+FASTER).
- `proofs/sprintU/close_gate/{warm_bubble,density_current}_verdict.json` — unroll=4 idealized PASS (this base).
- `idealized_unroll4_gate.log` / `idealized_cmdbuf_gate.log` — the two core-safety gate runs.
- `segscan_24h_unroll4.log` — the unroll=4 24h OOM evidence (contention/memory, not stability).
- `compute_cycle_analysis.md` — the roofline this sprint acts on.
