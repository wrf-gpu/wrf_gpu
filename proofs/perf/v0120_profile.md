# v0.12.0 standalone-path profiling — ranked optimization opportunities

**Author:** opus perf engineer (`worker/opus/v0120-perf`), 2026-06-07
**Scope:** lightweight profiling of the v0.12.0 standalone CLI step (`gpuwrf.cli run`,
`run_forecast_operational_segmented`, fp64, dt=10 s, RTX 5090). This is a *ranked
opportunity list* for future optimization, grounded in (a) the warmed-step Nsight
Systems profile already in this repo, (b) the XLA flag A/B findings, (c) a live HLO
op histogram from the standalone segment, and (d) the host/device transfer audit.
It is **not** a validation gate.

Provenance of the profiler artifacts re-used here (all tracked in `proofs/perf/`):
`nsys_warmed_step_stats_cuda_gpu_kern_sum.csv`, `nsys_warmed_step_stats_cuda_api_sum.csv`,
`nsys_warmed_step_stats_cuda_gpu_sum.csv` (warmed coupled step, RTX 5090 — the
**authoritative** op/kernel breakdown used below), `fusion_results.md` + `fusion_flag_probe_*`
(XLA flag A/B), and `fusion_transfer_audit.json` (host/device transfer audit). The
operational standalone entry is a *host-loop* driver (`run_forecast_operational_segmented`
wraps a jitted inner `_advance_chunk`), so a single-program `jax.jit(...).lower()` of the
whole forecast is not meaningful; the warmed-step kernel histogram from Nsight Systems is
the correct artifact and is what is ranked here.

---

## Step cost breakdown (warmed, from the Nsight Systems profile)

GPU-kernel time, top consumers (`nsys_warmed_step_stats_cuda_gpu_sum.csv`):

| % GPU time | Operation | Class | Notes |
|---|---|---|---|
| 26.2 | `RedzoneAllocatorKernelImpl` | XLA autotuning | redzone buffer-overrun check kernel — *autotuning/diagnostic overhead*, not model compute |
| 13.8 | `[CUDA memcpy Device-to-Device]` (129 641 ops) | data movement | intra-device copies between fused regions |
| 11.9 | `loop_add_fusion_9` (116 136 inst) | elementwise stencil | launch-bound |
| 8.2 | `pcrGtsvBatchSharedMemKernelLoop<double>` | tridiagonal solve (fp64) | the implicit vertical w/phi (cuSPARSE PCR) — algorithmic |
| 7.7 | `[CUDA memcpy Host-to-Device]` | boundary forcing | LBC updates (see transfer audit) |
| 6.4 / 5.8 | `loop_multiply_fusion_1` / `loop_subtract_fusion_1` | elementwise | launch-bound |

CUDA-API time, top consumers (`nsys_warmed_step_stats_cuda_api_sum.csv`):

| % API time | Call | Notes |
|---|---|---|
| 38.1 | `cuLaunchKernelEx` (250 757 calls) | **the launch tax** — many tiny dependent kernels |
| 20.5 | `cuMemcpyDtoDAsync_v2` (129 641) | device-to-device copies |
| 12.2 | `cuMemcpyDtoHAsync_v2` | device-to-host (largely radiation-cadence / output boundaries) |
| 5.5 | `cuGraphLaunch` | **only 5.5%** — most kernels are NOT yet CUDA-graph-captured |

**One-line diagnosis:** the warmed step is **launch-bound** (38% of API time in
`cuLaunchKernelEx`, GPU only ~5–8% utilized between dependent launches), with a large
chunk of nominal GPU time spent in XLA's **redzone autotuning** kernel rather than model
arithmetic. The fp64 vertical tridiagonal solve is the single largest *real-compute*
kernel.

## Host/device transfer audit (timestep loop is clean)

`fusion_transfer_audit.json` (0.5 h warmed coupled scan): post-init H2D = 3.69 MB,
D2H = 3.69 MB, classified to the boundary band (not per-substep). **No per-step host↔device
transfer inside the integration loop** — consistent with the project rule. The transfers
that exist are the lateral-boundary feed and the per-hour wrfout pull, both expected and
small. No transfer-elimination opportunity in the hot loop.

---

## Ranked opportunities

### 1. CUDA-graph capture of short fusion chains — `--xla_gpu_graph_min_graph_size=1`  *(MEASURED on the coupled path — REGRESSION, NOT landed)*
- **Lever:** lower XLA's graph-capture threshold from the default 5 to 1, so short (<5-op)
  dependent fusion chains are captured into CUDA graphs, batching the ~250 k tiny launches.
- **Prior evidence (dynamics-only):** 1.71× (45.94 → 26.84 ms/step), `fusion_results.md`.
- **THIS SPRINT — coupled full-physics standalone d01 A/B (the leg the prior sprint deferred):**
  **baseline 16.71 vs gms1 19.81 s/forecast-hour = 0.84× (~19 % SLOWER)**, final state
  **bit-identical** (Δmax = Δmean = 0 on u/v/θ/w/φ/μ/qv), both `all_finite`. Cold compile also
  rose 29 s → 137 s (the flag changes the cache key + captures more segments). Provenance:
  `proofs/perf/flag_baseline.json` vs `flag_gms1.json`.
- **Verdict:** the dynamics-only win does NOT carry to the coupled step — the physics
  couplers don't benefit and the extra capture overhead dominates. **Do NOT set this flag.**
  The launch-tax reduction must instead target the *coupled* graph (coupled-aware command-buffer
  config / per-coupler fusion), which is future work, not a quick safe win. This is a
  clean falsification of the carry-over assumption, exactly what the coupled A/B was for.

### 2. Reduce XLA autotuning / redzone-check overhead  *(MEDIUM; profile-only, NOT landed)*
- **Observation:** `RedzoneAllocatorKernelImpl` is **26% of GPU-kernel time** in the warmed
  profile. Redzone checks guard autotuned kernels against buffer overruns; seeing them this
  prominent in steady state suggests autotuning/redzone work is more than a one-time cost.
- **Candidate levers:** `--xla_gpu_autotune_level` (lower), or disabling the redzone
  allocator check once kernels are autotuned (`--xla_gpu_redzone_scratch_max_megabytes=0`
  / autotune cache reuse via the persistent compilation cache so autotuning is amortized).
- **Why NOT landed blindly:** these flags trade off the safety of XLA's own numerical
  autotuning checks; they need a measured A/B + finiteness + idealized-gate re-confirm
  before they can be called safe. **Documented as an opportunity, not landed.** Likely
  interacts with the persistent JIT cache (autotune results should be cached across runs).

### 3. Bandwidth reduction on non-acoustic fields (gated fp32 storage)  *(LARGE; future ADR-007)*
- Once command buffers make the step bandwidth-bound (opportunity #1), the next ceiling is
  memory bandwidth. The precision matrix already authorizes FP32_GATED storage for
  u/v/theta/qv/hydrometeors, but the operational path currently forces fp64 (the acoustic
  solve detonates in fp32). A *gated* fp32-storage path on the non-acoustic transported
  fields (keeping mu/p/ph/w/qke fp64) would roughly halve the bandwidth of those fields.
- **Status:** future ADR-007 work; needs the full equivalence gate. Documented in
  `proofs/perf/fp32_downcast_plan.md`. **Not a quick safe win** — it is a precision change
  requiring full re-validation, which this sprint explicitly avoids.

### 4. Acoustic-substep `lax.scan` unroll — `GPUWRF_ACOUSTIC_UNROLL`  *(SMALL; opt-in, already landed env-gated)*
- 1.225× at unroll=4 (`unroll_ab_verdict.json`), but ~7× cold compile and a memory
  footprint that OOMs the coupled path under contention. Already landed env-gated, default
  OFF. Complementary to #1 (fewer kernels for graph capture to batch). Keep opt-in.

### 5. Device-to-device copy reduction  *(MEDIUM; downstream of #1)*
- `cuMemcpyDtoDAsync` is 20.5% of API time / 13.8% of GPU op time (129 k copies). Many are
  XLA materializing intermediate buffers between fusion regions. CUDA-graph capture (#1)
  removes the *launch* cost of these; reducing their *count* needs tighter fusion
  (donation, layout) and is a deeper XLA-scheduling investigation. Lower priority until #1
  + #2 are measured.

---

## Honest ceiling

The biggest near-term lever everyone assumed (graph-capture #1) was **measured this sprint
and rejected on the coupled path** — the 1.71× was a dynamics-only result that does not
survive contact with the physics couplers (0.84× coupled). So the honest picture is:
**no quick launch-env speedup is available for the COUPLED operational step today.** The
remaining real levers are (a) cutting the *coupled* launch tax with a coupled-aware
command-buffer / per-coupler fusion config (a real engineering investigation, not a flag),
(b) reducing XLA autotuning/redzone overhead (#2, needs a safe-A/B), and (c) bandwidth
reduction via gated fp32 storage (#3, a precision change needing full re-validation, out of
scope here). The established speedup stands unchanged at **~5× (band 5–8×) apples-to-apples
d02** (`speedup_denominator.md`); this sprint adds no speedup but removes a false assumption
and pins the real bottlenecks. No ≥10× claim is made.
