# v0.17 all-7 nested GPU utilization investigation

Scope: non-invasive log and source analysis only. No GPU lock was taken, no profiler was run, and the live release-critical run was not touched. The utilization numbers below are from a one-shot read of `proofs/v017/gpu_util_log.csv` while it was still being appended.

## Executive finding

The all-7 live-nested path is not blocked by more single-domain kernel optimization. It is dominated by host-orchestrated nested cadence: thousands of small compiled domain advances, explicit host synchronizations between them, eager boundary-package construction between domains, and synchronous output/materialization at history boundaries.

The highest-confidence root cause is the `block_between=True` nested recursion:

- `src/gpuwrf/runtime/domain_tree.py:288-302` advances one domain and immediately `jax.block_until_ready(...)`s.
- `src/gpuwrf/runtime/domain_tree.py:296-314` serializes parent step -> force child boundary -> child subcycle -> next sibling.
- `src/gpuwrf/integration/nested_pipeline.py:758-765` hard-codes `block_between=True` for the production nested pipeline.

For the all-7 geometry, one forecast hour has about 200 root steps. Each root step issues approximately 25 `_advance_chunk` calls/sync points and 22 forced-boundary package builds, so one forecast hour is roughly 5,000 compiled JAX calls/sync points plus 4,400 boundary builds. That is the right order of magnitude for the observed GPU-busy bursts separated by host/API gaps.

## Measured utilization pattern

Final parsed snapshot:

- Samples: 254
- Time span: `2026-06-15 17:27:58.245` to `2026-06-15 17:36:24.305`
- Cadence: median 2.0 s
- Full-file time-weighted utilization:
  - `<10%` GPU util: 42.9%
  - `<20%` GPU util: 45.3%
  - `>=50%` GPU util: 50.4%
  - `>=80%` GPU util: 39.8%
  - mean util: 44.8%, median util: 51.0%

The full file is non-stationary. It contains later sustained busy regions, so full-file idle understates the release-critical oscillatory behavior.

Worst rolling windows in the same CSV:

- 15 samples, about 28 s: 14/15 samples `<10%` util, mean util 3.4%, from `17:34:24.292` to `17:34:52.295`.
- 30 samples, about 58 s: 24/30 samples `<10%` util, mean util 14.4%, from `17:28:02.249` to `17:29:00.256`.
- 60 samples, about 118 s: 35/60 samples `<10%` util, mean util 32.2%, from `17:28:22.251` to `17:30:20.265`.

Oscillatory prefix before `17:30:24`:

- `<10%` idle fraction: 57.6%
- `>=50%` busy fraction: 34.1%
- mean util: 32.5%, median util: 5.0%
- busy-burst duration (`>=50%`): median 2.0 s, p75 4.0 s, with occasional 28-32 s busy cascades.
- idle-run duration (`<10%`): median 4.0 s, mean 6.0 s, max 16.0 s.
- non-busy gap after a busy burst: median 10.0 s, mean 10.4 s, max 22.0 s.
- busy-start period: median 14.0 s, mean 15.8 s.

Full-file segment summary:

- busy-burst duration (`>=50%`): median 2.0 s, p75 12.0 s, max 44.0 s.
- idle-run duration (`<10%`): median 4.0 s, mean 6.4 s, max 28.0 s.
- non-busy gap after a busy burst: median 6.0 s, mean 9.3 s, max 32.0 s.
- memory.used ranged 15,798 to 21,277 MiB, consistent with repeated chunk allocation/freeing rather than a single resident long scan.

Interpretation: the 2 s sampling cadence quantizes short bursts. A reported 2 s burst may be a sub-2 s kernel cluster caught by one sample, and a 4 s gap may contain launch/API work. The pattern is still clear: the GPU queue frequently drains between domain chunks.

## Static root-cause audit

### 1. Explicit per-domain synchronization in the nested recursion

Evidence:

- `src/gpuwrf/runtime/domain_tree.py:283-314`: recursive nested scheduler. Parents are advanced, children are forced, then child subcycles are run one child at a time.
- `src/gpuwrf/runtime/domain_tree.py:288-292`: leaf-domain advance calls `advance(...)`, updates counters, then blocks on `theta`.
- `src/gpuwrf/runtime/domain_tree.py:298-302`: non-leaf parent advance also blocks on `theta` after each single parent step.
- `src/gpuwrf/runtime/domain_tree.py:376-385`: `run_operational_domain_tree(..., block_between=True)` defaults to syncing between chunks.
- `src/gpuwrf/integration/nested_pipeline.py:758-765`: production nested pipeline passes `block_between=True`.

Why this idles the GPU:

JAX dispatch is asynchronous, but `block_until_ready` drains the queue after every domain advance. After the sync returns, Python must walk the recursion, build events, construct boundary packages, and dispatch the next child/domain chunk. During that host-side interval there is no queued GPU work, so utilization drops to near zero.

This sync is not required for WRF live-nesting semantics. The semantic dependency is dataflow: parent state must feed child boundary construction before the child advances. JAX device dependencies can preserve that without a host sync after every chunk. The sync is a memory-lifetime policy, not a physics policy.

### 2. Very fine `_advance_chunk` granularity in all-7 nesting

Evidence:

- `src/gpuwrf/runtime/domain_tree.py:327-339`: `_operational_advance_factory` calls `_advance_chunk`.
- `src/gpuwrf/runtime/operational_mode.py:4282-4316`: `_advance_chunk` is a jitted `lax.scan` over `n_steps`.
- `src/gpuwrf/runtime/domain_tree.py:296-309`: non-leaf domains advance one step at a time, then child leaves advance `parent_grid_ratio` steps.

All-7 cadence estimate:

- Geometry: d01 -> d02 -> seven 1 km children, all ratios 3 (`proofs/v017/canary_all7_geometry/coverage_report.md:38-48`).
- Root dt is 18 s (`proofs/v017/canary_all7_geometry/coverage_report.md:91-93`), so one forecast hour is 200 root steps.
- Per root step:
  - d01 advance: 1 `_advance_chunk`.
  - force d01->d02: 1 boundary package.
  - three d02 substeps, each doing d02 advance + force seven children + advance seven leaf children.
  - Total: `1 + 3*(1+7) = 25` `_advance_chunk` calls/sync points.
  - Total boundary force builds: `1 + 3*7 = 22`.
- Per forecast hour: about 5,000 `_advance_chunk` calls/syncs and 4,400 boundary force builds.

Why this idles the GPU:

Each `_advance_chunk` can be efficient internally, but thousands of small compiled calls per forecast hour expose Python dispatch, CUDA launch/API overhead, allocator work, and explicit sync overhead. This is a nested-host-loop issue, not the single-domain roofline issue previously studied.

### 3. Forced child-boundary package construction is eager JAX outside `_advance_chunk`

Evidence:

- `src/gpuwrf/runtime/domain_tree.py:344-353`: `_operational_force` calls `build_child_boundary_package(...)` between parent and child advances.
- `src/gpuwrf/nesting/boundary_construction.py:219-297`: boundary package construction interpolates parent fields, extracts child rings, pads/fits, and stacks two-time boundary leaves.
- `src/gpuwrf/nesting/boundary_construction.py:256-263`: eight parent-field interpolations per forced edge.
- `src/gpuwrf/nesting/boundary_construction.py:246-254`: per-leaf old/new ring packing.
- `src/gpuwrf/nesting/interp.py:248-276`: interpolation is a static-index gather using `jnp.take`.
- `src/gpuwrf/nesting/interp.py:1-7` and `src/gpuwrf/nesting/interp.py:71-81`: weights are precomputed and device-resident, so this is not a per-step host weight build.

Why this idles the GPU:

The weights are already precomputed, which is good. The remaining problem is that `build_child_boundary_package` is not captured as part of a larger jitted cascade. It runs as eager JAX ops between compiled domain chunks, creating many small device kernels plus Python dispatch boundaries. With 4,400 force builds per forecast hour, even small per-build gaps matter.

### 4. Platform allocator and output-interval segmentation trade memory for utilization

Evidence:

- `src/gpuwrf/cli.py:300-346`: nested runs re-exec with `XLA_PYTHON_CLIENT_ALLOCATOR=platform` unless the operator overrides it. The comment explicitly says this uses synchronous `cudaMalloc/cudaFree`.
- `src/gpuwrf/cli.py:449-454`: nested path sets platform allocator before backend init.
- `src/gpuwrf/integration/nested_pipeline.py:675-689`: nested pipeline also sets `XLA_PYTHON_CLIENT_ALLOCATOR=platform` to avoid BFC fragmentation.
- `src/gpuwrf/integration/nested_pipeline.py:730-777`: forecast is driven one output interval at a time and blocks after each segment (`jax.block_until_ready` at lines 771 and 777).

Why this idles the GPU:

The platform allocator was introduced for a valid reason: long nested runs previously OOMed from allocator fragmentation. But synchronous `cudaMalloc/cudaFree` plus frequent segment/domain boundaries can create visible CUDA API stalls. The memory-used sawtooth in the utilization log supports this as a likely contributor. Nsight Systems is needed to separate allocator/API time from Python dispatch time.

### 5. Hourly wrfout and finite summaries cause real device-to-host transfers

Evidence:

- `src/gpuwrf/runtime/domain_tree.py:266-280`: `maybe_output` calls the output callback at cadence.
- `src/gpuwrf/integration/nested_pipeline.py:616-652`: `_PerDomainWrfoutWriter.__call__` computes diagnostics and writes wrfout synchronously.
- `src/gpuwrf/integration/nested_pipeline.py:543-563`: Noah-MP output diagnostics use `jax.device_get` and `np.asarray` for each emitted M9 field.
- `src/gpuwrf/io/wrfout_writer.py:919-969`: `write_wrfout_netcdf` prepares and writes a host-materialized NetCDF payload.
- `src/gpuwrf/io/wrfout_writer.py:996-1029`: `prepare_wrfout_payload` explicitly materializes all output fields at the device-to-host boundary.
- `src/gpuwrf/io/wrfout_writer.py:1252-1421`: `_build_output_fields` pulls the core prognostic/output fields.
- `src/gpuwrf/io/wrfout_writer.py:2055-2056`: `_coerce_array` uses `np.asarray(value, dtype=...)`, which materializes JAX arrays on host.
- `src/gpuwrf/integration/daily_pipeline.py:619-640`: `finite_summary` uses `np.asarray(value)` on every numeric state leaf.
- `src/gpuwrf/integration/nested_pipeline.py:652` and `src/gpuwrf/integration/nested_pipeline.py:787`: nested path still calls the full host finite summary.

Why this idles the GPU:

These transfers are at output/final-summary boundaries, not inside every dynamics step. They are still throughput-relevant for the all-7 product because all domains emit hourly wrfout, and the nested path writes synchronously. These stalls should be separated from the high-frequency per-domain gaps in nsys.

### 6. Two-way feedback is eager by design, but likely not active in this run

Evidence:

- `src/gpuwrf/runtime/domain_tree.py:356-373`: `_operational_feedback` runs `apply_state_feedback` eagerly and documents that this is intentional to lower peak VRAM.
- `src/gpuwrf/coupling/boundary_feedback.py:547-593`: `apply_state_feedback` applies per-leaf gather/scatter/smoothing and rebuilds totals.
- `proofs/v017/canary_all7_geometry/coverage_report.md:30-32`: all-7 geometry says `feedback = 0`.
- `src/gpuwrf/integration/nested_pipeline.py:718-719`: feedback follows `config.feedback`, default false.

Conclusion:

Feedback is a real future utilization risk if two-way nesting is enabled, but it is probably not the current all-7 one-way gap source.

## Achievable-speed estimate

Given observed warm all-7 rate from the prompt:

- Current GPU: about 20 min per forecast-hour.
- CPU baseline: about 15 min per forecast-hour on 12 ranks.
- Prompt's effective busy fraction: about 20% (80% idle).

Simple utilization-only scaling:

| Effective busy target | Projected GPU min/forecast-hour | vs 12-rank CPU 15 min/hr |
|---:|---:|---:|
| 50% | 8.0 | 1.9x faster |
| 60% | 6.7 | 2.2x faster |
| 80% | 5.0 | 3.0x faster |
| 100% | 4.0 | 3.8x faster |

Break-even is only about 27% effective busy: `20 min/hr * 20% / 26.7% = 15 min/hr`.

Conservative interpretation:

- If the full longer CSV snapshot's mean utilization, about 45%, is closer to steady state than the worst windows, then a target of 80-100% maps to roughly 11.2-9.0 min/hr. That still beats the 12-rank CPU.
- Not every gap is closable: parent->child live-nesting dependencies are real, hourly output transfers are real, and platform allocator may be needed for memory safety. But the explicit per-domain host syncs are not fundamental to WRF semantics, and closing even half of the high-frequency idle should flip the all-7 run from slower-than-CPU to a GPU win.

Bottom line: yes, this likely flips to a GPU win if the major host gaps are closed without increasing compute work or causing OOM. The realistic first target is 6-10 min/forecast-hour, not the theoretical 4 min/hr.

## Ranked fix options

### 1. Make nested dispatch asynchronous between domains; sync only at output/segment boundaries

Change idea:

- Expose/enable `block_between=False` in the nested production path.
- Keep the same Python recursion order, but do not call `jax.block_until_ready` after every domain advance.
- Rely on JAX data dependencies for parent state -> boundary package -> child advance ordering.
- Keep an explicit sync at wrfout/output boundaries and at final segment end.

Expected payoff:

- High. Directly targets the 5,000 per-hour sync points.
- Expected idle reduction: 30-60% of high-frequency gaps.
- Plausible wallclock: 8-12 min/hr first, possibly better if allocator stalls also shrink.

Effort:

- Small to medium. There is already a `block_between` parameter, but production currently hard-codes true. Needs memory/VRAM A/B because more work may stay live before host sync.

Bit-identity risk:

- Low numerical risk. Removing host waits should not change JAX data dependencies or operation order.
- Medium operational risk from VRAM lifetime and allocator behavior.

Gate:

- 1 h all-7 A/B: `block_between=True` vs false, compare wrfout hashes/field deltas, peak VRAM, nsys gap profile.

### 2. JIT/capture one nested cascade unit instead of dispatching per-domain chunks

Change idea:

- Build a compiled "one d02-step cascade" or "one d01-step cascade" that includes parent advance, boundary package construction, and child subcycles in one JAX program.
- Keep WRF parent-before-child semantics inside `lax.scan`/structured loops.

Expected payoff:

- Very high if current gaps are launch/API dominated.
- Expected idle reduction: 50-80% of high-frequency gaps.

Effort:

- Medium to high. It crosses current module boundaries and may increase compile size/VRAM.

Bit-identity risk:

- Low to medium if the same operations are preserved in the same order.
- Higher if XLA fuses/reassociates floating-point expressions across current boundaries.

Gate:

- Start with one d02 step plus two child domains, then all-7 10 root steps, then 1 h all-7. Require field identity or accepted roundoff envelope before using for release.

### 3. JIT the boundary package per edge, and optionally batch all child-force builds under d02

Change idea:

- Wrap `build_child_boundary_package` in an edge-specialized `jax.jit` with static geometry.
- For d02's seven children, dispatch/capture forced-boundary package construction together where shapes allow, or at least reduce eager op chains per edge.

Expected payoff:

- Medium. Targets 4,400 per-hour force builds.
- Expected idle reduction: 10-30%, larger if nsys shows boundary eager kernels dominate gaps.

Effort:

- Small to medium for per-edge jit.
- Medium for all-child batched force construction because shapes differ.

Bit-identity risk:

- Low for per-edge jit: same JAX ops.
- Medium for batched/padded variants.

### 4. CUDA Graphs / XLA command buffer for the repeated nested launch sequence

Change idea:

- Capture repeated `_advance_chunk`/boundary/child sequence to reduce CUDA launch/API overhead.
- Use only after nsys proves API launch gaps are large.

Expected payoff:

- Medium to high if the trace shows many tiny launch gaps and CUDA API overhead.
- Expected idle reduction: 20-50%.

Effort:

- Medium. Prior single-domain command-buffer work was wall-neutral, but this nested host-loop is a different problem and should be re-tested.

Bit-identity risk:

- Low. Command capture should not change math.

Operational risk:

- Tooling risk and profiler interaction. Needs warm-cache A/B and all-7 compile-cache care.

### 5. Revisit allocator policy: platform vs cuda_async/BFC for all-7

Change idea:

- Keep platform allocator as the safe default, but profile platform API stalls.
- If stalls dominate, test `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` or default BFC on a short all-7 run with peak VRAM tracking.

Expected payoff:

- Medium if cudaMalloc/cudaFree shows up as large gaps.
- Expected idle reduction: 10-30%.

Effort:

- Small for A/B, medium for making it safe by segment size/VRAM policy.

Bit-identity risk:

- None numerically.

Operational risk:

- Medium to high OOM/fragmentation risk. This is why platform allocator exists.

### 6. Batch same-shape sibling nests

Change idea:

- d06 and d07 are both 40x40 (`proofs/v017/canary_all7_geometry/coverage_report.md:43-44`); run them as a leading batch dimension where the code can tolerate it.
- Consider shape buckets or padding for other small island nests.

Expected payoff:

- Low to medium. It reduces launch count and improves occupancy for tiny nests, but the two 40x40 nests are only 3,042 mass columns combined.
- Expected idle reduction: 5-15%.

Effort:

- Medium to high because state, metrics, boundary packages, output, and proof comparisons must handle batched domains or padding.

Bit-identity risk:

- Medium. Padding/masking and batched reductions can shift exact operation order.

### 7. Async/output-side cleanup for nested wrfout and finite checks

Change idea:

- Use the single-domain pattern: materialize output once on the main thread, write NetCDF in a background writer.
- Replace success-path `finite_summary` with the existing device-side `finite_guard_summary` style for nested final/hourly checks.

Expected payoff:

- Low for high-frequency oscillation, medium for hourly stalls.
- Expected idle reduction: 0-10% overall, but can remove obvious per-hour dead time.

Effort:

- Small to medium.

Bit-identity risk:

- Low if payload materialization remains identical.

## Proposed nsys run for manager approval

Do not run while `opus-all7-validate` holds the GPU lock. After the live run frees the lock, run a 1 h all-7 profile with the same platform allocator first. This captures the current release behavior and should be short enough to inspect without burning a full 24 h run.

```bash
RR=<DATA_ROOT>/wrf_gpu_validation/v017_all7_nsys_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RR/gpu_output" "$RR/proofs" "$RR/scratch"

cd <USER_HOME>/src/wrf_gpu2
scripts/with_gpu_lock.sh --timeout 21600 --label manager-all7-nsys -- bash -lc "
set -euo pipefail
cd <USER_HOME>/src/wrf_gpu2/.wt-rc
export PYTHONPATH=\$PWD/src
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMP_NUM_THREADS=4
export GPUWRF_MYNN_BOULAC_ONZ=1
export GPUWRF_SCRATCH=$RR/scratch

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=process-tree \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  --force-overwrite=true \
  --output=$RR/nsys_all7_1h \
  taskset -c 0-3 python -m gpuwrf run \
    --input-dir <DATA_ROOT>/wrf_downscale/canary_all7/run \
    --namelist <DATA_ROOT>/wrf_downscale/canary_all7/run/namelist.input \
    --output-dir $RR/gpu_output \
    --proof-dir $RR/proofs \
    --scratch-dir $RR/scratch \
    --max-dom 9 \
    --hours 1

nsys stats --force-overwrite=true --force-export=true \
  --report cuda_gpu_kern_sum --report cuda_gpu_sum --report cuda_api_sum --report nvtx_sum \
  --format csv --output $RR/nsys_all7_1h_stats $RR/nsys_all7_1h.nsys-rep || true

python <USER_HOME>/src/wrf_gpu2/scripts/m7_nsys_stats_extract.py \
  $RR/nsys_all7_1h.nsys-rep \
  --sqlite $RR/nsys_all7_1h.sqlite \
  --output $RR/nsys_all7_1h_summary.json || true

python <USER_HOME>/src/wrf_gpu2/scripts/m7_d2h_audit.py \
  $RR/nsys_all7_1h.nsys-rep \
  --sqlite $RR/nsys_all7_1h.sqlite \
  --marker-regex 'XlaModule:#hlo_module=jit__advance_chunk\\b' \
  --output $RR/nsys_all7_1h_d2h_audit.json || true
"
```

What to inspect:

- CUDA API time in `cuda_api_sum`: `cudaMalloc`, `cudaFree`, `cuLaunchKernel`, graph launches if any.
- Gaps between kernels in the SQLite timeline.
- Memcpy rows: distinguish hourly wrfout D2H from unexpected inter-domain D2H.
- NVTX/XLA module cadence: count `_advance_chunk` launches and verify per-hour launch count against the 5,000 estimate.
- CPU OS runtime rows: NetCDF writes, filesystem, and Python waits at output boundaries.

Optional second run, only if the platform trace shows allocator API stalls and manager accepts OOM risk:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
# rerun the same 1 h nsys command to isolate allocator-induced idle.
```

## Handoff

Objective: investigate all-7 nested GPU utilization gaps from log and code only, without disturbing the live release-critical run.

Files changed:

- `proofs/v017/gpu_utilization_investigation.md`

Commands run:

- Read project instructions: `sed -n` on `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `.agent/decisions/V017-PERFORMANCE-RELEASE-PLAN.md`.
- Read local skill: `.agent/skills/profiling-nvidia-gpu/SKILL.md`.
- Parsed `proofs/v017/gpu_util_log.csv` with CPU-only Python pinned to cores 0-3.
- Static source reads/searches with `rg`, `nl`, `sed`, all pinned to cores 0-3.
- Checked all-7 input path metadata under `<DATA_ROOT>/wrf_downscale/canary_all7/run`.

Proof objects produced:

- This report.

Unresolved risks:

- No nsys trace yet, so allocator/API/launch/D2H attribution is a static hypothesis plus log correlation.
- Full-file utilization is non-stationary; the worst rolling windows confirm the prompt's idle pattern, but a profiler trace must determine how representative each phase is.
- Removing syncs may raise peak VRAM. The current sync/allocator policy was introduced to avoid nested OOMs.

Next decision needed:

- Manager should approve and run the proposed 1 h all-7 nsys capture after the live `opus-all7-validate` run releases the GPU lock.
