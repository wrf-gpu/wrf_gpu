# Sprint Contract: v0.14 GPT Performance/Transfer Audit

Date: 2026-06-11
Owner: manager
Assignee: GPT-5.5 xhigh in tmux
Status: PREPARED; dispatch in parallel with the Fable performance sprint only
after the Switzerland/Gotthard 72h field gate is green or explicitly
accepted/bounded by the manager.

## Objective

Independently audit the measured Canary L2 d02 72h CPU-vs-GPU wall-clock result
and find the largest concrete GPU compute, memory, transfer, synchronization,
and orchestration inefficiencies before v0.14 release.

This sprint is intentionally independent from the Fable performance audit. Do
not assume its conclusions. Produce a compact, artifact-backed answer the
manager can use to decide whether v0.14 can claim speed, needs safe fixes first,
or must defer speed claims.

GPT is report-only for this sprint. Do not edit source files, docs, or
contracts. Fable owns any simple identity-preserving speedup implementation in
the parallel sprint.

The audit must also defend or falsify the original project speed premise. For
parallel stencil/column/radiation workloads above a certain size, a GPU should
normally outperform a CPU by large factors. If the measured result stays near
`1x`, work backwards and explain whether the cause is unfair measurement,
small-grid overhead, JAX/runtime/kernel inefficiency, IO/transfer overhead, or a
real algorithmic limit. Do not accept "GPU is not faster here" without evidence.

## Trigger Data

- Canary 72h run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`
- GPU wall:
  `proofs/wall_clock_l2_d02.json` reports `8226.936 s` total and
  `8152.310 s` forecast-only.
- Approximate CPU denominator:
  retained 28-rank CPU-WRF wrfout timestamp span `8713.126 s`.
- Current observed speedup: `1.059x` total / `1.069x` forecast-only.

## Required Context

Read first:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/profiling-nvidia-gpu/SKILL.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `.agent/decisions/V0130-SPEED-ROADMAP.md`
- `.agent/decisions/M7-PERF-MEASUREMENT-CLOSEOUT.md`
- `docs/GPU_RUNBOOK.md`
- `proofs/v014/canary_d02_72h_field_gate_summary.md`
- `src/gpuwrf/integration/daily_pipeline.py`
- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/io/wrfout_writer.py`
- `src/gpuwrf/io/async_wrfout.py`
- `scripts/run_gpu_lowprio.sh`
- `scripts/monitor_resource_usage.sh`

## Required Analysis

1. Parse existing timing/resource artifacts and derive:
   - per-hour GPU wall distribution;
   - GPU memory/utilization/power trends;
   - process RSS trends;
   - wall lost outside forecast-only if any;
   - signs of compile-time being included.
2. Inspect the runtime pipeline for hidden synchronizations and transfers:
   host materializations, `np.asarray`, `device_get`, writer payload build,
   hourly wrfout output, scoring, validation, resource logging, cache settings,
   recompiles, donation/liveness blockers.
3. Inspect likely v0.13/v0.14 performance regressors:
   RRTMG g-point/column tiling, memory fixes, strict field-comparison plumbing,
   physics activation, boundary/live-nesting cadence, pressure/HPG fixes, and
   any scan/sub-jit changes.
4. Define a minimal fair benchmark matrix:
   - warm GPU forecast-only without compile;
   - with and without wrfout writer;
   - with and without resource logging;
   - CPU-WRF denominator with explicit timing rather than wrfout mtimes;
   - Switzerland d01 comparison once HPG is fixed.
5. Rank fixes by likely speedup gain and implementation risk.
6. Backwards speed-premise analysis:
   - why the rough pre-architecture `~10x` expectation is or is not still
     plausible;
   - what would have to be true for `~10x` to be impossible;
   - which artifact or profiler measurement would falsify the dominant claim.
7. Architectural near-optimum analysis:
   - whether a different graph/stencil/matrix representation, custom
     Triton/CUDA/Pallas kernels, persistent kernels, data-layout rewrite,
     larger fusion boundary, column batching, or physics batching could produce
     near-optimum GPU efficiency;
   - rank options by gain, complexity, validation risk, and WRF-faithfulness
     risk.
8. Compute-vs-memory analysis:
   - identify optional caching/precompute/residency choices that spend more
     VRAM for higher throughput;
   - estimate extra memory and speed impact;
   - reject them only if they recreate large memory-failure risk or break target
     GPU classes. Compute speed has strategic priority over extra memory
     savings when the footprint remains stable and fits target hardware.

## Constraints

- Read-only analysis only. Do not edit source files, docs, contracts, tests, or
  proof scripts.
- Do not use `ask-hermes`, Telegram, or any human notification bridge.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not run a GPU job unless the manager has freed the GPU or the command uses
  `scripts/run_gpu_lowprio.sh` and exits cleanly with rc 75 when locked.
- If you need a microbenchmark, prefer to write a proposed command and wait for
  manager launch unless it is CPU-only.
- Keep output compact; the manager context window is a scarce resource.

## Required Output

Write:

`proofs/v014/gpt_performance_regression_audit.md`

Report format:

- One-sentence verdict.
- Compact timing table from existing artifacts.
- Ranked inefficiency table with columns:
  `rank`, `component`, `evidence`, `likely impact`, `proof command`,
  `safe fix candidate`.
- Fair benchmark matrix with exact commands/artifacts.
- Top 5 immediate actions before v0.14 speed claims.
- Top 5 v0.15 performance roadmap candidates.
- Projection table for README/release notes:
  small cases (current Canary/Switzerland sizes on RTX 5090), optimal RTX 5090
  32 GB grids that still fit in VRAM, and asymptotic large-grid H200/GB300
  regime where initialization/compile is amortized. Include confidence and key
  assumptions.
- `WHY_NOT_10X_YET` section:
  classify the dominant explanation as `MEASUREMENT_UNFAIRNESS`,
  `SMALL_GRID_OVERHEAD`, `CURRENT_KERNEL_INEFFICIENT`,
  `IO_TRANSFER_ORCHESTRATION`, `REAL_ALGORITHMIC_LIMIT`, or a ranked mixture.
- `NEAR_OPTIMUM_KERNEL_PATHS` section:
  graph/matrix/stencil/custom-kernel/data-layout/persistent-kernel options with
  gain/complexity/validation-risk estimates.
- `COMPUTE_OVER_MEMORY_OPTIONS` section:
  optional caching/precompute/residency modes that could trade extra VRAM for
  speed, with memory and expected speed impact.

Print when done:

`GPT PERF_REGRESSION_AUDIT DONE - see proofs/v014/gpt_performance_regression_audit.md`

## Acceptance Gate

Manager acceptance requires:

- report exists at the required path;
- existing resource/timing CSVs are parsed, not merely cited;
- transfer/synchronization risks are explicitly inspected;
- report proposes concrete benchmark commands;
- no source files changed.
