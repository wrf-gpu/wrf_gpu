# Sprint Contract: v0.14 GPT Performance/Transfer Audit

Date: 2026-06-11
Owner: manager
Assignee: GPT-5.5 xhigh in tmux
Status: PREPARED; dispatch after the active HPG correctness worker is done or
the manager explicitly declares the GPU/debug lane free.

## Objective

Independently audit the measured Canary L2 d02 72h CPU-vs-GPU wall-clock result
and find the largest concrete GPU compute, memory, transfer, synchronization,
and orchestration inefficiencies before v0.14 release.

This sprint is intentionally independent from the Fable performance audit. Do
not assume its conclusions. Produce a compact, artifact-backed answer the
manager can use to decide whether v0.14 can claim speed, needs safe fixes first,
or must defer speed claims.

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

## Constraints

- Read-only analysis by default. Do not edit model source.
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

Print when done:

`GPT PERF_REGRESSION_AUDIT DONE - see proofs/v014/gpt_performance_regression_audit.md`

## Acceptance Gate

Manager acceptance requires:

- report exists at the required path;
- existing resource/timing CSVs are parsed, not merely cited;
- transfer/synchronization risks are explicitly inspected;
- report proposes concrete benchmark commands;
- no source files changed.
