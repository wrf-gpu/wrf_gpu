# Sprint Contract: v0.14 Fable Performance Regression Audit

Date: 2026-06-11
Owner: manager
Assignee: Fable medium/high in a fresh tmux window
Status: PREPARED; dispatch after the active HPG correctness worker is done or
the manager explicitly declares the GPU/debug lane free.

## Objective

Explain the v0.12-to-v0.14 wall-clock speed regression and identify the
highest-leverage path back to a defensible GPU speedup before v0.14 release.

The immediate trigger is the accepted Canary L2 d02 72h field gate:

- GPU total wall: `8226.936 s`
- GPU forecast-only wall: `8152.310 s`
- approximate CPU-WRF wall from retained 28-rank wrfout timestamps:
  `8713.126 s`
- observed speedup: only `1.059x` total / `1.069x` forecast-only

This is not acceptable as a speed headline. The sprint endpoint is a
manager-actionable root-cause report: either name the dominant regression
mechanisms and the shortest safe recovery plan, or prove that the current
benchmark is not a fair speed comparison and define the correct benchmark.

## Required Context

Read first:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `.agent/decisions/V0130-SPEED-ROADMAP.md`
- `.agent/decisions/M7-PERF-MEASUREMENT-CLOSEOUT.md`
- `.agent/sprints/2026-06-10-v015-fable-kernel-efficiency-review/sprint-contract.md`
- `proofs/v014/canary_d02_72h_field_gate_summary.md`
- `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z/proofs/wall_clock_l2_d02.json`
- `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z/resources/`
- `src/gpuwrf/integration/daily_pipeline.py`
- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/io/async_wrfout.py`
- `src/gpuwrf/io/wrfout_writer.py`

## Questions To Answer

1. What changed between the v0.12 speed story and the v0.14 Canary 72h gate?
   Separate domain size, CPU denominator, physics enabled, writer/scoring/atlas,
   JAX compilation/cache, memory tiling/chunking, Python orchestration, and
   actual device kernel time.
2. Which parts of the `8226.936 s` GPU wall are forecast compute, compile,
   host IO/writer, scoring/comparison, resource logging, synchronization, and
   other overhead?
3. Is the retained CPU denominator fair? If not, define the fair denominator
   and exact command/artifact to produce it.
4. Are there obvious high-gain inefficiencies introduced by v0.13/v0.14 fixes
   that could plausibly explain the loss of speedup without changing physics?
5. What is the shortest rigorous route to a `>2x` fair wall-clock speedup, or
   to an honest proof that this grid/physics configuration cannot show it yet?

## Constraints

- Read-only analysis by default. Do not edit model source.
- Do not use `ask-hermes`, Telegram, or any human notification bridge.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not run a GPU job unless the manager has freed the GPU or the command uses
  `scripts/run_gpu_lowprio.sh` and exits cleanly with rc 75 when locked.
- Prefer small CPU-only parsing/profiling scripts and existing timing artifacts
  first.
- If a short GPU microbenchmark is essential, write the command and expected
  duration in the report and wait for manager approval unless the GPU is
  demonstrably idle and no validation/debug run is active.

## Required Output

Write:

`.agent/reviews/2026-06-11-v014-fable-performance-regression-audit.md`

Report format:

- Verdict paragraph, max 150 words.
- Ranked table of root-cause candidates with columns:
  `rank`, `mechanism`, `evidence`, `estimated wall impact`, `confidence`,
  `fix path`, `risk`.
- Fair-benchmark decision: exact CPU/GPU commands or artifacts needed.
- Recovery plan to `>2x`, with no more than 5 ordered actions.
- Any no-go signal for v0.14 release speed claims.
- Context-sparing manager handoff, max 10 bullets.

Print when done:

`FABLE PERF_REGRESSION_AUDIT DONE - see .agent/reviews/2026-06-11-v014-fable-performance-regression-audit.md`

## Acceptance Gate

Manager acceptance requires:

- the report exists at the required path;
- it uses the current Canary artifacts rather than memory alone;
- it separates measurement unfairness from real GPU inefficiency;
- it identifies at least one concrete next benchmark/profiling command;
- it does not modify source files.
