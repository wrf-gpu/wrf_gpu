# Worker Report - Bottleneck Analysis Codex

Summary: ANALYSIS_READY. This was an analysis-only sprint; no `src/`, tests, scripts, governance, goal, reviewer, tester, manager, or memory files were modified. The highest-value opportunities are: (1) fuse tiny elementwise RK/acoustic scan-carry kernels and reduce D2D materialization, (2) add persistent JAX compile cache / precompile to remove 100 s-class cold starts, and (3) capture an XLA memory profile before aliasing/transient-buffer work. Nsight Compute hardware counters remain unavailable, so memory-vs-compute classification is partly source/HLO reasoning.

## Files Changed

- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/hot_kernels.md`
- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/memory_compute_classification.md`
- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/operator_audit.md`
- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/precision_downcast_audit.md`
- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/memory_pressure.md`
- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/cold_jit_reduction.md`
- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/multi_gpu_path.md`
- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/optimization_roadmap.md`
- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/worker-report.md`

## Commands Run And Output

- `sed -n ... PROJECT_CONSTITUTION.md`, `AGENTS.md`, `CLAUDE.md`, `PROJECT_PLAN.md`, `.agent/milestones/ROADMAP.md`, `.agent/goals/M1-DONE.md`, sprint contract, and local skills.
  - Output: read required project rules, active goal, sprint contract, and `writing-gpu-kernels` / `writing-execplans` skills.
- `git branch --show-current`
  - Output: `worker/gpt/bottleneck-analysis-codex`.
- `git status --short`
  - Output before edits: clean.
- `rg --files ...`
  - Output: located predecessor proof objects and current `src/gpuwrf/**` files.
- `taskset -c 0-3 jq ... nsys_summary.json`
  - Output: kernel_count `645341`, total GPU kernel time `0.792807785` s, longest kernels include PCR, add, multiply, reverse, subtract.
- `taskset -c 0-3 jq ... ncu_hot_kernels.json`
  - Output: `BLOCKED-PROFILER`; all top-three NCU attempts failed due `ERR_NVGPUCTRPERM`.
- `taskset -c 0-3 jq ... d2h_audit_v2.json`
  - Output: `PASS`; `d2h_inter_kernel_inside_window=0`, `h2d_inside_loop_window=0`, 25 pre-kernel D2H copies remain classified as allowed boundary I/O.
- `taskset -c 0-3 ls -la .agent/sprints/2026-05-27-m7-d2h-probe-opus`
  - Output: only `sprint-contract.md`; requested `top_3_suspects.md` and `operator_map.json` are absent.
- `taskset -c 0-3 jq ... pipeline_run_20260521.json`
  - Output: 24h total wall `732.6321056330053` s, forecast-only `687.8953256039749` s, first two hourly segments `294.3183872150039` and `264.3168711899998` s, steady later hours ~`5.87` s.
- `taskset -c 0-3 jq ... step_feasibility.json`
  - Output: 1 km warm step PASS, cold compile-inclusive `70.41960362600003` s, peak `7278` MiB, known resident bytes `661541896`, transient estimate `6969994232`.
- `taskset -c 0-3 sed -n ... src/gpuwrf/contracts/precision.py PRECISION_POLICY.md src/gpuwrf/contracts/state.py`
  - Output: current State has 47 fields; FP64 locked pressure/geopotential/mass fields and FP32-gated advected/moist fields.
- `taskset -c 0-3 sqlite3 ...`
  - Output: top cumulative kernels by GPU time are `loop_add_fusion_4` 244.292 ms, FP64 PCR loop 169.201 ms, `loop_multiply_fusion` 158.696 ms, `loop_subtract_fusion` 143.873 ms, FP64 PCR first pass 30.734 ms.
- `taskset -c 0-3 git diff --check`
  - Output: empty stdout/stderr; exit 0.
- `taskset -c 0-3 find .agent/sprints/2026-05-27-bottleneck-analysis-codex -maxdepth 1 -type f -printf '%f\n' | sort`
  - Output: the eight deliverable markdown files, `worker-report.md`, `sprint-contract.md`, and pre-existing hidden worker helper files.
- `taskset -c 0-3 wc -c .agent/sprints/2026-05-27-bottleneck-analysis-codex/worker-report.md`
  - Output before this final report update: `4544 .agent/sprints/2026-05-27-bottleneck-analysis-codex/worker-report.md`.
- `taskset -c 0-3 rg -n 'Summary:|ANALYSIS_READY' .agent/sprints/2026-05-27-bottleneck-analysis-codex/worker-report.md`
  - Output before this final report update: line 3 contains `Summary: ANALYSIS_READY`; line 47 contains the verdict token.
- `taskset -c 0-3 git status --short`
  - Output before staging: only the nine new sprint-folder markdown files were untracked.

## Proof Objects Produced

- Eight markdown deliverables listed under Files Changed.
- This report with verdict `ANALYSIS_READY`.

## Risks

- AC2 is not hardware-counter complete because NCU permissions block achieved bandwidth, occupancy, and register-spill metrics.
- AC3 is partially INCONCLUSIVE because the referenced Opus `top_3_suspects.md` and `operator_map.json` do not exist in this worktree.
- Wall-time savings are estimates from existing traces, not post-change measurements.

## Handoff

- objective: full bottleneck and optimization-potential analysis of current iter-2 `wrf_gpu`, with no code changes.
- files changed: sprint-folder markdown files only.
- commands run: listed above with outputs.
- proof objects produced: the eight analysis docs and this report.
- unresolved risks: NCU unavailable; Opus suspect artifacts missing; HLO memory attribution still needed.
- next decision needed: approve or reject the rank-1 launch/D2D fusion sprint and the separate compile-cache/precompile sprint.
