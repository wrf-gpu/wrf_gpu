# Sprint Contract — Bottleneck + Optimization Potential Analysis (Codex GPT-5.5)

**Sprint ID**: `2026-05-27-bottleneck-analysis-codex`
**Created**: 2026-05-27 (user direction: parallel bottleneck analysis, no rewrite)
**Status**: READY — ANALYSIS ONLY, NO CODE CHANGES
**Predecessors**:
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json` (kernel times)
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` (D2H invariant)
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json` (current 732s)
- `.agent/sprints/2026-05-27-m7-d2h-probe-opus/operator_map.json` (S1/S2/S3 fusion candidates)
- `src/gpuwrf/**` (current code)

## Objective

Full bottleneck and optimization-potential analysis of wrf_gpu in its current iter-2 state. Identify where wall-clock time is being spent, where memory bandwidth is the bottleneck vs where compute is the bottleneck, which kernels are launch-bound, which physics couplers can be fused, which dtypes can be downcast safely, what allocator/transient memory pressure exists, and where the largest realistic speedup opportunities are.

**No code is changed.** This sprint produces an analysis document + a prioritized optimization roadmap that downstream sprints (if approved) can execute against. The user explicitly said: "no re-write for now, just the full analysis and plan."

This is paired with `2026-05-27-bottleneck-analysis-agy` (the same analysis run by Gemini) for cross-model robustness.

## Acceptance

- **AC1 — Hot-kernel ranking**: from `.agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json` (top kernels: `pcrGtsvBatchSharedMemKernelLoop<double>`, `loop_add_fusion_4`, `loop_multiply_fusion`, etc.), list the top-10 kernels by GPU time. For each: total time, % of frame, why it's hot, recommended action (fuse with neighbour / change dtype / replace algorithm / accept as fundamental). Emit `.agent/sprints/2026-05-27-bottleneck-analysis-codex/hot_kernels.md`.

- **AC2 — Memory bandwidth vs compute**: classify each top-10 kernel as memory-bound, compute-bound, or launch-bound based on Nsight Compute metrics (or HLO-level reasoning if NCU permissions block). For memory-bound kernels, name the candidate fusion targets. For launch-bound kernels, name the candidate kernel-merge opportunities (small kernels that should batch). Emit `.agent/sprints/2026-05-27-bottleneck-analysis-codex/memory_compute_classification.md`.

- **AC3 — Operator-level audit**: read the 3-suspect Opus D2H probe (`.agent/sprints/2026-05-27-m7-d2h-probe-opus/top_3_suspects.md`). S1, S2, S3 were named there as theoretical fusion candidates but parked because D2H was already 0. Re-evaluate now: are they real optimization wins? Estimate the realistic wall-clock savings for each. Emit `.agent/sprints/2026-05-27-bottleneck-analysis-codex/operator_audit.md`.

- **AC4 — Precision downcast audit**: read `src/gpuwrf/contracts/precision.py` and `PRECISION_POLICY.md`. List every State field currently FP64; for each, judge: must-stay-FP64 (mass/pressure-gradient-sensitive), candidate-FP32 (low-impact), candidate-BF16 (advection scalars). Estimate memory + bandwidth savings if each downcast lands. Emit `.agent/sprints/2026-05-27-bottleneck-analysis-codex/precision_downcast_audit.md`.

- **AC5 — Transient memory + allocator pressure**: from the 1km audit (`.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`), peak was 7.28 GB on 32 GB. Trace where the 7.28 GB came from: persistent State, transient buffers, JIT compile cache, XLA buffer-aliasing failures. Identify allocator-pressure opportunities. Emit `.agent/sprints/2026-05-27-bottleneck-analysis-codex/memory_pressure.md`.

- **AC6 — Cold-JIT compile time**: 102-106 s cold start is operationally expensive. List options to reduce: AOT compilation (`jax.export`), persistent compile cache, shape-stable JIT, lower optimization level. Emit `.agent/sprints/2026-05-27-bottleneck-analysis-codex/cold_jit_reduction.md`.

- **AC7 — Multi-GPU readiness**: the halo placeholder exists but no MPI/NCCL work has happened. List the changes needed to make wrf_gpu multi-GPU: which fields need halo exchange, what the message size would be, what the expected communication overhead is, whether `jax.experimental.shard_map` or explicit NCCL is the better fit. Emit `.agent/sprints/2026-05-27-bottleneck-analysis-codex/multi_gpu_path.md`.

- **AC8 — Prioritized optimization roadmap**: synthesize AC1-AC7 into a ranked list of optimization sprints, each with: scope, estimated wall-time saving, estimated risk to correctness, estimated effort. Top 5 should be ranked. Emit `.agent/sprints/2026-05-27-bottleneck-analysis-codex/optimization_roadmap.md`.

- **AC9 — Worker report** verdict `ANALYSIS_READY`, with the top 3 highest-value optimization opportunities highlighted.

## Files Worker May Modify

- `.agent/sprints/2026-05-27-bottleneck-analysis-codex/**` only
- May write throwaway analysis scripts to `/tmp/m7_bottleneck_*` (uncommitted)

## Files Worker Must Not Modify

- `src/gpuwrf/**`
- `publication/**`, `publish/**`
- `tests/**`
- `scripts/**`
- governance files

## Hard Rules

1. **NO CODE CHANGES.** Pure analysis sprint.
2. **CPU pinning**: `taskset -c 0-3`.
3. **No GPU runtime.** This sprint reads existing proof objects + source code. If a measurement is genuinely needed (e.g., `jax.jaxpr` of a function), use a tiny throwaway probe under `/tmp/`.
4. **No remote push.** Local commit on `worker/gpt/bottleneck-analysis-codex` only.
5. **Honest INCONCLUSIVE** on any AC where the data isn't enough to give a confident answer — say so + name what would be needed.

## Proof Objects

- 8 markdown deliverables under `.agent/sprints/2026-05-27-bottleneck-analysis-codex/`
- `worker-report.md` with verdict

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-4 h
- Branch: `worker/gpt/bottleneck-analysis-codex`
- Worktree: `/tmp/wrf_gpu2_botcodex`
- GPU usage: NONE
