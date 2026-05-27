# Sprint Contract — Bottleneck + Optimization Potential Analysis (Gemini agy)

**Sprint ID**: `2026-05-27-bottleneck-analysis-agy`
**Created**: 2026-05-27 (parallel to codex bottleneck analysis, for cross-model robustness)
**Status**: READY — ANALYSIS ONLY, NO CODE CHANGES
**Predecessor proof objects**: same as the codex sibling sprint.

## Objective

Independent Gemini 3.5 Flash (agy) bottleneck + optimization analysis. Same scope as the codex sibling, different model angle. The manager will compare the two analyses afterwards to identify overlap (high confidence) vs disagreement (areas needing deeper investigation).

## Scope

Read the following before producing the analysis:
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/nsys_summary.json` (top kernels)
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json`
- `.agent/sprints/2026-05-27-m7-d2h-probe-opus/top_3_suspects.md`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`
- `PRECISION_POLICY.md`, `PERFORMANCE_TARGETS.md`

Then produce:

1. **Top-10 hot kernels** with classification (memory-bound / compute-bound / launch-bound) and recommended action.
2. **Fusion opportunities** (consecutive small kernels that should batch).
3. **Precision downcast candidates** (which FP64 fields can drop to FP32 / BF16 safely).
4. **Transient-memory analysis** (where the 7.28 GB 1km peak comes from; allocator-pressure mitigations).
5. **Cold-JIT-compile mitigations** (AOT compilation, persistent compile cache, shape stability).
6. **Multi-GPU readiness** (halo exchange design, shard_map vs explicit NCCL).
7. **Prioritized roadmap**: top 5 optimization sprints ranked by (wall-clock saving × correctness risk × effort).

Save the full analysis to `/tmp/gemini_bottleneck_analysis.md` and a 5-line executive summary to `/tmp/gemini_bottleneck_summary.md`.

## Hard Rules

1. **NO code changes.** Pure read-and-analyze.
2. **No GPU runtime.**
3. **Honest INCONCLUSIVE** when data is missing.
4. **No fabricated kernel names** or numbers — only what's verifiable from the proof objects + source code.

## Dispatch

- Tester: claude opus 4.7 → actually `agy` via `agy --print` non-interactive
- Wall-time: ≤ 30m (agy print-timeout)
- Worktree: none (agy reads directly from `/home/enric/src/wrf_gpu2` with `--add-dir`)
- GPU usage: NONE

## Output

- `/tmp/gemini_bottleneck_analysis.md` (full report, archived to `.agent/sprints/2026-05-27-bottleneck-analysis-agy/` after agy completes)
- `/tmp/gemini_bottleneck_summary.md` (executive summary)
