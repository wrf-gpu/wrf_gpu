# Sprint Contract — M5 ADR-007 Precision Policy

**Sprint ID**: `2026-05-20-m5-adr007-precision-policy`
**Created**: 2026-05-20 evening by manager (Claude Opus 4.7 1M-context)
**Trigger**: Gemini stage-M4 architectural review (`.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md`) flagged FP64 throttling on consumer Blackwell as a project-existential threat to the 4-8× speedup target. User (project arbiter) approved 2026-05-20 evening: dispatch this sprint immediately after M5-S1 closes.

## Objective

Evaluate the actual FP64 vs FP32/BF16 wall-clock advantage on the M2/M4/M5 kernels using RTX 5090 + 9950X, and amend ADR-003's precision lock with a per-field Authorization Matrix gated by **operational RMSE impact** (per `feedback_validation_philosophy.md`) — not per-cell parity. Produce ADR-007 with concrete downcast permissions + named profiler evidence.

## Non-Goals

- Do NOT yet implement the precision downcast in the production code. This sprint produces the ADR and the supporting profiler artifacts; implementation lands in a follow-on sprint per ADR-007's phased rollout.
- Do NOT touch ADR-001 (backend selection) or ADR-002 (state layout) — only ADR-003 is amended.
- Do NOT optimize beyond what the profiler evidence supports — no preemptive BF16 conversions of fields that are not memory-/compute-bound.
- Mass continuity, pressure gradient accumulation, and any field flagged catastrophic-cancellation-prone stay FP64; not in scope to revisit.

## File Ownership

Worker may modify:
- `.agent/decisions/ADR-007-precision-policy.md` (new file)
- `.agent/decisions/ADR-003-dycore-precision.md` (Authorization Matrix amendment + cross-reference to ADR-007)
- `artifacts/precision-bench/*` (new directory for profiler evidence)
- `scripts/precision_bench.py` (new — profiler runner)
- `tests/test_precision_bench.py` (new — sanity test that the bench script runs)
- Worker report.

Worker may NOT modify:
- `src/gpuwrf/**/*.py` production code (precision downcast is a follow-on sprint)
- ADR-001, ADR-002, ADR-005, ADR-006, ADR-004 (unused)
- `feedback_validation_philosophy.md` or any other memory file.

## Inputs

Required read order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/profiling-nvidia-gpu/SKILL.md`
4. `.agent/skills/validating-physics/SKILL.md`
5. `.agent/decisions/ADR-001-backend-selection.md`
6. `.agent/decisions/ADR-002-state-layout.md`
7. `.agent/decisions/ADR-003-dycore-precision.md` (the document being amended)
8. `.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md` (the trigger; cite where appropriate)
9. `~/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md` (the binding-metric framework)
10. `PERFORMANCE_TARGETS.md`
11. `MILESTONE-M2-CLOSEOUT.md`, `MILESTONE-M4-CLOSEOUT.md` (existing baseline profile artifacts)
12. M5-S1 attempt-6 final artifacts under `artifacts/m5/*` (current Thompson kernel state)

## Acceptance Criteria

1. **`scripts/precision_bench.py` produces concrete wall-clock numbers**: runs the M2 column kernel, M4 dycore step, and M5-S1 Thompson step at FP64 and FP32 on the RTX 5090, capturing `nsys`/`ncu`-derived metrics: kernel duration, achieved bandwidth, register count, occupancy. Writes `artifacts/precision-bench/<kernel>-<precision>.json` per (kernel, precision) cell.
2. **CPU baseline runs on 9950X for the same workloads**: a directly-comparable single-domain Canary 3km timestep wall-clock at FP64 (matching Gen2 CPU build), used as the denominator for the speedup ratio. Use the Gen2 build env (`source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`).
3. **Projected speedup table per ADR-007** with three rows (M2 column, M4 dycore, M5 Thompson column) × three columns (FP64 GPU/CPU, FP32 GPU/CPU, BF16 GPU/CPU where applicable). Each cell cites a profiler artifact.
4. **Per-field Authorization Matrix** in ADR-007 names every field in `state` (qv, qc, qr, qi, qs, qg, Ni, Nr, T, p, rho, U, V, W, etc.) with one of: `FP64-locked` (catastrophic-cancellation-prone or mass-continuity-critical) / `FP32-OK` / `BF16-OK` / `needs-empirical-test`. Each row cites either a numerical-stability argument or an empirical-test plan. Operational RMSE impact on `U10/V10/T2` is the binding metric for "OK" verdicts, NOT per-cell parity.
5. **ADR-003 amendment** cross-references ADR-007 and removes the blanket fp64 lock for fields the Authorization Matrix permits to downcast. Phased-rollout note: actual code-level downcast lands in follow-on per-scheme sprint(s).
6. **Verdict on the 4× target feasibility** stated explicitly in ADR-007: either (a) "feasible under the proposed mixed-precision policy" with named bottleneck path, or (b) "not feasible; project requires re-scoping" with concrete alternative paths (data-center GPU, ML hybrid emulator, scope reduction).
7. `python scripts/validate_agentos.py` passes.
8. `pytest -q` passes (no regression; new precision-bench sanity test added).

## Validation Commands

```bash
python scripts/validate_agentos.py
python scripts/precision_bench.py --run-all      # produces artifacts/precision-bench/*
pytest -q
```

## Performance Metrics

For each (kernel, precision) cell:
- `kernel_wall_time_us`: median over ≥100 warm runs
- `achieved_bandwidth_gb_per_s`
- `registers_per_thread` (null acceptable on the workstation due to perfmon restriction)
- `occupancy_pct` (null acceptable)
- `kernel_launches_per_step` (Thompson should remain 1)
- For CPU baseline: `wall_time_us_per_3km_timestep` on the same workload at FP64

## Proof Object

`.agent/decisions/ADR-007-precision-policy.md` ≥4000 bytes including:
- Hardware throughput context (RTX 5090 FP64 vs FP32 vs BF16, 9950X AVX-512 FP64)
- Authorization Matrix table with citations
- Per-field stability rationale or empirical-test plan
- 4× target feasibility verdict
- Cross-references to ADR-003 amendment + `feedback_validation_philosophy.md`

## Risks

- **Profiler perfmon restriction**: workstation may not surface register/local-memory counters; ADR-007 must use available metrics (wall-clock + bandwidth) and document the gap.
- **BF16 precision-loss surprises**: some fields (rain number `Nr` near zero, fp16 underflow in `qv*rho` products) may need empirical test before downcast permission. Authorization Matrix uses `needs-empirical-test` for those rather than `BF16-OK`.
- **The 4× target may be unreachable even with mixed precision**: this is a possible honest finding. ADR-007 must surface it explicitly if the data shows it, not paper over.
- **Gemini quota** out until ~20:45 — parallel side-runner unavailable for this sprint start. Codex worker proceeds solo; Gemini parallel side-audit dispatches when quota resets if sprint is still running.

## Handoff Requirements

- Worker report ≥2000 bytes with: ADR-007 path, Authorization Matrix in concise form, per-cell profiler artifact paths, 4× verdict, ADR-003 amendment diff.
- Tester (Claude Opus 4.7 xhigh) verifies: (a) each Authorization Matrix verdict cites either a stability argument OR an empirical test, no hand-waving; (b) the 4× feasibility verdict matches the data; (c) ADR-003 amendment is internally consistent with ADR-007.
- Reviewer (codex critical-review) verifies governance + ADR cross-references.

## Dispatch Pattern (per current project policy)

- Primary worker: codex gpt-5.5 xhigh (frontrunner).
- Parallel side-runner: Gemini 3.5 when quota resets (dispatches partway through, audits the throughput math + Authorization Matrix logic + verdict).
- Tester: Claude Opus 4.7 xhigh.
- Reviewer (binding): codex critical-review (memory/governance class) with Gemini parallel side-runner default-on per large-review policy.

## Expected wall-time

Worker phase: 2-4 hours (profiler runs are the long pole; benchmark must be statistically meaningful with ≥100 warm iterations per cell).
Tester phase: 30-60 min.
Reviewer phase: 30-60 min.
Total: 4-8 hours wall-clock.

## Sequencing

This sprint dispatches **in parallel** with the M5-S1.x lookup-table-export sprint (independent file ownership: ADR-007 worker touches profiler + ADR; M5-S1.x worker touches Thompson kernel code + tables). Both must close before M6 dispatch.
