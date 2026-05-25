# Sprint Contract — M6b D2H Warmed Re-capture (opus tester)

## Objective

D2H grep verdict (commit `tester/opus/m6b-d2h-grep`): the 53 D2H transfers Nsight captured in M6b were XLA first-graph constant-staging copies, taken before the mandatory warm-up call. Constitutional invariant not violated; just a profiling-discipline error in the M6b acceptance script.

This small sprint **re-runs Nsight with proper warm-up** (warm-up call inside `cudaProfilerStart`/`cudaProfilerStop` window) and confirms steady-state transfers = 0.

## Non-Goals

- NO code edits beyond a tiny Nsight orchestrator script change (the warm-up insertion).
- NO modifications to `wrf.exe`.
- NO modifications to validation-mode or operational-mode dynamics.
- NO sub-sprint dispatch.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_d2hwarm` on branch `tester/opus/m6b-d2h-warmed-recapture`.

Write-only:
- `scripts/m6b_d2h_warmed_recapture.py` (NEW) — runs operational_mode for 1 warm-up step (outside profile window), then `cudaProfilerStart`, runs 5 more steps under profile, `cudaProfilerStop`. Captures Nsight trace.
- `tests/test_m6b_d2h_warmed_zero.py` (NEW) — asserts D2H = 0 inside the profile window
- `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/` — proofs + memo

Read-only everywhere else.

## Inputs

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-d2h-grep/d2h_localization.md` (the artifact finding + warm-up recommendation)
3. `src/gpuwrf/runtime/operational_mode.py` (post carry-expansion)
4. `PROJECT_CONSTITUTION.md` (no H2D/D2H in timestep loop)

## Acceptance Criteria

### Stage 1 — Warmed Nsight capture (MANDATORY)

Run operational_mode 1 step (warm-up), then 5 steps under nsys profile. Parse trace for D2H inside profile window.

Expected: **0 D2H** inside profile window. (D2H may be present in warm-up step; that's XLA bookkeeping.)

If non-zero: localize via grep (per D2H grep's prior recommendation list); document; route to a fix sprint.

### Stage 2 — Test

`tests/test_m6b_d2h_warmed_zero.py` asserts D2H = 0 inside warmed window.

### Stage 3 — Memo

`d2h_warmed_memo.md` with: trace summary, D2H count, comparison to M6b original (53 vs warmed N), GO/NO-GO for M6b RETRY's transfer-cleanliness gate.

### Stage 4 — No regression

`pytest tests/test_m6b_carry_expansion_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_d2h_warmed_*.py -v` all PASS.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_d2hwarm
taskset -c 0-3 python scripts/m6b_d2h_warmed_recapture.py 2>&1 | tee .agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/proof_warmed_run.txt
nsys stats --report cudaapi proof_warmed.nsys-rep 2>&1 | tee .agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/proof_warmed_trace_summary.txt
pytest tests/test_m6b_carry_expansion_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_d2h_warmed_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/proof_no_regression.txt
```

## Risks

- Nsight may not have a clean `cudaProfilerStart` Python binding; fall back to `nsys profile --capture-range=cudaProfilerApi` + a small ctypes call OR run nsys with `-s none --trace=cuda,nvtx` and use NVTX markers.

## Handoff Requirements

When proofs + memo committed: stop. Manager merges and dispatches M6b honest 1h RETRY.

Time budget: **45-90 min**.
