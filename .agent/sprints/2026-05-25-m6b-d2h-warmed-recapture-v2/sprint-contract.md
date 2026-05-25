# Sprint Contract — M6b D2H Warmed Re-capture v2 (opus, parallel with M6b V3)

## Objective

Standalone bisect verdict honestly noted: "D2H warmed zero was not newly proven; existing warmed summary still records nonzero inter-kernel D2H." After the reframe + theta-offset fix, the D2H transfer profile may have changed. Re-run the warmed Nsight capture with the current operational_mode.py to either:
- Confirm inter-kernel D2H = 0 (per ADR-027 invariant), OR
- Document the remaining call sites for a targeted lift sprint.

## Non-Goals

- NO modifications to `operational_mode.py` body (M6b V3 is running in parallel; their lock).
- NO modifications to `validation_wrappers.py` or `dynamics/core/`.
- NO new operator semantics.
- NO sanitizer.
- NO 1h forecast.
- NO touching `/tmp/wrf_gpu2_m6b_v3` (codex worker is there).
- NO remote push.
- Stay on CPU cores 0-3.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_d2h_v2` on branch `tester/opus/m6b-d2h-warmed-recapture-v2`.

Write-only:
- `scripts/m6b_d2h_warmed_recapture.py` — extend with explicit pre-/inter-kernel D2H breakdown if not already
- `tests/test_m6b_d2h_warmed_zero_v2.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2/` — proofs + memo

Read-only everywhere else.

## Inputs

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/d2h_warmed_memo.md` (the prior NO-GO)
3. `.agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/worker-report.md` (the prior lift work)
4. `.agent/decisions/ADR-027-d2h-invariant-clarification-DRAFT.md`
5. `src/gpuwrf/runtime/operational_mode.py` (current, post-reframe)

## Acceptance Criteria

### Stage 1 — Warmed Nsight capture (MANDATORY)

3 warm-up calls outside `cudaProfilerStart`, then 5-step profile window inside. Filter D2H events; decompose into pre-kernel (XLA bookkeeping, OK per ADR-027) vs inter-kernel (constitutional violation if >0).

### Stage 2 — Decompose + memo (MANDATORY)

Per-cluster D2H attribution by source kernel. Compare to the prior `tester/opus/m6b-d2h-warmed-recapture` finding (20 inter-kernel D2H from `loop_add_fusion_63` + `input_transpose_fusion_102`). Has the post-reframe operational_mode.py reduced this?

### Stage 3 — Verdict (MANDATORY)

`d2h_warmed_memo_v2.md`:
- **GO-D2H-CLEAR**: inter-kernel D2H == 0 → ADR-027 invariant satisfied → ADR-027 DRAFT → PROPOSED
- **GO-D2H-LIFT-SPRINT**: inter-kernel D2H > 0 with localized sources → recommend targeted lift sprint
- **NO-GO**: cannot determine; need deeper Nsight inspection

### Stage 4 — No regression

`pytest --collect-only`.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_d2h_v2
taskset -c 0-3 python scripts/m6b_d2h_warmed_recapture.py 2>&1 | tee .agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2/proof_warmed_run.txt
nsys stats --report cudaapi --format csv proof_warmed.nsys-rep 2>&1 | head -100 | tee .agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2/proof_warmed_trace_summary.txt
pytest --collect-only 2>&1 | tail -3 | tee .agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2/proof_no_touch.txt
```

## Risks

- Nsight overhead may inflate timings; capture both profiled and unprofiled times.

## Handoff Requirements

When verdict committed: stop. Manager folds into the next dispatch.

Time budget: **30-60 min**.
