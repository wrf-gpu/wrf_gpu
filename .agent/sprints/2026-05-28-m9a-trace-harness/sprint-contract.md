# Sprint Contract — M9.A: Operational-Mode Trace Harness + Dycore Savepoint Extension to 1 000 Steps

**Sprint ID**: `2026-05-28-m9a-trace-harness`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m9a-trace-harness`
**Worktree**: `/tmp/wrf_gpu2_m9a`
**Wall-time**: 4-6 h
**GPU usage**: YES — for the 1000-step coupled run
**Verifier**: Opus 4.7 (after worker reports DONE)

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥ 15-case seasonal ensemble; ≥ 10× speedup preserved.

## Objective

Two deliverables:
1. **Trace harness** for `_physics_boundary_step` that logs every intermediate field at every operator boundary on both the JAX side and the WRF Fortran side, in a structured comparable format.
2. **Dycore savepoint depth extension** from 100 to 1 000 coupled steps, producing a `savepoint_parity_1000.json` proof object that establishes (or breaks) the dycore-correctness ratchet.

These are the diagnostic inputs to **M9.B**, which produces the final `divergence_map.json` from them.

## Required inputs

1. `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md` (binding goal)
2. `src/gpuwrf/runtime/operational_mode.py` — full file (the trace insertion point is `_physics_boundary_step`)
3. `src/gpuwrf/coupling/physics_couplers.py` — full file
4. `scripts/m6b6_coupled_step_compare.py` — existing 100-step comparator (the pattern to extend)
5. `tests/test_m6b6_coupled_step_parity.py` — current parity test
6. `.agent/sprints/2026-05-27-testing-plan-execution-redo/savepoint_parity_deep*.json` — existing deep-savepoint structure
7. WRF Fortran reference traces — under `Gen2/wrf_savepoint_dumps/` or equivalent (find with `find /mnt -path '*savepoint*' 2>/dev/null | head`)

## Acceptance

### AC1 — `scripts/operational_trace_compare.py`

A new script that:
- Loads a WRF Fortran reference trace (one Canary 20260521 IC, 1 h forecast horizon = ~360 dt steps).
- Runs the same IC through `_physics_boundary_step` on the JAX side, capturing every intermediate field at every operator boundary (state-before, state-after for each of: dycore RK3 outer, dycore acoustic substep, microphysics, surface, PBL, radiation, lateral BC).
- Emits side-by-side per-operator per-field difference statistics (max abs diff, RMSE, location of max diff).

The script is `taskset -c 0-3 python scripts/operational_trace_compare.py --case 20260521 --horizon-steps 360 --output proofs/m9/operational_trace_360steps.json` and runs in ≤ 20 min wall time.

### AC2 — `proofs/m9/operational_trace_360steps.json`

Output of the harness on Canary 20260521 IC. Schema:
```json
{
  "trace_version": "1.0",
  "case": "20260521",
  "horizon_steps": 360,
  "dt_seconds": 10,
  "commit": "<HEAD>",
  "operators": [
    {
      "step": 0,
      "operator": "dycore_rk3_outer",
      "fields": {
        "u": { "max_abs_diff": <v>, "rmse": <v>, "argmax_diff_idx": [i,j,k] },
        "v": { ... }, "theta": { ... }, "qv": { ... }, ...
      }
    },
    ...
  ],
  "first_divergence": {
    "step": <n>, "operator": "<name>", "field": "<name>",
    "max_abs_diff": <v>, "rel_diff": <v>
  }
}
```

### AC3 — `scripts/m6b6_coupled_step_compare_1000.py`

Extension of the existing 100-step comparator to 1 000 coupled steps. Reuses the dycore reference trace (NOT the operational trace — pure dycore). Runs in ≤ 30 min wall time.

### AC4 — `proofs/m9/savepoint_parity_1000.json`

Output schema:
```json
{
  "depth": 1000,
  "status": "PASS" | "FAIL",
  "first_divergence_step": <n or null>,
  "max_abs_diff_at_final_step": <value>,
  "per_field_summary": { "u": <stats>, "v": <stats>, "theta": <stats>, ... },
  "wall_clock_seconds": <n>,
  "commit": "<HEAD>"
}
```

### AC5 — `.agent/sprints/2026-05-28-m9a-trace-harness/worker-report.md`

Standard format. **Honest verdict mandatory** — if the dycore breaks at step 250 (or anywhere before 1 000), report `M9A_PARTIAL` with the divergence step + magnitude. Do NOT modify code to hide the divergence.

### AC6 — Existing test regressions

`taskset -c 0-3 pytest -q tests/test_m6b6_coupled_step_parity.py` MUST PASS unchanged. If this regresses, abort and report.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3` on every command.
2. **GPU usage**: ALLOWED (this sprint needs the 1000-step coupled run on GPU). One GPU instance only — do NOT spawn parallel GPU runs.
3. **Files writable**: `scripts/operational_trace_compare.py`, `scripts/m6b6_coupled_step_compare_1000.py`, `proofs/m9/**`, `.agent/sprints/2026-05-28-m9a-trace-harness/**`.
4. **Files NOT writable**: `src/**` (no model-code changes — this is diagnostic), governance, public repo.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **WRF Fortran reference trace** — locate it; if absent, the worker MUST note exactly what's missing and how to generate it, then report `M9A_PARTIAL` with a clear blocker.
8. **Auto-notify on exit**: dispatcher sends `tmux send-keys -t 1 "AGENT REPORT: m9a-trace-harness DONE exit=$?" Enter`.
9. **End with verdict**: `M9A_COMPLETE / M9A_PARTIAL / M9A_BLOCKED` + one-line summary.
