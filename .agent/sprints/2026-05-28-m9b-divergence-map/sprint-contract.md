# Sprint Contract — M9.B: Divergence Map + Operational-Variable Comparison + LU_INDEX Audit

**Sprint ID**: `2026-05-28-m9b-divergence-map`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m9b-divergence-map`
**Worktree**: `/tmp/wrf_gpu2_m9b`
**Wall-time**: 3-5 h
**GPU usage**: YES — for 1 h side-by-side run, comparison only
**Dispatched AFTER**: M9.A reports `DONE` (depends on the trace harness)

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥ 15-case seasonal ensemble; ≥ 10× speedup preserved.

## Objective

Consolidate the M9.A trace into a single **`divergence_map.json`** that is the M9 closure deliverable. Add a side-by-side operational-variable comparison (SWDOWN/GLW/HFX/LH/PBLH/TSK/T2/U10/V10/PSFC) at lead 1 h and an LU_INDEX/static-field audit (folds in M10 scope).

This is **the diagnostic that determines viability**. The principal will read this and decide whether the remaining defects are localised + fixable or whether the project is a dead end.

## Required inputs

1. `proofs/m9/operational_trace_360steps.json` (from M9.A)
2. `proofs/m9/savepoint_parity_1000.json` (from M9.A)
3. `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md` (existing RCA — what we already suspect)
4. `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json` (current skill gap)
5. `scripts/operational_trace_compare.py` (extend if needed for variable comparison)
6. `src/gpuwrf/contracts/state.py` (LU_INDEX leaf status)
7. WRF Fortran lead-1h reference outputs for Canary 20260521 (find with `find Gen2 -name "wrfout*" -newer Gen2/runs/wrf_l3 | head`)

## Acceptance

### AC1 — `proofs/m9/divergence_map.json` (the M9 closure artefact)

Schema:
```json
{
  "version": "1.0",
  "commit": "<HEAD>",
  "case": "20260521",
  "summary": {
    "dycore_parity_depth": 1000,
    "dycore_status": "PASS" | "DIVERGES_AT_STEP_<n>",
    "operational_first_divergence": {
      "step": <n>, "operator": "<name>", "field": "<name>",
      "magnitude": <v>, "rel_to_field_scale": <v>
    },
    "static_field_status": {
      "HGT": "BITWISE_MATCH" | "<delta>",
      "LANDMASK": "<status>",
      "LU_INDEX": "MAX_CATEGORY_DELTA_<n>",
      "roughness_m": "<status>",
      "soil_category": "<status>"
    },
    "operational_variable_status_lead_1h": {
      "SWDOWN": { "max_abs_diff": <v>, "rmse": <v>, "interpretation": "<text>" },
      "GLW": { ... }, "HFX": { ... }, "LH": { ... }, "PBLH": { ... },
      "TSK": { ... }, "T2": { ... }, "U10": { ... }, "V10": { ... }, "PSFC": { ... }
    }
  },
  "diagnosis": {
    "primary_defect": { "operator": "<name>", "field": "<name>", "evidence": ["<list>"], "confidence": "<low|medium|high>" },
    "secondary_defects": [ ... ],
    "static_field_defects": [ { "field": "LU_INDEX", "evidence": "<text>" }, ... ],
    "is_localised_to_known_operators": true | false,
    "estimated_repair_milestones": ["M11", "M12", "M13", "M14"]
  },
  "viability_verdict": "VIABLE" | "VIABLE_WITH_NOAH_MP_DEFERRAL" | "DEAD_END",
  "verdict_evidence": "<text — the single paragraph explaining the verdict>"
}
```

The `viability_verdict` is the headline. It must be earned by the evidence above; the worker MUST NOT default to `VIABLE` if the divergence pattern is diffuse + multi-source.

### AC2 — `scripts/operational_variable_compare.py`

A new script (or extension of M9.A's `operational_trace_compare.py`) that produces the lead-1h variable comparison block in AC1. Runs in ≤ 15 min.

### AC3 — `scripts/static_field_audit.py`

A new script that produces the `static_field_status` block in AC1 by comparing GPU/JAX initial-state static fields against WRF wrfinput files. Runs in ≤ 5 min, no GPU.

### AC4 — `.agent/sprints/2026-05-28-m9b-divergence-map/worker-report.md`

Standard format. **Honest verdict mandatory** — write `M9B_PARTIAL` if you cannot reach a confident viability verdict; write `M9B_DEAD_END` if the divergence pattern looks unrepairable in the remaining scope.

### AC5 — Existing tests regression-free

`taskset -c 0-3 pytest -q tests/test_m6b6_coupled_step_parity.py tests/savepoint/` MUST PASS.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: ALLOWED for ≤ 1 h forecast + comparison. One GPU instance only.
3. **Files writable**: `scripts/operational_variable_compare.py`, `scripts/static_field_audit.py`, `proofs/m9/**`, `.agent/sprints/2026-05-28-m9b-divergence-map/**`.
4. **Files NOT writable**: `src/**`, governance, public repo.
5. **Diagnostic only.** No fixes. The point of M9 is to identify the divergence; M11-M14 fix.
6. **No remote push.**
7. **Manager repo ONLY**.
8. **Auto-notify on exit**: dispatcher sends `tmux send-keys -t 1 "AGENT REPORT: m9b-divergence-map DONE exit=$?" Enter`.
9. **End with verdict**: `M9B_COMPLETE / M9B_PARTIAL / M9B_DEAD_END` + one-line summary.
