# Sprint Contract — M8.A: Evidence Manifest + Proof Registry + Statistics-Design ADR

**Sprint ID**: `2026-05-28-m8a-manifest-stats`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m8a-manifest-stats`
**Worktree**: `/tmp/wrf_gpu2_m8a`
**Wall-time**: 2-4 h
**GPU usage**: NONE
**Verifier**: Opus 4.7 (separate window after worker reports DONE)

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥ 15-case seasonal ensemble; ≥ 10× speedup preserved.

## Objective

Produce three deliverables that anchor every future sprint-close gate:
1. `current_state_manifest.json` — authoritative snapshot of every current numerical claim (skill, perf, D2H, divergence) cited to file:line.
2. `proof_index.json` — registry of every M8-M23 sprint-close gate with schema/version/command fields.
3. `ADR-029-STATISTICS-DESIGN-TOST.md` — predeclared TOST margins, paired-design spec, power-analysis approach.

## Required inputs (read in order)

1. `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md`, `.agent/decisions/ADR-028-PROJECT-RESET-2026-05-28.md`
2. `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md`
3. `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json`
4. `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_speedup.json`
5. `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md`
6. `.agent/sprints/2026-05-28-project-reset-critic/critique.md` (INV-1..11 expansion definitions)
7. `.agent/sprints/2026-05-28-project-reset-blinded/plan.md` (B5 invariant spec, B9 sprint sizing)
8. `src/gpuwrf/runtime/operational_mode.py` (operational entry path)
9. `proofs/` if it exists (the registry is authoritative)
10. `.agent/decisions/ADR-027*.md` (D2H invariant)

## Acceptance

### AC1 — `.agent/sprints/2026-05-28-m8a-manifest-stats/current_state_manifest.json`

Schema:
```json
{
  "manifest_version": "1.0",
  "manifest_commit": "<current HEAD>",
  "generated_at": "<ISO8601 UTC>",
  "skill": {
    "iter2_5day_canary": {
      "T2_K": { "gpu_rmse": <value>, "cpu_rmse": <value>, "delta_pct": <value>, "source": "post_iter2_skill_diff.json:<line>" },
      "U10_ms": { ... }, "V10_ms": { ... }
    },
    "iter1_canary": { ... }
  },
  "performance": {
    "corrected_22x_apples_to_apples": { "value": 22.2558, "source": "post_iter2_speedup.json:<line>", "denominator": "<spec>" },
    "rejected_156x_claim": { "value": 156.82, "rejected": true, "source": "MILESTONE-M7-CLOSEOUT-AMENDMENT.md:<line>" }
  },
  "d2h_invariant": { "ADR_027_status": "PROPOSED", "current_measurement": <value or null>, "source": "<path>" },
  "savepoint_parity": {
    "B6_100_step_status": "PASS", "depth": 100,
    "extension_to_1000": { "status": "NOT_RUN", "blocker": "<spec>" }
  },
  "static_fields": {
    "LU_INDEX_divergence": { "max_category_delta": 14, "source": "first_hour_diff.json:<line>" },
    "HGT_LANDMASK_status": "BITWISE_MATCH"
  },
  "validation_corpus": {
    "current_cases_complete_24h": 2, "target_15_case_status": "BLOCKED",
    "source": "canary_multiday_skill.json:<line>"
  },
  "source_rca_divergences": [
    { "rca_claim": "<text>", "current_source": "<file:line>", "diff": "<text>" }
  ],
  "missing_directories": ["tests/savepoint/", "<others>"],
  "verdict": "EVIDENCE_FROZEN"
}
```

Every numeric value cited to `file:line`. Failure to cite = sprint reject.

### AC2 — `.agent/sprints/2026-05-28-m8a-manifest-stats/proof_index.json`

Schema:
```json
{
  "registry_version": "1.0",
  "gates": {
    "M8": [ { "id": "M8.A.1", "name": "current_state_manifest", "schema_ref": "<path>", "command": "<bash>", "owner": "manager" }, ... ],
    "M9": [ { "id": "M9.A.1", "name": "divergence_map", "schema_ref": "<path>", "command": "<bash>", "owner": "codex" }, ... ],
    ...
    "M23": [ ... ]
  },
  "invariants": {
    "INV-1": { "name": "D2H zero", "schema": "<nsys spec>", "applies_from": "M8" },
    ...
    "INV-11": { "name": "Evaluation sufficiency", "schema": "<spec>", "applies_from": "M20" }
  }
}
```

Every M8-M23 milestone's gates fully enumerated (no `<TODO>` entries; if a gate is too early to specify, write `"specified_at": "M<n>"` instead).

### AC3 — `.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md`

Must contain:
- Predeclared TOST margins per variable (T2, U10, V10) with WRF-CPU-natural-variance justification cited to operational-meteorology literature OR the AEMET/operational benchmarks present locally.
- Paired-design specification: pairing key (case × valid-time × station), missing-data handling, season stratification.
- Power analysis: minimum-detectable effect at n=15 vs n=30 cases per metric; required sample size to detect a 10% RMSE difference at α=0.05, β=0.20.
- Reviewer requirement: M20 + M21 both require a statistics reviewer (Opus or Gemini agy).
- Status: `PROPOSED`.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3` on every command.
2. **No GPU runtime.** Pure read + write.
3. **No code changes** outside `.agent/sprints/2026-05-28-m8a-manifest-stats/` and `.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md`.
4. **Every claim cites file:line.** No vague numbers.
5. **No remote push.** Local commit on `worker/gpt/m8a-manifest-stats`.
6. **Manager repo ONLY** — do not touch `/home/enric/src/wrf_gpu/`.
7. **Auto-notify on exit**: dispatcher sends `tmux send-keys -t 1 "AGENT REPORT: m8a-manifest-stats DONE exit=$?" Enter`.
8. **End with verdict**: `M8A_COMPLETE / M8A_PARTIAL` + one-line summary.
