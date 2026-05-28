# Sprint Contract — Plan Critic Round 1 (after M8 + M9 close)

**Sprint ID**: `2026-05-28-m9-plan-critic`
**Role**: ADVERSARIAL PLAN CRITIC
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m9-plan-critic`
**Worktree**: `/tmp/wrf_gpu2_m9_critic`
**Wall-time**: 60-90 min
**GPU usage**: NONE
**Dispatched**: 2026-05-28 after M8 close + M9 viability verdict

## Cadence trigger

Per memory [[feedback-plan-reflection-cadence]], after every 2 closed milestones the manager dispatches a codex plan-critic in parallel with manager-side reflection. M8 + M9 = 2 closed. This is the first such cadence dispatch under the reset.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on **≥30-case** seasonal ensemble (amended from ≥15 per ADR-029 power analysis); ≥10× speedup preserved.

## Required inputs

1. `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md` — current plan (as amended)
2. `.agent/decisions/ADR-028-PROJECT-RESET-2026-05-28.md` — reset ADR
3. `.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md` — power analysis source
4. `.agent/sprints/2026-05-28-m8a-manifest-stats/current_state_manifest.json` — frozen evidence
5. `.agent/sprints/2026-05-28-m8a-manifest-stats/proof_index.json` — gate registry
6. `.agent/sprints/2026-05-28-m8a-manifest-stats/m8-verifier-report.md` — Opus M8 verification
7. `proofs/m9/operational_trace_hourly.json` — 24h GPU-vs-WRF divergence across 16 fields
8. `proofs/m9/divergence_map.json` — manager-written M9 viability artefact (read whichever lands first)
9. `.agent/sprints/2026-05-28-m9a-trace-harness/worker-report.md` — codex M9.A report
10. `tests/savepoint/` — current savepoint scaffold

## Objective

Adversarial critique of the plan-as-it-stands now that M8 has frozen evidence and M9 has produced an empirical divergence map. Specifically:

## Acceptance — answer EACH explicitly

- **PC1**: Now that M9 shows hour-1 divergence across **most** operational fields (not a single localised defect), is M11-M14 still the right Phase B structure? Should Phase B be replaced or augmented by a **theta-reference-state-convention triage sprint** before M11 (cheap to identify, may explain a large fraction of the divergence)?

- **PC2**: M9 measured TSK max_abs = 0.0 (perfect bitwise data-replay match). Does this confirm M16 (prognostic Noah-MP) is dispensable for the binding-goal path? If we can ship v0.1.0 with hourly data-replay land surface and still pass TOST on T2/U10/V10 across 30 cases, M16 saves 8-14 weeks. What's the strongest argument for keeping M16 anyway?

- **PC3**: SWDOWN max 1122 W/m² exceeds the solar constant (~1361 W/m² at TOA, ~1000 W/m² at surface). PBLH max 986 m and HFX max 4105 W/m² are both physically impossible. Are these **comparison-methodology artifacts** (e.g. unit mismatch, vertical-level misalignment, NaN handling) or **real model bugs**? Propose specific tests to disambiguate.

- **PC4**: theta avg RMSE 75 K, max 345 K. Strongly suspect WRF wrfout stores perturbation T = θ - 300 while GPU stores absolute θ. If true: a one-line conversion in the comparator removes ~75% of the apparent divergence. Is the M9 trace deliverable trustworthy as-is, or does it need a comparator audit before being used to drive M11-M14?

- **PC5**: 1000-step dycore parity (savepoint_parity_1000.json) is BLOCKED — codex sandbox blocks JAX CUDA init even with `danger-full-access`. The 24h operational pipeline DID use GPU, so the block is specific to one script's initialization path. Is this a real risk to the project, or is it sufficient to run that script from a manager-environment shell once and call it good?

- **PC6**: The plan's M11-M14 estimates 8-12 weeks with partial parallelism. Given the M9 finding (multi-source multi-field divergence at hour 1), is that timeline still credible? Re-estimate.

- **PC7**: Should we **skip M10 (static-field/LU_INDEX parity)** as a standalone sprint and fold it into M14 (lateral BC + state-completeness)? Or is it cheap enough (1-2 wk) to keep as its own milestone? M9 confirms the static-field defect (LU_INDEX max 14-cat delta) — at minimum the fix is in scope.

- **PC8**: What is the single most likely-to-be-wrong assumption in the amended plan now?

- **PC9**: Strongest objection in one sentence.

- **PC10**: If you could reorder, drop, or add ONE milestone for maximum probability of TOST equivalence within 6 calendar months, what would it be?

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **No GPU runtime.**
3. **Files writable**: `.agent/sprints/2026-05-28-m9-plan-critic/**` only.
4. **No remote push.**
5. **Manager repo ONLY**.
6. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m9-plan-critic DONE exit=$?" Enter`.
7. **End with verdict**: `CRITIQUE_COMPLETE` + one-line summary.
