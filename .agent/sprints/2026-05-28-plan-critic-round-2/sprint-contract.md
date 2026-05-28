# Sprint Contract — Plan Critic Round 2 (after Phase A close + M11/M12/M13 partials + harness + Oracle)

**Sprint ID**: `2026-05-28-plan-critic-round-2`
**Role**: ADVERSARIAL PLAN CRITIC
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/plan-critic-round-2`
**Worktree**: `/tmp/wrf_gpu2_pc2`
**Wall-time**: 60-90 min
**GPU usage**: NONE
**Dispatched**: 2026-05-28 after Phase A close + M11(running)/M12(partial)/M13(partial)/M17(dispatched after harness finding)/oracle(complete)

## Cadence trigger

Per memory `[[feedback-plan-reflection-cadence]]`, the manager has now closed (or merged-as-partial) M8, M9, M9.C, M10, M12 (partial), M13 (partial), oracle, plus dispatched M17 promoted-early and the diagnostic harness landed. That is 6 closes + 2 in-flight = well past the every-2-milestones cadence. This is critic round 2.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 statistically equivalent to CPU WRF v4 under TOST at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs (read in order)

1. `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md` — current plan (twice amended)
2. `.agent/decisions/ADR-028-PROJECT-RESET-2026-05-28.md`, `.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md`, `.agent/decisions/ADR-030-M16-CONDITIONAL-FOR-V010.md`
3. `proofs/m9/divergence_map_v2.json` — M9.C ruled-out theta-convention; 12 REAL_BUGs confirmed
4. `proofs/m10/static_field_parity_after_fix.json` — LU_INDEX bitwise; perf 21.92×
5. `proofs/diagnostic_harness/diagnostic_report_smoke_3step.json` — **MICROPHYSICS SILENT FAILURE** found
6. `proofs/m12/post_m12_skill_diff.json` — T2 unchanged at 10.80K despite HFX formula fix; speedup 14.61×
7. `proofs/m13/radiation_parity_hour_1.json` + `proofs/m13/radiation_trace_24h.json` — SWDOWN hour-1 100% within tol; 24h -57.8% RMSE
8. `proofs/regression/aggregate_SMOKE.json` — Oracle suite first run: 2 PASS / 14 FAIL
9. `tests/regression/oracle_cases.yaml` + `tests/regression/tolerances.yaml`
10. `.agent/sprints/2026-05-28-m9-plan-critic/critique.md` — round 1 critic

## Acceptance — answer EACH explicitly

- **PC2.1 — Microphysics silent failure**: was the M17 promotion the right move? Should it be even earlier (before M11)? Or are there other "silent failures" the harness might catch that should be diagnosed FIRST?

- **PC2.2 — M12 didn't move T2**: was the M12 PARTIAL the right call? Should we keep iterating on surface flux now, or wait for M17 + re-diagnose?

- **PC2.3 — Diagnostic-driven dispatch**: now that the harness provides per-operator NOISY_ZERO/ACTIVE/MISSING signals, should every future sprint's contract REQUIRE a pre-fix harness run AND a post-fix harness run? What's the cost-benefit?

- **PC2.4 — Oracle suite completeness**: the suite as delivered has 1 case, 16 field tests, no regime matrix. The supreme commandment says 99% coverage. How should the completeness extension be scoped? List the missing regime cells in priority order.

- **PC2.5 — Phase B re-estimation**: with M11 still in flight, M12 PARTIAL, M13 PARTIAL, M17 just dispatched: what's the realistic Phase B close date now? Was the round-1 12-18 wk estimate honest or too optimistic given the silent-failure finding?

- **PC2.6 — Dycore p/ph_perturbation silent flatlines**: harness smoke also found `dycore_rk3: NOISY_ZERO on p_perturbation, ph_perturbation`. Is this M11 scope or does it need a new M11.1? What's the severity?

- **PC2.7 — Performance trajectory**: 22.26× → 21.92× → 14.61× across M10 → M11 → M12. Speedup is *decreasing* with each correctness fix. At what milestone does the speed floor (10×) realistically break, and what's the mitigation?

- **PC2.8 — Plan amendment proposed**: list the top 3 textual amendments to PROJECT-RESET-PLAN-FINAL.md that should be applied NOW based on M11/M12/M13/M17/harness/oracle evidence.

- **PC2.9 — Strongest objection in one sentence**: what does the manager most need to hear before the next 5 milestones dispatch?

- **PC2.10 — Single highest-impact action**: if you could only dispatch ONE next sprint, which would it be and why?

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **No GPU runtime.**
3. **Files writable**: `.agent/sprints/2026-05-28-plan-critic-round-2/**` only.
4. **No remote push.**
5. **Manager repo ONLY**.
6. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: plan-critic-2 DONE exit=$?" Enter`.
7. **End with verdict**: `CRITIQUE_COMPLETE` + one-line summary.
