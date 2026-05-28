# Plan Critic Round 2

Role: ADVERSARIAL PLAN CRITIC
Sprint: `2026-05-28-plan-critic-round-2`
Date: 2026-05-28

## Findings

1. **BLOCKER: silent operators invalidate the current Phase B closure logic.** The harness found `microphysics_thompson: NOISY_ZERO` on 6/7 moisture fields and `dycore_rk3: NOISY_ZERO` on `p_perturbation` and `ph_perturbation`. M12 and M13 partials then produced almost no T2 improvement. The plan cannot keep dispatching subsystem parity work as if every operator in the coupled graph is active.
2. **BLOCKER: the oracle suite is a smoke test, not coverage.** `aggregate_SMOKE.json` has 1 executed case, 16 field tests, 2 pass / 14 fail, and no season/domain/lead/regime matrix. It is useful as a regression tripwire, but it does not satisfy the "99% coverage" commandment or the reset plan's seasonal equivalence direction.
3. **HIGH: performance margin is already being consumed before correctness is restored.** Corrected 24h speedup went from 22.26x legacy corrected to 21.92x after M10 to 14.61x after M12; M13's full pipeline was blocked by XLA autotune OOM. Waiting until M22 for performance truth is too late.

## PC2.1 - Microphysics silent failure

M17 promotion was the right move. In fact, if the harness had existed before M11/M12/M13 dispatch, a pre-fix harness smoke should have run before all three. The M12 result proves why: HFX formula work changed almost nothing (`HFX` RMSE drop 0.21%, T2 RMSE still 10.802 K), which is exactly what happens when upstream moisture/latent heating is silently inactive.

I would not diagnose other silent failures before M17 unless they block the same files. The microphysics finding is narrow, confirmed, and directly connected to `qv`, latent heat, LH, and T2. But the dycore `p_perturbation` / `ph_perturbation` flatline must be treated as the next blocker, not background noise. A radiation-specific harness run is also needed because the 3-step smoke intentionally did not fire RRTMG (`radiation_cadence_steps=60`, `steps_total=3`).

Decision: keep M17 promoted; require an immediate post-M17 harness run, then dispatch dycore p/ph triage before accepting Phase B closure.

## PC2.2 - M12 did not move T2

`M12_PARTIAL` was the only honest close. AC1 and AC5 failed: hour-1 HFX/LH parity failed, HFX 24h RMSE barely changed (`980.137 -> 978.034 W m-2`), and T2 stayed at `10.802 K`.

Do not keep iterating on surface flux now. Further HFX tuning before M17 and dycore p/ph diagnosis will likely fit a downstream diagnostic to a wrong lower-atmosphere state. The right next M12 action is a re-diagnosis after M17 and M11.1: rerun the harness, then run a narrow M12.1 only if `surface_layer -> mynn_pbl` is still active but wrong against a WRF surface-layer/bottom-BC oracle.

## PC2.3 - Diagnostic-driven dispatch

Yes: every future sprint touching dycore, physics, coupling, boundary conditions, runtime operator order, or wrfout diagnostics should require both:

1. pre-fix `diagnostic_report.json`;
2. post-fix `diagnostic_report.json`.

The run must be configured to exercise the touched operator. A 3-step smoke is enough for dycore, surface layer, MYNN, lateral boundary, and microphysics wiring; it is not enough for radiation cadence, 24h diurnal behavior, or boundary relaxation over longer leads.

Cost-benefit: the smoke proof cost about 95 s wall time in the recorded artifact, and even a 1h targeted harness is cheap compared with losing days to fixes against inactive operators. The extra cost is GPU scheduling and occasional OOM risk; the benefit is avoiding false closes. This should be a merge gate, not a nice-to-have.

## PC2.4 - Oracle suite completeness

Scope the extension as a coverage matrix first, then fill it with available or blocked cases. The current suite should remain the smoke baseline, but a new oracle-completeness sprint must produce `oracle_regime_matrix.yaml` with explicit cells, status (`AVAILABLE`, `BLOCKED`, `MISSING_RUNNER`, `MISSING_ORACLE`), generation command, expected variables, and smoke/full mode.

Missing regime cells in priority order:

1. **Available-but-not-executed nest coverage:** run the existing `canary_20260521_24h_d03` case to cover 1 km/nested behavior.
2. **Lead-time coverage:** generate/repair `canary_20260521_72h_d02`; 24h-only cannot validate 24-72h skill.
3. **Radiation diurnal cells:** sunrise, midday, sunset, and night, with RRTMG firing multiple times.
4. **Moisture/microphysics cells:** dry clear sky, nonprecipitating cloud, precipitating/orographic cloud, ice-phase/hydrometeor-active cases.
5. **Surface-regime cells:** ocean, coast, lowland land, high-elevation land, steep terrain, and land/sea transition cells.
6. **Wind-regime cells:** trade-wind acceleration, lee/wake flow, weak-flow/stagnation, and high-wind events.
7. **Seasonal cells:** winter, spring, summer, autumn across L2 and L3; this is required before TOST claims are meaningful.
8. **Boundary/nesting cells:** d02 lateral-boundary-dominated strips, d03 nest interior, and boundary-relaxation stress cases.
9. **Idealized physics/dycore cells:** warm bubble, density current, mountain wave, acoustic/CFL ladder, and conservation-budget cases.
10. **Failure-mode regression cells:** the known M9/M12/M13/M17 defects as locked regression cases, so "fixed once" cannot silently recur.

## PC2.5 - Phase B re-estimation

Round 1's 12-18 week estimate was honest for the evidence then, but it is now too optimistic as a planning median. It assumed the remaining defects were wrong active operators, not a class of wired-but-silent operators.

From 2026-05-28, realistic Phase B close is:

- best credible case: 2026-09-17, about 16 weeks, if M17 and M11.1 are simple and M14 does not expand;
- planning date: 2026-10-15, about 20 weeks;
- pessimistic-but-plausible: 2026-11-12, about 24 weeks, if radiation interface work, boundary completeness, and performance recovery all need follow-on sprints.

The plan should stop presenting Phase B as 12-18 weeks without a caveat that this was pre-harness and pre-M12/M13 partial evidence.

## PC2.6 - Dycore p/ph perturbation flatlines

This needs a new M11.1 unless the active M11 contract is explicitly amended before merge. It is adjacent to M11, but not safely inside the current acceptance criteria: M11 is theta limiter + guard accounting, while the harness found pressure/geopotential perturbation activity failure.

Severity: high, Phase-B-blocking. If `p_perturbation` and `ph_perturbation` are expected prognostic/update fields, zero dycore delta can explain PSFC, wind, theta, and boundary-coupling divergence. If they are intended diagnostic/derived fields, the harness expectation is wrong and must be corrected. Either way, Phase B cannot close until this is proven with a WRF savepoint/operator oracle or a documented state-contract explanation.

## PC2.7 - Performance trajectory

The speed floor can realistically break at M13/M14, not M22. M12 is already `14.61x`; with the fixed CPU denominator, the 10x floor corresponds to a 24h GPU wall of about 1630 s. M12 used about 1116 s, leaving only about 514 s of headroom. Another roughly 46% wall-time increase breaks the floor, and M13 already hit an XLA autotune OOM on a full operational run.

Mitigation:

- add a post-sprint speed/D2H mini-gate for every correctness sprint after M17, with a warning tripwire below 12x;
- dispatch a performance recert sprint before M19 if speed drops below 12x or any full-run OOM persists;
- keep diagnostic instrumentation compile-time dead in production paths;
- profile RRTMG and microphysics after correctness, especially f64 fusion/memory blowups;
- use ADR-007 precision authorizations and operator-level fusion only after correctness artifacts exist;
- do not wait for M22 to discover the final code cannot clear 10x.

## PC2.8 - Plan amendment proposed

Top 3 textual amendments to apply now:

1. **Strengthen the diagnostic-harness section.** Replace "attach a fresh diagnostic_report.json" with "attach pre-fix and post-fix diagnostic_report.json, configured so every touched operator is ACTIVE unless intentionally inactive; unexplained MISSING/NOISY_ZERO blocks COMPLETE."
2. **Restructure Phase B around silent-coupling triage.** Move M17 from Phase E into Phase B, add M11.1 for dycore `p_perturbation`/`ph_perturbation` activity, and state that M12/M13/M14 cannot close until post-M17/M11.1 harness reports remove or explain all silent fields.
3. **Insert an oracle-regime-matrix sprint before skill recovery.** The oracle suite must explicitly cover domain, lead time, season, radiation phase, moisture/microphysics, surface, wind, boundary/nesting, idealized, and known-failure regression cells. The current one-case smoke suite is a regression canary, not coverage.

## PC2.9 - Strongest objection in one sentence

The plan is still at risk of spending the next five milestones tuning physics symptoms while the coupled operator graph contains silent identity paths that make every downstream parity result ambiguous.

## PC2.10 - Single highest-impact action

If M17 were not already dispatched, M17 would be the single highest-impact sprint. Since it is already running, the next single dispatch should be **M11.1 dycore pressure/geopotential perturbation activity triage**.

Why: M17 fixes the confirmed microphysics identity path, but `p_perturbation` and `ph_perturbation` flatlines are the next silent failure class and can contaminate theta, PSFC, winds, surface fluxes, and boundaries. Do not dispatch another broad physics parity sprint until M11.1 either makes those fields active or proves the harness/state-contract expectation is wrong.

## Handoff

- objective: adversarially critique the reset plan after Phase A close, M11/M12/M13/M17 partial/in-flight evidence, diagnostic-harness smoke, and oracle-suite smoke; answer PC2.1-PC2.10.
- files changed: `.agent/sprints/2026-05-28-plan-critic-round-2/critique.md`.
- commands run: `pwd`; `sed` reads of `PROJECT_CONSTITUTION.md`, `AGENTS.md`, sprint contract, local `conducting-blind-review` and `reporting-to-human` skills; `git status --short --branch`; `find`/`rg` discovery under `.agent`, `proofs`, and `tests`; `taskset -c 0-3` reads of required plan, ADRs, proof objects, oracle manifests, tolerances, and round-1 critique; `taskset -c 0-3` reads of M11/M12/M13/M17/harness supporting contracts/reports; `taskset -c 0-3 jq` summaries of M10/M12/M13 speed and skill artifacts.
- proof objects produced: this critique report.
- unresolved risks: no GPU runtime was used; no independent rerun of harness/oracle/speed artifacts was attempted; M11 is in flight so M11.1 recommendation is based on the harness smoke artifact, not on M11 final output; M13 full-run evidence is partial because the worker reported XLA autotune OOM.
- next decision needed: decide whether to freeze new Phase B dispatches until M17 and M11.1 post-fix harness reports remove or explain all `NOISY_ZERO` fields.

CRITIQUE_COMPLETE - Promote harness-gated silent-coupling triage above further physics tuning; Phase B is now a 16-24 week risk, not a 12-18 week plan.
