# ADR-030 — M16 Prognostic Noah-MP CONDITIONAL for v0.1.0

**Status**: PROPOSED (autonomous manager decision under principal directive 2026-05-28; principal review at next interaction)
**Date**: 2026-05-28
**Decision owner**: Manager (Claude Opus 4.7)
**Supersedes**: M16 row in PROJECT-RESET-PLAN-FINAL.md as "mandatory critical-path 8-14 wk milestone"
**Inputs**:
- `.agent/sprints/2026-05-28-m9-plan-critic/critique.md` PC2 + PC10
- `proofs/m9/divergence_map.json` — TSK max_abs_diff = 0.0 (data-replay bitwise)
- `proofs/m9/operational_trace_hourly.json` — 24-h × 16-field divergence trace

## Context

The reset plan made M16 (full prognostic Noah-MP on GPU, 8-14 weeks) a mandatory critical-path milestone for v0.1.0 release. M9 diagnostic produced a hard finding: on Canary 20260521 d02, hourly TSK data-replay achieves **bitwise match** with WRF (max_abs_diff = 0.0 K, RMSE 0.0 K across all 24 hours). The plan-critic (PC2, PC10) argued this evidence does not prove M16 is *unnecessary* — but it does prove the **replay mechanism is bitwise exact** for at least skin-temperature state, which is the largest contributor to surface-flux + radiation coupling. If replay-driven land state lets the model pass TOST on T2/U10/V10 across the ≥30-case M20 ensemble, then carrying an 8-14 wk Noah-MP port on the v0.1.0 critical path is not justified.

Counter-argument: the project's binding goal includes "professional-forkable GPU-native regional NWP". A v0.1.0 forecast that depends on hourly *CPU-WRF-derived* land state inputs is not a fully self-contained forecast — it borrows from CPU WRF for the surface boundary. That undermines the "GPU-native" claim and the forkable claim (forkers without a parallel CPU WRF run cannot reproduce).

## Decision

**M16 is CONDITIONAL for v0.1.0** with the following decision tree:

| Condition (evaluated at M19+M20 close) | M16 disposition |
|---|---|
| Replay-driven model passes TOST on T2/U10/V10 across ≥30-case M20 ensemble | **Defer M16 to v0.2.0**. v0.1.0 ships with documented replay-land-surface dependency. The release notes name CPU WRF as the land-state oracle. The paper section "limitations" carries this caveat explicitly. |
| Replay-driven model fails TOST on T2/U10/V10 in ways attributable to land-state evolution (skin temp drift, soil moisture decoupling, snow/canopy mismatch) | **M16 stays on v0.1.0 critical path**. The 8-14 wk port executes. |
| M20 ensemble cannot produce ≥30 cases with hourly land-replay inputs available (data gaps) | **M16 stays on v0.1.0 critical path** — replay isn't operationally available. |
| Principal explicitly requires v0.1.0 to be forecast-independent-of-hourly-CPU-replay | **M16 stays on v0.1.0 critical path** — principal release criteria override. |

The disposition is decided at M19 close (single-case skill recovery) and re-confirmed at M20 close (ensemble corpus build). If at M19 the single-case skill with replay land is already passing the ±20 % M19 gate, M20 proceeds in parallel with M21 TOST instead of waiting for an M16 port.

## Timeline impact

- **Best case (M16 deferred)**: 32-45 wk → **18-29 wk** for v0.1.0 (drops 14 wk worst-case). Target slips Q1-Q2 2027 → Q3-Q4 2026. Closer to the principal's original Sept-Oct 2026 target.
- **Worst case (M16 stays)**: timeline unchanged at 32-45 wk.

## Acceptance criteria for M16-deferred v0.1.0 release

If M16 is deferred, the release must include:
1. `INSTALL.md` updated to require **CPU WRF land-state outputs** as part of the IC/BC bundle for any new forecast.
2. `LAND-STATE-DEPENDENCY.md` (new) — single-page document explaining what the replay does, why it's not yet prognostic, what the v0.2.0 plan is.
3. Paper limitation section explicitly names this as a known gap, with measured skill impact (from M9.C onwards) reported.
4. `tests/savepoint/test_replay_land_state.py` — proves the replay path is bitwise reproducible from given inputs.
5. v0.1.0 release notes flag the conditional Noah-MP roadmap.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Replay-dependent forecast is judged "not really a GPU port" by external reviewers / community | Medium | High to project credibility | Explicit honesty in release + paper; M16 stays in v0.2.0 |
| Single-case skill passes with replay but ensemble TOST fails because replay quality varies across cases | Medium | High to schedule | M20 corpus build runs in parallel with M16 prep so M16 can dispatch quickly if needed |
| M16 keeps being deferred indefinitely | Medium | Medium | ADR-030 commits to M16 in v0.2.0 unconditionally |

## Reversibility

If at M19 the M19 gate fails with skin-temperature drift as the dominant defect, ADR-030 is amended to M16-mandatory and the timeline reverts to 32-45 wk.

## Acceptance

Manager applies this decision autonomously per principal directive 2026-05-28 ("do whatever it takes ... I will be off the computer for the rest of the day"). Principal review at next interaction.
