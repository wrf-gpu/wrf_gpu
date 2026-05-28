# Project Reset Blinded Plan

Sprint: `2026-05-28-project-reset-blinded`
Role: BLINDED PLANNER
Branch: `worker/gpt/project-reset-blinded`

## Evidence Boundary

This plan was built from the permitted inputs in the sprint contract. I did not read `.agent/decisions/PROJECT-RESET-PLAN-DRAFT.md`, did not read `.agent/sprints/2026-05-28-project-reset-critic/`, and did not read manager-authored 2026-05-28 decision material. Contract-named inputs that are absent in this worktree are treated as reset blockers, not assumed evidence: `tests/savepoint/`, `scripts/run_canary_*.sh`, and `proofs/`. The contract also listed `src/gpuwrf/operational_mode.py` and `src/gpuwrf/state/state.py`; the present files are `src/gpuwrf/runtime/operational_mode.py` and `src/gpuwrf/contracts/state.py`.

The binding goal is strict: Canary L2/L3 24-72 h RMSE on T2, U10, and V10 must be statistically indistinguishable from CPU WRF v4 by paired t-test with p > 0.05 on at least 15 seasonal cases, while keeping at least 10x speedup versus 28-rank CPU WRF on the same workstation (`.agent/sprints/2026-05-28-project-reset-blinded/sprint-contract.md:13`). The constitution says physics correctness comes before speed, high-frequency state must stay GPU-resident, and WRF compatibility means useful variables, namelist mapping, fixtures, and validation behavior rather than a line-by-line Fortran port (`PROJECT_CONSTITUTION.md:5`, `PROJECT_CONSTITUTION.md:9-14`).

## B1 - Position Assessment

| Block | Weight | Current evidence | Block completion | Weighted contribution |
|---|---:|---|---:|---:|
| Forecast skill and statistical equivalence | 35% | Latest side-by-side skill check still fails: `all_variables_within_20pct=false`, `gpu_materially_worse_than_cpu=true`; T2 RMSE GPU 10.80 K vs CPU 2.15 K, U10 7.24 m/s vs 2.31 m/s, V10 7.62 m/s vs 2.75 m/s on one 24 h case (`post_iter2_skill_diff.json:3-7`, `:29-35`, `:60-66`, `:91-97`, `:104-110`). Binding goal requires at least 15 seasonal cases and 24-72 h leads, which is not present (`sprint-contract.md:13`). | 8% | 2.8% |
| Dycore and physics correctness against WRF | 20% | Operational loop, state, and physics couplers exist. Current source differs from the last RCA: the guard preserves finite bounded dynamics rather than always resetting theta/mu (`operational_mode.py:230-241`, `:578-586`); surface now runs before MYNN and radiation is available on cadence (`operational_mode.py:587-593`); MYNN reads stored surface fluxes and applies bottom-BC increments (`physics_couplers.py:146-154`, `:275-292`, `:306-330`). Skill is still bad after iter2, so proof is missing or another fault remains. | 30% | 6.0% |
| GPU residency, transfer invariant, and speed | 15% | Corrected speed is 50.20x apples-to-apples, not 156.82x (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:7`, `:22`, `:27-33`). D2H inside loop, restart, repeatability, 1 km VRAM headroom, and 50x speed survived the amendment (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:61-72`). ADR-027 defines the acceptance invariant as warmed `d2h_inter_kernel == 0`, with H2D inside loop forbidden (`ADR-027-d2h-invariant-clarification-PROPOSED.md:50-58`, `:91-94`). Needs recertification after correctness changes. | 85% | 12.8% |
| Operational L2/L3 3 km and 1 km pipeline | 10% | A 24 h 3 km pipeline has run, and 1 km grid fit was reported with 78% VRAM headroom (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:21`, `:61-68`). Contract-required `scripts/run_canary_*.sh` entry points were absent in this worktree, and the binding goal includes 24-72 h L2/L3 forecasts, not only one 24 h D02 case (`sprint-contract.md:13`). | 45% | 4.5% |
| Validation fixtures, savepoints, and proof objects | 10% | Station scoring exists and uses the same scaffold for CPU and GPU, 73 AEMET stations x 24 h = 1747 joined rows (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:42`, `post_iter2_skill_diff.json:117-142`, `:2112-2138`). But `tests/savepoint/` and `proofs/` were absent, and the amendment says finite station scores were incorrectly elevated to skill evidence (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:56-59`). | 25% | 2.5% |
| Boundary, nesting, and WRF-compatible data contracts | 5% | State carries WRF-shaped prognostic, surface, and boundary fields (`contracts/state.py:35-83`, `:327-351`). RCA still flags lateral boundary width handling as a real architectural defect after theta/mu is fixed (`top_3_suspects.md:155-164`). L2.1 d01 ingest was held until skill regression is understood (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:84-89`). | 35% | 1.8% |
| Governance, auditability, and forkability | 5% | Constitution, ADRs, sprint contracts, and amendment discipline exist. The amendment corrected an inflated speed claim and overclaimed milestone close (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:27-33`, `:72-77`). Central proof registry and operational entry scripts are missing from the permitted evidence set. | 55% | 2.8% |

Estimated overall completion toward the binding goal: **33%**. The number is low despite strong speed evidence because the binding goal is dominated by WRF-comparable forecast skill and seasonal statistical proof, and the latest measured skill still fails by large margins.

## B2 - Milestone Roadmap

Delta completion values are planner estimates from the 33% position above; they sum to the remaining 67 percentage points. Each milestone must produce a proof object before closure.

| ID | Milestone | Numeric proof required | Est. weeks | Delta completion | Risk |
|---|---|---|---:|---:|---|
| M0 | Evidence freeze and current-state reconciliation | `current_state_manifest.json` names the exact commit, latest skill metrics, corrected speed, D2H status, absent directories, and source/RCA divergences. It must cite the post-iter2 RMSE numbers (`post_iter2_skill_diff.json:29-35`, `:60-66`, `:91-97`) and the current source facts that changed since RCA (`operational_mode.py:578-593`, `physics_couplers.py:146-154`, `:275-330`). | 1 | +2% | Low |
| M1 | Proof registry, savepoint harness, and operational entry inventory | A committed `proof_index.json` covers 100% of sprint-close gates, every future proof has schema/version/command/commit fields, `tests/savepoint/` is either restored or replaced by named fixture tests, and all Canary run entry points are named. No model-code changes. | 1-2 | +4% | Medium |
| M2 | Dycore theta/mu and guard correctness | On a WRF fixture and the 20260521 case: zero nonfinite values over 24 h, zero unexpected guard fallbacks for valid WRF-range states, relative dry-mass drift <= 1e-6 per 24 h, and theta/mu first-hour normalized RMSE against CPU WRF savepoint <= 1e-3 or an ADR-approved tolerance. Guard behavior must be measured at `_limit_guarded_dynamics_state` (`operational_mode.py:230-241`). | 2-3 | +7% | High |
| M3 | Surface-layer to MYNN coupling proof | A 1 h column oracle shows surface-layer `theta_flux`, `qv_flux`, `tau_u`, and `tau_v` produce post-PBL column-integrated theta/qv/u/v changes within 5% of the expected flux budget. The current source passes fluxes from surface to MYNN (`physics_couplers.py:146-154`, `:275-330`); the milestone proves units, signs, and WRF compatibility. | 1-2 | +4% | Medium |
| M4 | Radiation and land-surface diurnal physics | RRTMG and surface state use time-varying solar/skin/land inputs rather than constant placeholders. Current RRTMG uses constant albedo 0.15, emissivity 0.98, and `coszen=0.50` (`physics_couplers.py:379-381`). Proof: RRTMG cadence and coszen vary correctly over 24 h, land-station T2 diurnal amplitude differs from CPU WRF by <= 1 K on the pinned case, and all radiation heating rates are finite. | 3-4 | +7% | High |
| M5 | Lateral boundary and nesting repair | Decode and apply all required boundary width/time slices. Proof: every relax-zone offset uses the intended parent/boundary field, boundary strip RMSE against decoded wrfbdy fields is <= 1e-6 relative for u/v/theta/qv/ph/mu, and interior-vs-boundary error split no longer shows boundary-dominated first-hour drift. RCA says the width-1 strip is a real defect masked by theta/mu problems (`top_3_suspects.md:155-164`). | 2-3 | +5% | High |
| M6 | WRF savepoint and conservation ladder | 100% of tiered savepoint tests pass with no unreviewed xfails. Required groups: acoustic/dycore, pressure/geopotential/mu, surface/MYNN, RRTMG, boundary replay, restart, and hourly output diagnostics. Conservation proofs include dry mass relative drift <= 1e-6 per 24 h and water budget residual <= 1% of column source/sink magnitude. | 4-5 | +8% | High |
| M7 | Single-case L2/L3 skill recovery | For 20260521 and the L2 D02 replay, GPU vs CPU side-by-side station scoring uses the same scorer and common sample mask. T2, U10, V10 RMSE and MAE must each be within 20% of CPU on at least the current 24 h sample count scale. Current post-iter2 RMSE ratios are far outside that: T2 10.80 vs 2.15, U10 7.24 vs 2.31, V10 7.62 vs 2.75 (`post_iter2_skill_diff.json:29-35`, `:60-66`, `:91-97`). | 2-3 | +6% | High |
| M8 | Operational 24-72 h L2/L3 and 1 km hardening | Produce 24 h, 48 h, and 72 h L2/L3 forecast artifacts for 3 km and 1 km target domains with restart bitwise PASS, repeatability bitwise PASS, no nonfinite outputs, and VRAM peak <= 90%. The current reported evidence covers 24 h and 1 km headroom, not the full 24-72 h binding surface (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:18-21`, `:61-68`; `sprint-contract.md:13`). | 2-3 | +5% | Medium |
| M9 | Performance and transfer recertification after correctness | Warmed Nsight proof has `d2h_inter_kernel == 0`, loop H2D == 0, and pre-kernel D2H below ADR-027 thresholds (`ADR-027-d2h-invariant-clarification-PROPOSED.md:50-58`, `:79-92`). GPU throughput remains >= 10x versus the 28-rank CPU WRF baseline, with CPU analysis commands pinned by `taskset -c 0-3` where applicable and speed parsed from raw de-duplicated timing logs. | 1-2 | +5% | Medium |
| M10 | Seasonal statistical equivalence ensemble | At least 15 cases spanning seasons, both L2 and L3 where applicable, 24-72 h leads, paired by valid station/time. For each of T2, U10, V10: paired test on RMSE deltas gives p > 0.05, effect sizes and confidence intervals are reported, and no variable has an operationally material degradation hidden by low power. This is the binding statistical gate (`sprint-contract.md:13`). | 4-6 | +12% | High |
| M11 | Forkable release and closeout governance | Release bundle has ADRs for architectural changes, public-facing claims cite current proof objects, no 156x claim remains, fresh clone instructions reproduce validation, and the final milestone close includes honest unresolved limitations. The amendment explicitly says the 156x number must not be cited and the current system is not publication-ready (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:72-77`). | 1-2 | +2% | Medium |

## B3 - Dependency Graph

M0 blocks everything because current source and RCA evidence disagree, and missing `tests/savepoint/`, `scripts/run_canary_*.sh`, and `proofs/` must be made explicit before code work.

M1 depends on M0 and blocks reliable closure of all later work. M2, M3, and M5 can start in parallel after M1 if interfaces are frozen. M4 can begin after M1, but its skill interpretation depends on M2 because the RCA says skin/radiation effects cannot be interpreted while level-1 theta is frozen or guarded incorrectly (`top_3_suspects.md:139-148`). M6 is an integration gate over M2-M5. M7 depends on M2-M6. M8 can harden non-skill pipeline mechanics after M1, but final M8 closure depends on M7 because the pipeline must not encode a wrong physics path. M9 depends on all correctness-affecting code changes and M8, because every correctness fix can change GPU runtime and transfer behavior. M10 depends on M7-M9 and on case-manifest readiness from M1. M11 can draft throughout but closes last.

Parallelizable work:

- M2 dycore/guard, M3 surface/MYNN, and M5 boundary can run in separate worktrees after M1 freezes interfaces.
- CPU WRF baseline manifesting for the 15-case ensemble can start after M1 while GPU correctness work proceeds.
- Documentation and ADR drafts can run in parallel, but cannot merge until the proof-producing sprint lands.
- Performance tooling can be prepared before M9, but official numbers wait until correctness code is stable.

## B4 - Critical Path

The single longest dependency chain is:

M0 evidence freeze -> M1 proof harness -> M2 dycore/guard correctness -> M4 radiation/land-surface diurnal physics -> M6 savepoint/conservation ladder -> M7 single-case skill recovery -> M8 24-72 h operational hardening -> M9 performance/D2H recertification -> M10 seasonal statistical equivalence -> M11 release governance.

Using the optimistic ends of the B2 estimates, the shortest plausible closure is **21 development weeks**: 1 + 1 + 2 + 3 + 4 + 2 + 2 + 1 + 4 + 1. That assumes M3 and M5 finish in parallel before M6, no major ADR detour, and no seasonal case fails after single-case recovery. A more realistic planning band is 30-40 weeks because M2, M4, M5, and M10 are high-risk physics/statistics gates.

## B5 - Invariant Ladder

Every sprint close must satisfy these invariants before any "done" language is allowed:

1. Proof object exists: a machine-readable proof file names objective, commit, branch, commands, input paths, output paths, measured numbers, status, and reviewer. No proof object means no close.
2. Evidence freshness: if source code changed, all affected proof objects are regenerated from that commit. Historical M7 numbers can be cited only as prior evidence, not as current acceptance.
3. GPU loop transfer invariant: warmed Nsight capture with at least three warm-up calls and at least five profiled timestep-loop iterations has `d2h_inter_kernel == 0`; loop H2D is zero; pre-kernel D2H is recorded and below ADR-027 thresholds (`ADR-027-d2h-invariant-clarification-PROPOSED.md:50-58`, `:79-92`).
4. Device-resident operational path: accepted runs use `run_forecast_operational`, whose docstring forbids diagnostics, host-read callbacks, host pulls, and sanitizers inside the compiled path (`operational_mode.py:641-646`).
5. Numerical sanity: no nonfinite state, theta guard fallback counts are measured, dry mass is positive, qv and condensates remain nonnegative or have an explicit WRF-compatible limiter proof. Guard behavior is measured where fallback can occur (`operational_mode.py:220-241`, `:578-617`).
6. WRF fixture or oracle parity: any physics or dynamics claim must be backed by a WRF savepoint, analytic oracle, conservation budget, or ensemble evidence. Regression tolerances are stated per variable and cannot be widened without ADR or contract approval.
7. Skill ratchet: same CPU/GPU scorer, same station/time mask, and side-by-side T2/U10/V10 metrics. The finite-score-only gate that caused M7 overclaim is forbidden (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:56-59`).
8. Performance ratchet: after correctness changes, recertify speed from raw logs and never cite 156x. The active prior is 50.20x apples-to-apples (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:22`, `:27-33`).
9. Restart and repeatability: bitwise restart and repeatability remain required because they survived M7 and are cheap regression sentinels (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:18-20`, `:65-67`).
10. Scope and interface control: interface freezes precede parallel work; architecture changes require ADR; user-approved scope expansion is required for irreversible changes (`PROJECT_CONSTITUTION.md:16`).
11. Statistical validity: any equivalence claim must report p values, effect sizes, confidence intervals, case list, season labels, lead-hour coverage, and missing-data masks. A p > 0.05 result without power/effect-size context is not enough.

## B6 - Multi-AI Verification Pattern

Each implementation sprint should have one implementer, one independent proof replayer, and one domain reviewer. For GPU-performance or statistical claims, add specialized review instead of asking a general reviewer to infer correctness from logs.

Mandatory cross-checks:

- Implementer produces the code change and first proof object, but does not mark the sprint closed.
- Independent verifier reruns the exact commands, checks the diff, confirms no unrelated files changed, and confirms proof-object schema completeness.
- Domain reviewer checks WRF meaning: state variables, units, physics order, boundary semantics, and whether the test is a real WRF fixture/oracle rather than a synthetic happy path.
- Performance auditor is mandatory for M9 and any performance claim: raw timing logs, de-duplicated CPU denominator, warmed Nsight transfer audit, and no debug-path profiling.
- Statistics reviewer is mandatory for M10: paired design, case independence, seasonal coverage, missing-data handling, p-value interpretation, effect sizes, and confidence intervals.
- Manager/principal review is reserved for milestone closes, ADRs, scope expansion, and public claims.

Gemini-class or similarly large-context oversight is justified for M0 plan merge, M2/M4/M5 ADR-level architecture choices, M6 savepoint ladder design, M10 statistical-equivalence review, and final publication-facing claims. It is over-budget for routine formatting, one-line parser fixes, simple fixture plumbing, or rerunning already-defined commands.

The absent `tests/savepoint/` directory changes the verification pattern: M1 must create or restore the savepoint fixture surface before later reviewers can rely on "savepoint passed" as a meaningful phrase.

## B7 - Risk Register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Skill regression is not one bug but a stack of dycore, boundary, radiation, and land-surface mismatches | High | High | M0 reconciles current source vs RCA, M2-M6 isolate each subsystem, and M7 does not close until the same scorer shows T2/U10/V10 recovery. |
| Overfitting to the 20260521 24 h case | High | Medium | Treat 20260521 only as a pinned regression case. M10 requires at least 15 seasonal cases and 24-72 h leads per the binding goal (`sprint-contract.md:13`). |
| Performance collapses after restoring correct physics | High | Medium | Keep M9 after correctness, audit transfers with ADR-027, preserve device residency, and require >= 10x speed after all physics changes rather than before them. |
| Missing proof/test/run-entry infrastructure lets milestones close on narrative | High | Medium | M1 creates proof registry, savepoint harness, and operational entry inventory before model-code fixes. No proof object, no close. |
| Statistical equivalence is claimed with low power or wrong masks | High | Medium | Use paired station/time masks, report effect sizes and confidence intervals, require statistics review, and reject p > 0.05 without power/context. |

## B8 - What Can Fail Silently

1. The project can pass a pinned 20260521 24 h skill gate and still fail the binding goal across seasons, 48-72 h leads, or L2/L3 differences. The current latest skill evidence covers only 24 common valid times for one case (`post_iter2_skill_diff.json:104-110`).
2. A p > 0.05 equivalence result can be a low-power false comfort if the 15 cases are not independent enough, station/time masks differ, or effect sizes are operationally large despite non-significance.
3. A speed or D2H milestone can pass by profiling a short or simplified path while production 24-72 h cadence, radiation, boundary, or 1 km runs reintroduce transfers or skip expensive physics. ADR-027 explicitly distinguishes pre-kernel bookkeeping from forbidden inter-kernel D2H and requires warmed operational captures (`ADR-027-d2h-invariant-clarification-PROPOSED.md:50-58`, `:91-94`).

## B9 - Sprint Sizing Recommendation

Use short, proof-first sprints:

| Milestone | Recommended sprint count | Sprint length |
|---|---:|---|
| M0 | 1 | 1 day |
| M1 | 2 | 1-3 days each |
| M2 | 3 | 3-5 days each |
| M3 | 2 | 2-4 days each |
| M4 | 3 | 3-5 days each |
| M5 | 2 | 3-5 days each |
| M6 | 4 | 3-5 days each |
| M7 | 2 | 2-4 days each |
| M8 | 2 | 2-4 days each |
| M9 | 2 | 1-3 days each |
| M10 | 4-6 | 3-5 days each |
| M11 | 1 | 1-3 days |

A good sprint contract is narrow, names forbidden inputs, freezes interfaces, lists exact proof objects, and says what must not be changed. It should never mix broad planning, model-code edits, statistics, and performance claims in one sprint.

Example first sprint contract:

```markdown
# Sprint Contract - M0 Evidence Freeze

Sprint ID: 2026-05-29-m0-evidence-freeze
Role: evidence worker
Branch: worker/<agent>/m0-evidence-freeze
GPU usage: none
CPU pinning: use `taskset -c 0-3` for any Python or test command

Objective:
Produce the authoritative current-state manifest for the project reset. Do not change model code.

Permitted inputs:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/decisions/ADR-027*.md
- .agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md
- .agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json
- .agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md
- src/gpuwrf/runtime/operational_mode.py
- src/gpuwrf/coupling/physics_couplers.py
- src/gpuwrf/contracts/state.py
- exact inventory checks for tests/savepoint/, scripts/run_canary_*.sh, and proofs/

Forbidden inputs:
- .agent/decisions/PROJECT-RESET-PLAN-DRAFT.md
- .agent/sprints/2026-05-28-project-reset-critic/**
- any manager-authored 2026-05-28 decision material

Acceptance:
1. Write .agent/sprints/2026-05-29-m0-evidence-freeze/current_state_manifest.md.
2. Write .agent/sprints/2026-05-29-m0-evidence-freeze/current_state_manifest.json.
3. Manifest records latest T2/U10/V10 CPU/GPU RMSE, corrected 50.20x speed, D2H invariant status, source/RCA divergences, missing proof/test/run-entry directories, and next blocker.
4. Every number cites a file and line.
5. No files outside the sprint directory are modified.
6. Commit locally with a descriptive message.

Proof objects:
- current_state_manifest.json
- command_log.txt
- git_status_before_after.txt
```

## B10 - One Paragraph for the Principal

The project is fast and has real GPU-residency progress, but it is not yet a trustworthy WRF replacement. The honest speed number is about 50x, and the loop-transfer invariant has evidence behind it, but the latest side-by-side forecast check still shows the GPU run several times worse than CPU WRF on temperature and 10 m winds for the pinned 24 h case, while the required 15-case seasonal 24-72 h proof does not exist. I would reset around evidence: first freeze exactly what is true now, then prove the model core, surface physics, radiation, land surface, and boundaries against WRF fixtures, then recover one-case skill, recertify speed, and only then run the seasonal equivalence test. The shortest perfect path is about 21 development weeks; the main uncertainty is whether the remaining forecast error is a few fixable coupling defects or a deeper mismatch with WRF behavior.

PLAN_COMPLETE / Independent blinded reset plan produced from permitted evidence; missing savepoint, run-entry, and proof directories are first-class blockers in the roadmap.
