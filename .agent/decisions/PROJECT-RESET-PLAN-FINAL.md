# Project Reset Plan — FINAL (post critic + blinded merge)

**Status**: APPROVED-PENDING-PRINCIPAL-SIGNOFF
**Created**: 2026-05-28
**Supersedes**: `PROJECT-RESET-PLAN-DRAFT.md` (same date)
**Inputs**: `.agent/sprints/2026-05-28-project-reset-critic/critique.md`, `.agent/sprints/2026-05-28-project-reset-blinded/plan.md`
**Principal directives** (2026-05-28): skill gate = paired-test equivalence on ≥15 cases; freeze v0.0.1; Sept-Oct 2026 originally requested.

> **Headline correction**: the draft's 17-23 week target is not honest. Critic and blinded independently land on **30-45 weeks** (Q1-Q2 2027 delivery) for the full scope. The reset accepts this and prefers honesty to schedule pressure.

---

## Binding goal (top of every contract from now on)

A JAX-native GPU port of WRF v4 that delivers Canary L2/L3 forecasts whose **24-72 h RMSE on T2, U10, V10 is statistically equivalent to CPU WRF v4** under a **two-one-sided test (TOST)** at predeclared operational margins on a **≥ 15-case seasonal ensemble**, while preserving **≥ 10× speedup vs 28-rank CPU WRF** on the same workstation.

**Critic correction adopted**: "paired t-test p > 0.05" alone is *failure to reject*, not equivalence. The gate is TOST with predeclared margins, effect sizes, confidence intervals, and case-list/season/lead-hour transparency. See `.agent/sprints/2026-05-28-project-reset-critic/critique.md` C1#10.

---

## Position assessment — ~33 % (critic + blinded converge)

| Block | Weight | Done | Contribution |
|---|---:|---:|---:|
| Foundation: JAX/XLA + device residency + governance | 5 % | 100 % | 5.0 |
| Dycore @ 100 coupled steps savepoint parity (validation-mode only) | 10 % | 100 % | 10.0 |
| Physics couplers savepoint-verified vs WRF (operational mode) | 20 % | 30 % | 6.0 |
| Operational composition + guards + boundary completeness | 15 % | 25 % | 3.8 |
| Land surface prognostic (Noah-MP) | 15 % | 10 % | 1.5 |
| Static-field parity (LU_INDEX, roughness, soil category) | 5 % | 30 % | 1.5 |
| Validation corpus + data manifest + statistics design | 10 % | 15 % | 1.5 |
| Conservation closure + idealized-case suite | 5 % | 20 % | 1.0 |
| Statistical equivalence reached (final gate) | 10 % | 5 % | 0.5 |
| Performance preserved under correctness fixes | 5 % | 50 % | 2.5 |
| **Total** | **100 %** | | **~33 %** |

We have rails (foundation + dycore-in-isolation). Roughly **two-thirds of the project remains**, and the remaining work is **broader than the draft suggested** (added blocks: static-field parity, validation-corpus availability, conservation closure, idealized cases).

---

## Milestone roadmap M8 → M23 (16 milestones, 7 phases)

Each milestone names: definable numeric proof, weeks, Δ% completion gained, risk. **Risks are re-rated per the critic.** The critic recommended dropping M14/M15 as separate sprints — we keep them as the final gates because the principal directive explicitly requires statistical equivalence + v0.1.0 release, and skipping them risks closing the project on unverified claims.

### Phase A — Foundation reset (5-8 weeks)

| # | Milestone | Definable proof | Weeks | Δ% | Risk |
|---|---|---|---:|---:|---|
| **M8** | Evidence freeze + proof registry + savepoint harness | `current_state_manifest.json` cites every current RMSE/speed/D2H number to file:line; `proof_index.json` covers 100% of sprint-close gates; `tests/savepoint/` scaffold restored or replaced by named fixture tests; all Canary run entry points enumerated | 1-2 | +2 | Low |
| **M9** | Operational-mode savepoint parity audit | `divergence_map.json` lists first bitwise-divergence operator in `_physics_boundary_step`, step, magnitude, side-by-side WRF Fortran trace, including SWDOWN/GLW/HFX/LH/PBLH/TSK/T2/U10/V10/PSFC/LU_INDEX comparisons | 3-4 | +5 | Medium *(was Low; critic re-rate)* |
| **M10** | Static-field + LU_INDEX parity | `LU_INDEX/HGT/LANDMASK/XLAND/roughness/soil_category` bitwise vs WRF inputs; State pytree extended with `lu_index` leaf | 1-2 | +3 | Low |

### Phase B — Atmospheric correctness (10-16 weeks, partly parallel)

| # | Milestone | Definable proof | Weeks | Δ% | Risk |
|---|---|---|---:|---:|---|
| **M11** | Dycore theta/mu + guard accounting | Zero unexpected guard fallbacks on valid WRF-range states; per-step clip counts logged (INV-10); theta first-hour normalized RMSE vs WRF savepoint ≤ 1e-3; positive-definite limiter replaces envelope clip | 2-3 | +6 | High *(was Low; critic re-rate)* |
| **M12** | Surface flux + MYNN bottom-BC parity | `surface_layer.F` outputs reproduce bitwise; 1h column oracle: post-PBL column-integrated theta/qv/u/v changes within 5 % of expected flux budget; T2 RMSE ≤ 5.0 K on pinned 5-day Canary | 2-4 | +12 | High *(was Medium; critic re-rate — current iter2 wiring exists yet skill still bad)* |
| **M13** | Radiation + land-surface diurnal physics | RRTMG cadence verified; time-varying coszen/albedo/emissivity; land-station T2 diurnal amplitude differs from CPU WRF by ≤ 1 K on pinned case; radiation heating rates all finite | 3-4 | +8 | High |
| **M14** | Lateral boundary + nesting completeness | `apply_lateral_boundaries` covers U/V/W/T/QVAPOR/P/PB/PH/MU; relax-zone width = WRF; boundary strip RMSE ≤ 1e-6 relative vs decoded wrfbdy; interior-vs-boundary first-hour split no longer boundary-dominated | 2-3 | +5 | High *(NEW — critic C5#2 + blinded M5)* |

### Phase C — Closure + idealized validation (4-6 weeks, parallel)

| # | Milestone | Definable proof | Weeks | Δ% | Risk |
|---|---|---|---:|---:|---|
| **M15** | Conservation + closure budgets (mass / water / energy) | Dry mass relative drift ≤ 1e-6 per 24 h; water-budget residual ≤ 1 % of column source/sink magnitude; energy-budget closure within 0.5 W/m²; INV-7 invariant added | 2-3 | +4 | Medium *(NEW — critic C5#3)* |
| **M18** | Idealized GPU forecast-runner | Warm-bubble, density-current, Schaer mountain-wave, acoustic-substep, CFL ladder tests pass vs published references | 2-3 | +3 | Medium *(NEW — critic C5#5)* |

### Phase D — Land surface (8-14 weeks, the big lift)

| # | Milestone | Definable proof | Weeks | Δ% | Risk |
|---|---|---|---:|---:|---|
| **M16** | Prognostic Noah-MP on GPU | Multi-layer soil/snow/canopy thermal + moisture evolution; replaces hourly data replay; bitwise match vs WRF Noah-MP at 24 h on Canary | 8-14 | +14 | Very High *(critic re-rate — full physics port, not extension)* |

### Phase E — Skill recovery + microphysics (3-5 weeks)

| # | Milestone | Definable proof | Weeks | Δ% | Risk |
|---|---|---|---:|---:|---|
| **M17** | Microphysics admissibility removal + Thompson trust | Remove `_thermodynamically_admissible` guard; no NaN over 24 h on canonical case; T2 RMSE ≤ 2.5 K | 2-4 | +3 | Medium *(was Low; critic re-rate — coupled moisture/energy failures not covered by no-NaN alone)* |
| **M19** | Single-case L2/L3 skill recovery | Pinned 20260521 + L2 D02 replay: T2/U10/V10 RMSE *and* MAE each within 20 % of CPU WRF on same scorer + station mask | 2-3 | +6 | High |

### Phase F — Statistical equivalence (8-12 weeks)

| # | Milestone | Definable proof | Weeks | Δ% | Risk |
|---|---|---|---:|---:|---|
| **M20** | Validation corpus build + data manifest + statistics design | ≥ 15 Canary L2 + L3 seasonal cases with verified IC/BC data on disk; TOST margins predeclared per variable; case manifest committed | 3-6 | +6 | High *(NEW — current local inventory failed five-day gate; critic C5#4)* |
| **M21** | TOST equivalence on ≥ 15 cases at predeclared margins | TOST p < 0.05 (both lower and upper bounds rejected) on T2/U10/V10 RMSE deltas at the M20 margins; effect sizes + CIs reported | 4-6 | +10 | High |

### Phase G — Release (3-5 weeks)

| # | Milestone | Definable proof | Weeks | Δ% | Risk |
|---|---|---|---:|---:|---|
| **M22** | Performance + transfer recertification | Warmed Nsight: `d2h_inter_kernel == 0`; loop H2D == 0; ≥ 10× speedup vs 28-rank CPU WRF on canonical case re-measured under final code | 1-2 | +2 | Medium-High *(critic — only 22.26× current margin)* |
| **M23** | v0.1.0 release + arXiv preprint | Public repo tag v0.1.0; arXiv preprint companion; ADRs covering every architectural change; README disclaimer aligned with current evidence; no 156× claim retained | 1-2 | +1 | Low |

---

## Timeline (honest)

| Phase | Weeks | Cum. |
|---|---:|---:|
| A — Foundation reset (M8-M10) | 5-8 | 5-8 |
| B — Atmospheric correctness (M11-M14, partly parallel) | 8-12 with parallelism (10-16 raw) | 13-20 |
| C — Closure + idealized (M15+M18, parallel) | 2-3 (parallel with B/D) | 13-20 |
| D — Noah-MP (M16) | 8-14 | 21-34 |
| E — Skill recovery (M17+M19) | 3-5 | 24-39 |
| F — Statistical equivalence (M20+M21) | 6-10 (M20 parallel with D) | 30-45 |
| G — Release (M22+M23) | 2-4 | **32-49 weeks** |

**Realistic delivery: 32-45 weeks (Q1 2027 to Q2 2027).** The draft's Q3-Q4 2026 target is rejected as not honest.

Δ% gained: +2+5+3+6+12+8+5+4+3+14+3+6+6+10+2+1 = **+90 → ~123 %** if all milestones deliver their estimated gain. The 23-point margin is critic-recommended buffer for the inevitable surprises uncovered by M9 audit (which may itself reveal new failure modes).

---

## "Constantly improve without breaking" — invariant ladder (expanded)

Each sprint close MUST satisfy, with a `proof.json` listing measured values:

- **INV-1**: ADR-027 D2H invariant — warmed Nsight, `d2h_inter_kernel == 0`, pre-kernel D2H below ADR threshold, profiled-step count + parser-version recorded *(critic INV-1 expansion)*.
- **INV-2**: B6 savepoint parity @ 100 coupled steps stays bitwise.
- **INV-3**: From M9 onward, operational-mode parity extends to 1 000+ steps and to operational variables (SWDOWN/GLW/HFX/LH/PBLH/TSK/T2/U10/V10/PSFC/LU_INDEX); once verified, never regresses *(critic INV-2/3 expansion)*.
- **INV-4**: Pinned smoke case shows **no catastrophic regression**; median RMSE over a fixed mini-ensemble is non-increasing vs previous milestone; per-variable waivers require ADR *(critic INV-4 fix — the draft's "RMSE must decrease or hold equal on one case" was too brittle and would reject legitimate fixes)*.
- **INV-5**: Speedup ≥ 10× vs 28-rank CPU WRF, **d02-only denominator, de-duplicated CPU timing, JIT excluded, transfer-audited** *(critic INV-5 expansion)*.
- **INV-6**: No test deleted, no tolerance widened, no `xfail` added without ADR. **Proof-schema-validated**: every invariant missing a proof object = sprint failure, not "not run" *(critic INV-6 expansion)*.
- **INV-7** *(NEW)*: Mass / moisture / energy conservation budgets within thresholds (M15+).
- **INV-8** *(NEW)*: Static-field parity (LU_INDEX, HGT, LANDMASK, XLAND, roughness, soil category) bitwise vs WRF (M10+).
- **INV-9** *(NEW)*: Boundary forcing completeness — U/V/W/T/QV/P/PB/PH/MU coverage; relax-zone width matches WRF (M14+).
- **INV-10** *(NEW)*: Guard/limiter accounting — per-step clip counts logged; first clipped field/cell named; zero hidden fallback in accepted physics windows (M11+).
- **INV-11** *(NEW)*: Evaluation sufficiency — case count, season labels, station joined-rows, predeclared equivalence margins recorded (M20+).

This is the **one-way ratchet**: invariants only tighten, never loosen. The merge gate blocks if any invariant trips.

---

## Multi-AI verification (per critic C7)

**Implementer**: Codex GPT-5.5 xhigh — writes code, writes first proof object, does NOT mark sprint closed.

**Independent verifier**: Opus 4.7 — reruns commands, checks diff, confirms proof-object schema completeness.

**Domain reviewer**: Opus 4.7 (separate window) — checks WRF meaning: state variables, units, physics order, boundary semantics, real WRF fixture vs synthetic happy path.

**Performance auditor** (mandatory M22): raw timing logs, de-duplicated CPU denominator, warmed Nsight transfer audit, no debug-path profiling.

**Statistics reviewer** (mandatory M20+M21, NEW): paired design, case independence, seasonal coverage, missing-data handling, p-value interpretation, effect sizes, CIs.

**Manager (this conversation)**: owns sprint contracts, dispatches, merges, escalates, milestone closes.

**Third independent reviewer + blinded proof auditor** *(critic C7 tightening)*: required at milestone close for **M9, M11, M12, M13, M14, M16, M21**. Gemini agy fills this slot when stakes are high.

**Auto-notify on worker exit** *(principal directive 2026-05-28)*: every worker tmux dispatch includes `tmux send-keys -t 1 "AGENT REPORT: <name> exit=$?" Enter` so the manager is woken by event, not by polling. See `[[feedback-worker-tmux-notify-pattern]]`.

---

## Sprint sizing (per blinded B9 + principal directive)

- Sprint length: **3-5 days for narrow proofs, 1-2 weeks for parity sprints**, never longer.
- Each sprint contract: narrow scope, names forbidden inputs, freezes interfaces, lists exact proof objects, says what must not be changed.
- Each milestone is composed of 1-6 sprints; close requires every sprint's proof object plus a milestone-close audit.
- **No sprint merges with only codex sign-off** — mandatory Opus check.

---

## Publish repo + paper handling (principal-confirmed)

- `/home/enric/src/wrf_gpu/` and v0.0.1 paper + tag **stay frozen** until M23.
- One-sentence README disclaimer pushed early: *"v0.0.1 is a foundation preview; the operational-skill closure is the M8-M23 work tracked in the wrf_gpu2 development repo."*
- No new pushes to `wrf-gpu/wrf_gpu` until v0.1.0 closes M23.
- 156× claim must not appear anywhere — only the 22.26× corrected number, soon to be re-measured at M22.

---

## Risk register (top 5, merged from critic + blinded)

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Skill regression is a stack of defects, not one bug | High | High | M9 first-divergence audit before more fixes; M11-M14 isolate subsystems; M19 only closes on same-scorer T2/U10/V10 recovery |
| Overfitting to single 20260521 case | High | Medium | INV-4 mini-ensemble (median RMSE non-increase); M20 builds proper ≥15-case seasonal corpus |
| Performance collapses after correctness | High | Medium | M22 recertification AFTER all correctness fixes; ADR-027 device residency; speedup ≥10× only re-claimed at M22 |
| Missing proof/test/run-entry infrastructure → milestones close on narrative | High | Medium | M8 builds proof registry + savepoint harness BEFORE any model-code fix lands |
| Statistical equivalence claimed with low power / wrong masks | High | Medium | TOST margins predeclared at M20; statistics reviewer mandatory; effect sizes + CIs always reported |

---

## Three ways this project could fail silently (per blinded B8)

1. Pass pinned single-case skill gate, fail seasonal/lead-hour generalization. **Defense**: M20 corpus + INV-4 mini-ensemble.
2. p > 0.05 equivalence claim with low power. **Defense**: TOST margins + power analysis + effect sizes mandatory.
3. Performance / D2H milestone passes on short/simplified path while 24-72 h, radiation, boundary, 1 km runs reintroduce transfers. **Defense**: INV-1 warmed-Nsight on operational path; M22 only re-runs after correctness.

---

## What I'll dispatch next (after principal sign-off)

**Sprint contract for M8 (Evidence freeze + proof registry + savepoint harness)** — 1-2 weeks, low risk, codex frontrunner + Opus verifier. Specific deliverables:

- `current_state_manifest.json` listing every current measurement cited to file:line
- `proof_index.json` covering 100 % of M8-M23 sprint-close gates
- `tests/savepoint/` scaffold (restored or rebuilt around current `m6b6_coupled_step_compare.py`)
- Operational entry-point inventory (`scripts/run_canary_*.sh` audit)
- Statistics-design ADR for M20+M21 (TOST margins per variable, paired-design specification)
- All discrepancies between RCA (`top_3_suspects.md`) and current source (per blinded evidence-boundary note) reconciled

After M8 lands: M9 op-mode savepoint audit dispatches. Phase A close = end of week 8 (target).

---

## What changes in the project framework today

- **README.md** (manager repo) updated with reset narrative + revised goal + 30-45 week timeline
- **AGENTS.md** updated: binding goal at top, invariant ladder INV-1..11 referenced
- **CLAUDE.md** (manager repo) updated: manager skill highlights post-reset
- **`.agent/skills/manager/*.md`** updated: M8-M23 roadmap referenced, auto-notify pattern adopted
- **`.agent/skills/worker/*.md`** updated: contract template includes binding goal + INV-1..11
- **Project memory** updated: this final plan referenced, new feedback memories added
- **Tasks** restructured: #122-#129 (old M8-M15) closed/deleted; new tasks #130+ for M8-M23 created
