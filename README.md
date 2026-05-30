# wrf_gpu2

A GPU-native, WRF-compatible regional NWP system designed and built almost entirely by an AI agent swarm. The eventual operational target is **Canary Islands daily forecasting** (3 km then 1 km) on a single-workstation RTX 5090.

This is not a port of legacy WRF. It is a clean rewrite that targets the GPU memory hierarchy from day one and validates against WRF as an oracle rather than inheriting WRF's architecture.

## Current status — PROJECT RESET 2026-05-28

M0-M7 closed. **v0.0.1 shipped to `github.com/wrf-gpu/wrf_gpu` 2026-05-28** with bitwise dycore savepoint parity at 100 coupled steps vs unmodified WRF v4 and a corrected **22.26× apples-to-apples speedup** vs 28-rank CPU WRF. The same release also documented an **operational skill regression**: T2 RMSE +161-378 %, U10 +214-370 %, V10 +177-353 % vs CPU WRF on a 5-day Canary case.

The principal's reading 2026-05-28: that is not a usable GPU port. The project is therefore **reset** to close the operational-skill gap. The active plan is **[`.agent/decisions/PROJECT-RESET-PLAN-FINAL.md`](.agent/decisions/PROJECT-RESET-PLAN-FINAL.md)**, merged from an Opus draft + a codex adversarial critique + a codex blinded plan-from-scratch (2026-05-28).

### Where the project actually stands (2026-05-28)

- **Position**: ~33 % of the way to "Canary L2/L3 forecasts statistically equivalent to CPU WRF v4 under a TOST equivalence test on a ≥15-case seasonal ensemble" — both critic and blinded independently converged on this number.
- **Rails are built**: foundation, governance, multi-agent orchestration. ⚠️ The earlier "bitwise dycore parity at 100 steps" claim is **RETRACTED** — it was a JAX-vs-JAX self-compare, not vs WRF Fortran (see the 2026-05-29 update below). The dycore is being honestly rebuilt.
- **Operational stack is two-thirds remaining work**: physics couplers savepoint-verified, surface-flux + MYNN parity, radiation + land-surface diurnal, lateral boundary completeness, conservation closure, prognostic Noah-MP, static-field/LU_INDEX parity, idealized-case suite, statistics design, validation corpus, TOST equivalence ensemble.
- **Roadmap M8 → M23** in 7 phases over **32-45 honest weeks** (target Q1-Q2 2027). The earlier 17-23 week target is rejected as not honest after the critic + blinded review.
- **Publish repo + v0.0.1 paper/tag are frozen** until M23 (v0.1.0).

### Dycore rewrite update (2026-05-29)

The "bitwise dycore savepoint parity at 100 coupled steps" headline from v0.0.1 was found to be a **JAX-vs-JAX self-compare** (the comparator read back JAX's own output, never WRF Fortran). The operational dycore was in fact missing ~7 WRF operators. It is therefore being **honestly rebuilt** (the F7.A–J sprint chain), validated against **published idealized-case references** (Skamarock warm bubble, Straka density current) and against **pristine WRF v4.7.1 ground-truth savepoints** (now built — see `proofs/f7/DYCORE_STATUS.md`).

Status: large operator classes are now **verified correct vs WRF** (acoustic small-step core, implicit w/ph solve + `calc_coef_w`/`epssm`, flux-form WS5/3 advection, MUT/MUTS mass semantics, `calc_p_rho_phi`, `rhs_ph`/`ph_tend`). The exponential vertical runaway is eliminated and the warm bubble rises coherently. One localized residual remains (vertical scalar transport / deformation-vs-translation); the idealized-case gate is **not yet fully passing**, so the dycore is **not yet closed**. Single source of truth: **`proofs/f7/DYCORE_STATUS.md`**.

### M19 viability-drive update (2026-05-30)

The dycore is now **closed** (Skamarock warm bubble + Straka density current idealized gates pass vs the benchmark references). On top of it, the **full coupled real-case forecast now runs end-to-end**: a stable, physical, finite **72 h Canary d02 forecast** (dycore + Thompson microphysics + revised surface layer + MYNN PBL + RRTMG radiation + lateral boundaries), across 3 independent corpus cases. This drive closed the perf bottleneck (segmented host-loop scan), made the couplers genuinely fp64, and fixed two coupled-stability defects (a MYNN `w` re-injection that detonated at ~15 h, and a coupled-vs-decoupled surface-`w` boundary condition; the latter stress-tested by an adversarial cross-check and resolved with GPU-vs-CPU-WRF data).

**Honest skill + speed status — real engineering progress, but not yet at the v0.1.0 bar:**

- **Skill** (vs corpus CPU-WRF at 24/48/72 h): **T2 carries genuine skill** (RMSE ~1.0–1.2 K; beats a persistence baseline by 3–7 % at every lead). **Winds do not yet** — U10 is persistence-grade and **V10 is currently *beaten* by persistence**. The operational wind-skill goal is **not met**; V10/wind skill is the **#1 remaining science gap**.
- **Speed:** the honest, provenance-backed number is **~5.7× vs 28-rank CPU-WRF** on the same 3 km d02 (per-forecast-hour, fp64). The earlier **"22.26×" is retracted** — it compared one GPU domain against the *whole multi-domain CPU nest*. An fp32 downcast was implemented and validated numerically but gives **~0× additional speedup** (the fp64 acoustic core is the per-step hot path), so **≥10× is not yet reached** and needs deeper fusion/kernel work.

**Net:** the *engineering* — a stable, physically-faithful, fast-enough multi-day GPU WRF forecast — is in hand; the two gaps to v0.1.0 are **(1) wind forecast skill** and **(2) ≥10× speed**, both active milestones. Single source of truth for this drive: `proofs/m19/` (3-case verdict + persistence baseline + terrain-w resolution), `proofs/perf/` (segmented scan + speedup denominator + fp32 gates), `proofs/stability/`.

## Core goals (immutable)

1. **GPU-native architecture.** Whole-state device residency after init. No host/device transfers inside the timestep loop without an ADR. Fused timestep-scale kernels, not 200 000-launch micro-kernels.
2. **Operational skill parity with CPU WRF v4** on Canary L2/L3 cases: 24-72 h RMSE on T2, U10, V10 is **statistically equivalent under TOST** at predeclared operational margins on a **≥ 15-case seasonal ensemble**.
3. **Performance ≥ 10× vs 28-rank CPU WRF** on the same workstation, re-certified after every correctness fix (no stale speedup claims). Current corrected number: 22.26× on the d02 5-day Canary case (pre-skill-fix).
4. **Validation against WRF, not bitwise reproducibility.** Tiered pyramid: micro fixture parity → physical invariants → short-run / timestep convergence → station-RMSE TOST equivalence.
5. **Forkable and auditable.** Every claim has a proof object on disk. Every architecture decision has an ADR with cross-model review.
6. **Manager-led, agent-executed.** The user is consulted only at milestone closure and on genuine blockers. All sprint work runs autonomously, with workers auto-notifying the manager on exit via tmux send-keys.

## Where to look first (in this order)

| When you want to… | Read |
|---|---|
| Understand what cannot change | [`PROJECT_CONSTITUTION.md`](PROJECT_CONSTITUTION.md), [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md), [`PROJECT_SPEC.md`](PROJECT_SPEC.md) |
| Understand the active plan | **[`PROJECT_PLAN.md`](PROJECT_PLAN.md)** — the synthesis layer; updated when scope or strategy genuinely shifts |
| See milestone-by-milestone proof objects | [`.agent/milestones/ROADMAP.md`](.agent/milestones/ROADMAP.md) |
| See what's been decided so far | [`.agent/decisions/`](.agent/decisions/) — ADRs + Codex cross-model reviews + milestone closeouts |
| Track sprint activity | [`.agent/sprints/`](.agent/sprints/) — one folder per sprint with contract + worker/tester/reviewer reports + closeout |
| Understand agent roles & rules | [`AGENTS.md`](AGENTS.md), [`.agent/roles/`](.agent/roles/), [`.agent/rules/`](.agent/rules/) |
| Find the active milestone's goal condition | [`.agent/goals/`](.agent/goals/) — one `<M>-DONE.md` + one `<M>-MANAGER-RUNBOOK.md` per milestone |

## Live docs (updated as work progresses)

These change as the project advances. Trust the latest commit, not screenshots:

- [`PROJECT_PLAN.md`](PROJECT_PLAN.md) — status banner + manager decisions + escalations
- [`MILESTONES.md`](MILESTONES.md) — milestone gates (tightened per `PROJECT_PLAN.md §7`)
- [`.agent/milestones/ROADMAP.md`](.agent/milestones/ROADMAP.md) — proof-object checklists
- [`.agent/milestones/M*-*.md`](.agent/milestones/) — per-milestone files with Reviewer Decision flipping `Pending → Accepted` on close
- [`.agent/decisions/MILESTONE-M*-CLOSEOUT.md`](.agent/decisions/) — written by the manager at every milestone close
- [`RISK_REGISTER.md`](RISK_REGISTER.md) — grows as new risks surface
- [`.agent/sprints/<date>-<id>/`](.agent/sprints/) — live during execution; archived once closed

## Frozen governance (do not edit during sprints)

- [`PROJECT_CONSTITUTION.md`](PROJECT_CONSTITUTION.md) — immutable end goal + non-negotiables
- [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md) — agent operating rules
- [`ARCHITECTURE_PRINCIPLES.md`](ARCHITECTURE_PRINCIPLES.md), [`VALIDATION_STRATEGY.md`](VALIDATION_STRATEGY.md), [`PRECISION_POLICY.md`](PRECISION_POLICY.md), [`PERFORMANCE_TARGETS.md`](PERFORMANCE_TARGETS.md)
- [`INTERFACE_CONTRACTS.md`](INTERFACE_CONTRACTS.md) — placeholders until M3 ADR-002
- [`.agent/rules/*.md`](.agent/rules/), [`.agent/roles/*.md`](.agent/roles/), [`.agent/skills/*/SKILL.md`](.agent/skills/)

## How sprints run

1. Manager opens a milestone with a reviewed milestone plan (see `.agent/milestones/`).
2. Manager creates a sprint folder via `python scripts/create_sprint.py <slug>` and writes a narrow contract.
3. Manager dispatches roles via `bash scripts/dispatch_role.sh <role> <sprint-folder>`. Roles run in tmux windows inside the manager's session and `tmux send-keys` their summary back when done.
   - **worker** → codex gpt-5.5 (implementation)
   - **tester** → Claude Opus 4.7 xhigh (cross-AI verification — different blind spots than the worker)
   - **reviewer** → codex gpt-5.5 (binding judgment)
   - **critical-review** → codex gpt-5.5 (manager's second-opinion path for non-routine decisions)
4. Sprint closes when `python scripts/close_sprint.py <sprint-folder>` returns `ok: true`.
5. Manager integrates the sprint branch into `main` via `git merge --no-ff`.

The manager runs in a self-paced `/loop` with `ScheduleWakeup` backstop and stops when the active `check_<m>_done.py` returns `ok: true`.

## Validate

```bash
python scripts/validate_agentos.py     # required files + skill metadata
pytest -q                              # full test suite
python scripts/check_m1_done.py        # M1 oracle
python scripts/check_m2_done.py        # M2 oracle
python scripts/repo_status_snapshot.py
```

## Layout

```
.
├── PROJECT_PLAN.md                  active plan (synthesis layer)
├── PROJECT_CONSTITUTION.md          immutable end goal
├── AGENTS.md / CLAUDE.md            agent operating rules
├── ARCHITECTURE_PRINCIPLES.md       backend / runtime principles
├── VALIDATION_STRATEGY.md           four-tier validation pyramid
├── PRECISION_POLICY.md              FP64/FP32/BF16 rules
├── PERFORMANCE_TARGETS.md           profiler JSON schema + transfer rules
├── INTERFACE_CONTRACTS.md           GridSpec, State, Tendencies (placeholder)
├── RISK_REGISTER.md                 living risk list
├── MILESTONES.md                    milestone gates
├── PLANS.md                         execution-plan template (sprint authors copy into sprint folder)
├── LICENSE_NOTES.md                 WRF naming + licensing reminders
├── CONTRIBUTING_AGENT.md            five-rule agent onboarding
├── .agent/
│   ├── goals/                       per-milestone goal spec + manager runbook
│   ├── milestones/                  per-milestone files + ROADMAP.md
│   ├── decisions/                   ADRs, reviews, milestone closeouts
│   ├── sprints/                     one folder per sprint
│   ├── roles/                       role definitions
│   ├── rules/                       merge gates, branch policy, etc.
│   └── skills/                      project-local skills (authoritative)
├── docs/                            user-facing references (storage policy, glossary, etc.)
├── fixtures/                        manifest schemas + analytic samples + Canary slice
├── src/gpuwrf/                      implementation code (grows per milestone)
│   ├── fixtures/                    fixture generators + WRF slicer
│   ├── validation/                  comparison harness CLI
│   └── backends/                    M2 candidate implementations (cuda_tile, cupy, ...)
├── scripts/                         CLIs: dispatch_role.sh, check_m*_done.py, validators
├── tests/                           pytest suite
├── artifacts/                       per-sprint structured artifacts (profile JSONs, etc.)
├── codex/                           codex prompts (the active one is generated by dispatch_role.sh; legacy stubs removed)
├── evals/                           AgentOS structural eval tests
├── data/                            symlink to /mnt/data/wrf_gpu2/ (gitignored)
└── *.pdf + deepthink.txt            original research inputs
```

## Do not do

- Do not implement model code without a sprint contract.
- Do not lock a backend before M2-S8 (ADR-001).
- Do not claim physics correctness without WRF-fixture or analytic-oracle evidence.
- Do not claim GPU performance without profiler artifacts.
- Do not commit binary fixture data; large payloads live under `data/` (symlinked external storage).
- Do not edit governance files inside a sprint; raise an ADR or escalate at milestone close.
