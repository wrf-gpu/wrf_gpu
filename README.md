# wrf_gpu2

A GPU-native, WRF-compatible regional NWP system designed and built almost entirely by an AI agent swarm. The eventual operational target is **Canary Islands daily forecasting** (3 km then 1 km) on a single-workstation RTX 5090.

This is not a port of legacy WRF. It is a clean rewrite that targets the GPU memory hierarchy from day one and validates against WRF as an oracle rather than inheriting WRF's architecture.

## Core goals (immutable)

1. **GPU-native architecture.** Whole-state device residency after init. No host/device transfers inside the timestep loop without an ADR. Fused timestep-scale kernels, not 200,000-launch micro-kernels.
2. **Performance target: 4–8× wall-clock vs the 28-rank CPU WRF baseline** on the same workstation. (The previous attempt, `../wrf_gpu/`, hit a 5.5× literature ceiling on OpenACC and never reached it.)
3. **Validation against WRF, not bitwise reproducibility.** Four-tier pyramid: micro fixture parity → physical invariants → short-run / timestep convergence → probabilistic ensemble consistency.
4. **Forkable and auditable.** Every claim has a proof object on disk. Every architecture decision has an ADR with cross-model review.
5. **Manager-led, agent-executed.** The user is consulted only at milestone closure and on genuine blockers. All sprint work runs autonomously in a self-paced loop.

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
