# wrf_gpu2

A GPU-native, WRF-compatible regional NWP system designed and built almost entirely by an AI agent swarm. The eventual operational target is **Canary Islands daily forecasting** (3 km then 1 km) on a single-workstation RTX 5090.

This is not a port of legacy WRF. It is a clean rewrite that targets the GPU memory hierarchy from day one and validates against WRF as an oracle rather than inheriting WRF's architecture.

## Current status — v0.1.0 (release candidate, tag PENDING)

**What v0.1.0 is:** a JAX-native, single-GPU port of the WRF v4 split-explicit dycore plus a
physics suite (Thompson microphysics, WRF revised surface layer, MYNN PBL, RRTMG-style SW/LW
radiation), validated for **Canary Islands 1–3 km daily forecasting** on one RTX 5090. It runs a
**single-domain REPLAY path** — the lateral boundaries and land/SST fields are replayed from
existing CPU-WRF / Gen2 corpus artifacts — and is **not yet** a self-contained, multi-domain,
live-nesting WRF with native WPS/real.exe initialization. That distinction is stated plainly here
and tracked in [`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md); the gap chain is
the v0.2.0 roadmap.

**v0.4.0 scope (native standalone init + LBC):** the model can now assemble its own
`wrfinput`/`wrfbdy` from met_em-stage forcing instead of consuming `real.exe` output. **Native
init is PROVEN equivalent to `real.exe` at t=0** (S5 savepoint parity; only a documented 1-cell
categorical LSM-init residual), and the 24 h standalone forecast is stable/finite. The honest claim
is **standalone native-init equivalence + a stable forecast, with a documented near-surface
wind-bias limitation under investigation** (a domain-uniform near-surface westerly excess of
+0.75–1.2 m/s on a 2-date MAM sample; T2 correct). After 10 debug rounds that bias is ruled out vs
unmodified WRF against every faithful ported operator/scheme — it is a tracked, dynamical
forecast-skill item, **not** a fidelity bug and **not** fixed. This is **not** "full standalone
forecast skill." See [`.agent/decisions/V0.4.0-CLOSE.md`](.agent/decisions/V0.4.0-CLOSE.md) and
[`proofs/v040/v040_close_proof.json`](proofs/v040/v040_close_proof.json).

The **binding proof contract** is [`publish/VERIFICATION.md`](publish/VERIFICATION.md) (11 rows)
and the executed-outcome record is [`proofs/PROOF_TABLE.md`](proofs/PROOF_TABLE.md). Tally on the
HFX-fix release HEAD: **9 PASS / 1 FAIL (comparator-harness gap, not a production defect) / 1
INCONCLUSIVE**.

### Proof-table summary (every number traces to `proofs/PROOF_TABLE.md`)

| Row | Claim | Outcome |
|---|---|---|
| 1, 2 | Idealized dycore: Skamarock warm bubble + Straka density current vs published references | **PASS** (6/6 each) |
| 3 | Operator parity vs pristine WRF v4 savepoints | **FAIL — comparator-harness gap, NOT a production-dycore defect.** The savepoint oracle is an hourly `wrfout` history state, not a true per-RK/restart-complete WRF savepoint, so the validation-only coupled-step comparator is fed a state missing ~30 `small_step_prep`-derived leaves and goes non-finite at step 1. Independently confirmed by two models (Opus + GPT-5.5). The production dycore is proven by rows 1/2/7 + the d02/d03 real-case runs (operational `small_step_prep → _rk_scan_step` path, finite over full forecasts). Regenerating true per-step savepoints is a tracked v0.2.0 follow-up. |
| 4 | **Canary 3 km (d02)**: finite/stable to 72 h, beats persistence on winds | **PASS** — 3-case **D02_VALIDATED**, no regression from the HFX fix |
| 5 | **Canary 1 km (d03)**: 24 h finite, bounded gate, beats persistence (secondary claim) | **PASS** — **D03_1KM_VALIDATED**; T2 RMSE 1.92 K ≤ 3.0 K beats persistence; field-qualified U10/V10 |
| 6 | TOST machinery + underpowered n=3 single-season MAM descriptive check | **PASS (qualified)** — U10 equivalent within margin, V10 borderline (tost_p 0.052), T2 NOT equivalent (Δ +0.86 K). **Underpowered single-season MAM descriptive check, never "equivalence PASS."** Full seasonal n≥15–27 is v0.2.0. |
| 7 | Conservation: guards-off finite + genuinely fp64 on real d02 | **PASS** — guards not load-bearing |
| 8 | Reproducibility: deterministic re-run + restart-continuity | **PASS** |
| 9 | Performance vs 28-rank CPU-WRF d02 | **PASS** — **~5–8×** (warmed ~15–16 s/fc-hour), dt-matched floor **3.2×** (d02-only). **NOT ≥10×.** |
| 10 | Precipitation: honest characterization (not parity) | **PASS** — jax 0.393 mm vs WRF 0.347 mm, ratio **1.13**; water closure 2.6e-6 |
| 11 | Device residency: zero host↔device transfer in the timestep loop | **INCONCLUSIVE** — byte-counted audit could not extract per-event sizes; residency is **architecturally guaranteed** (whole-state pytree on device; the scanned timestep does no host transfer by construction). v0.2.0 follow-up; not a forecast-correctness gate. |

### Honest limitations (do not over-read the PASS rows)

- **Single-domain replay, not a full WRF.** Boundaries + land/SST are replayed from CPU-WRF/Gen2
  artifacts. Live multi-domain nesting, native WPS/real.exe init, prognostic Noah-MP, and d01
  cumulus are out of v0.1.0 scope — see `publish/GPU_PORT_GAPS_TODO.md`.
- **The surface-layer / HFX repair is an empirical, partial, MYNN-inspired land thermal-roughness
  fix — NOT a faithful `module_sf_mynn.F` port.** It collapsed the pre-fix d03 daytime warm bias
  to d02 quality and caused no d02/d03 regression, but the claim is narrowed accordingly. Faithful
  MYNN/HFX parity is the first v0.2.0 (0.1.1) item.
- **TOST is underpowered + single-season** (n=3 MAM). It is a descriptive paired-delta check, not
  a seasonal equivalence result.
- **Speed is ~5–8×, not ≥10×.** The fp64 acoustic core is the per-step hot path; an fp32 downcast
  was implemented and validated numerically but gives ~0× additional speedup. Closing to ≥10×
  needs deeper fusion/kernel work and is post-v0.1.0.
- **Row 3 (savepoint comparator) and row 11 (byte-counted device audit) are tracked v0.2.0
  follow-ups**, not relaxed-away passes.

Full release narrative: [`RELEASE_NOTES_v0.1.0.md`](RELEASE_NOTES_v0.1.0.md). Dycore status:
[`proofs/f7/DYCORE_STATUS.md`](proofs/f7/DYCORE_STATUS.md). v0.2.0 roadmap:
[`.agent/decisions/V0.2.0-PLAN.md`](.agent/decisions/V0.2.0-PLAN.md).

## Core goals (immutable)

1. **GPU-native architecture.** Whole-state device residency after init. No host/device transfers inside the timestep loop without an ADR. Fused timestep-scale kernels, not 200 000-launch micro-kernels.
2. **Operational skill parity with CPU WRF v4** on Canary L2/L3 cases: 24-72 h RMSE on T2, U10, V10 is **statistically equivalent under TOST** at predeclared operational margins on a **≥ 15-case seasonal ensemble**.
3. **Performance vs 28-rank CPU WRF** on the same workstation, re-certified after every correctness fix (no stale speedup claims). The long-term aspiration is ≥10×; the **honest v0.1.0 measured number is ~5–8×** (warmed ~15–16 s/fc-hour) on the d02 3 km Canary case (per-forecast-hour, fp64), with a **~3.2× dt-matched floor** — **NOT ≥10×** (proof table row 9). The earlier "22.26×" claim is **retracted** — it divided one GPU domain by the whole multi-domain CPU nest. See `publish/runtime_optimization_analysis.md` for the roofline-grounded provenance and `proofs/PROOF_TABLE.md` row 9 for the executed number.
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
- **Performance / optimization analysis** — [`publish/runtime_optimization_analysis.md`](publish/runtime_optimization_analysis.md): roofline-grounded, proof-cited account of where the per-step compute time goes (~5.3× clean / ~7.8× realistic vs 28-rank CPU-WRF d02, fp64, single RTX 5090), why the fp64 acoustic core is near-optimal, and the four measured-and-refuted optimizations (fp32 dynamics, CUDA command-buffers, fp32-Thompson, implicit sedimentation)
- **GPU port gaps / roadmap to a full WRF v4 replacement (honest TODO)** — [`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md): a code-grounded, prioritized (P0/P1/P2) inventory of what the current single-domain d02 (+ 1 km d03) **replay** path still lacks before it is a complete standalone nightly CPU-WRF replacement. The accurate current claim is narrow: a single-domain GPU forecast/replay path with WRF-faithful core pieces — **not yet a full WRF v4 port**. P0 chain to get there: real-terrain/map-factor/boundary dynamics closure (the top forecast-quality lever) → live multi-domain nesting → prognostic Noah-MP → d01 cumulus → WRF-compatible restart/output → coupled conservation budgets → native WPS/real.exe initialization **last** (the port still consumes CPU-WRF/Gen2 artifacts for IC/boundaries/land).
- **Post-0.1.0 roadmap (sequencing + effort + release cadence)** — [`.agent/decisions/POST-0.1.0-ROADMAP.md`](.agent/decisions/POST-0.1.0-ROADMAP.md): release-cadenced plan over the gaps TODO. After v0.1.0, work P0-6 then nesting; cut a `0.1.x` per item; **v0.2.0 = all gap items except native init**; native init (the riskiest, never-done item) is deliberately **last**, after 0.2.0. Effort calibrated to the ~1-week swarm wall-clock that built the validated core.

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
