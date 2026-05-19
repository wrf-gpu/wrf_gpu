# Project Plan

Status: **M1 closed 2026-05-19. M2 in progress (2/6 candidates satisfied as of 2026-05-19). Manager runs autonomously per `.agent/goals/M2-MANAGER-RUNBOOK.md`; subsequent milestones dispatch on manager's call with Codex critical review as the second opinion.**
Author: manager (Opus 4.7 1M).
Inputs: `v2 ai driven from scratch plan by deepthink.txt`, `wrf to gpu gpt5.5 deep research.pdf`, `WRF GPU Porting_ Architecture & Verification.pdf`, all governance files in this repo.
Scope: this document is the **synthesis layer** that reconciles the two research briefs with the bootstrap. It does not restate the constitution, scope, spec, principles, validation strategy, precision policy, performance targets, or risk register. Those remain authoritative as-is.

## 1. Why this plan exists

The bootstrap (M0) gave us a governance system and nine-milestone skeleton. It deliberately did not pick a stack, a verification framework, or a sprint sequence. The two research briefs disagree on the most strategic question — the execution backend — and the disagreement is sharp enough that a bad call here would cost us a year. This plan freezes the things both briefs agree on, refuses to freeze the things they disagree on, and routes the disagreement into M2 as an evidence-driven decision.

## 2. What both briefs agree on (freeze now)

These are accepted as project ground truth. No further debate without ADR.

1. **No incremental OpenACC port of legacy WRF.** Both briefs are unambiguous: the architectural ceiling is ~5–7× and our target is ≥8–10×, so the path is structurally barred. Already encoded in `PROJECT_CONSTITUTION.md` (immutable) and `docs/research-inputs.md`.
2. **Whole-state device residency** for the high-frequency model state. No host/device transfers in the timestep loop without an explicit ADR. Already in `PERFORMANCE_TARGETS.md` and the constitution.
3. **Validation is non-bitwise by default.** A four-tier pyramid: (1) micro fixture parity, (2) physical invariants, (3) short-run / timestep convergence, (4) probabilistic ensemble consistency. The two briefs propose different vocabularies (PyCECT vs. probtest, Serialbox vs. icon-exclaim savepoints, TSC) — these are **all complementary** and should be combined, not chosen between. Already encoded in `VALIDATION_STRATEGY.md`; this plan operationalizes it in §6.
4. **MPI rank per GPU with GPU-aware halo exchange** is the only honest path to multi-GPU. Single-GPU comes first per `PROJECT_SCOPE.md`, but the halo interface must be designed not to preclude this.
5. **Profiler artifacts are mandatory** for every performance claim. Schema already in `PERFORMANCE_TARGETS.md`.
6. **One bounded operational target first**: Canary Islands 3 km, then 1 km. Single domain, single physics path, no nesting in v0. Already in scope.
7. **Vertical kernel fusion** at timestep granularity, not micro-kernel orgy. Both briefs cite the launch-bound failure mode of every legacy port.
8. **Explicit precision policy** with FP64 reference for mass continuity and pressure gradients; FP32/BF16 only where validated per-variable per-scheme. Already in `PRECISION_POLICY.md`.

## 3. What they disagree on (do NOT freeze — route to M2)

The deepthink brief recommends **Python + JAX/XLA + Triton** as the AI-native stack. It argues opaque C++ template errors hallucinate AI agents and DSLs lack LLM training data.

The GPT-5.5 brief recommends **Kokkos/C++** (with optional GT4Py-style DSL for stencil-heavy dycore parts) and frames CUDA Fortran as a "tactical NVIDIA-only shortcut, Plan B." It argues all successful next-gen NWP rewrites (SCREAM, HOMMEXX, ICON-exclaim, Pace) are Kokkos- or DSL-based, not JAX.

**Both arguments have weight.** Neither has decisive evidence for *our* specific constraints (single-node RTX 5090, agent-driven authorship, Python-friendly ML coupling, but also long-term portability). The constitution and `ARCHITECTURE_PRINCIPLES.md` already mandate an M2 bakeoff. This plan **enforces** that and adds the bakeoff design (see §5).

Other disputed items:
- **AI swarm topology.** Deepthink proposes α/β/γ adversarial agents. The bootstrap already has a richer cast (manager, gpt-kernel-worker, sonnet-test-engineer, opus-reviewer, profiler-bot, research-scout, human-arbiter). The bootstrap topology stays.
- **Hard speedup target.** Deepthink writes "8×–12×". GPT-5.5 cites MeteoSwiss ICON at 5.5× socket-to-socket as a real production result and is more cautious. This plan adopts the target structure already in `PROJECT_SPEC.md` ("beats the existing CPU operational baseline enough to justify the rewrite path") and refuses to pre-commit a number before the M2 bakeoff produces evidence.
- **ML emulator hybridization.** Deepthink Phase 5. Out of v0 scope per `PROJECT_SCOPE.md`. Kept as research note in `docs/research-inputs.md` only.

## 4. Strategic shape of the project

```
                 M0 ─── M1 ─── M2 ─── M3 ─── M4 ─── M5 ─── M6 ─── M7 ─── M8
              bootstrap fixt. bakeoff state  dycore phys.  couple Canary release
                 │      │      │      │      │      │      │      │      │
                 │      │      └─── ADR-001 backend lock
                 │      │                    └─── ADR-002 state layout
                 │      │                                  └─── ADR-003 dycore precision
                 │      └─── fixtures unblock M2 and every later milestone
                 └─── governance unblocks everything
```

Two things are critical and not yet emphasized enough in the bootstrap text:

1. **M1 must produce *enough* of an oracle to support M2.** A schema and one toy fixture is not enough. M2 needs at least one shared 3D stencil fixture *and* one column-physics fixture so candidate backends can be compared on both regimes. This plan tightens M1's exit criteria in §7.
2. **M2 must include agent-success rate as a first-class metric.** Both briefs argue (from opposite sides) that author productivity matters. The bakeoff measures correctness, profiler artifacts, *and* how many sprints / how many rejection cycles each candidate cost. This plan adds agent-success rate to the M2 acceptance gate in §7.

## 5. Backend bakeoff design (binds M2)

**Bakeoff candidates** (the same two kernels implemented in each, with the same fixture). The set is anchored on the candidate families already named in `ARCHITECTURE_PRINCIPLES.md` and `MILESTONES.md § M2`:

- **A:** JAX / XLA (Python). Pure functional. AOT compile via `jit`.
- **B:** OpenAI Triton (Python). For the column-physics kernel, exercises the register-pressure escape hatch deepthink emphasizes.
- **C:** GT4Py + DaCe (Python). Stencil DSL.
- **D:** Kokkos / C++. The GPT-5.5 recommended path.
- **E:** CuPy raw CUDA or Numba CUDA (Python). Low-overhead Python escape hatch for context.
- **F:** Explicit CUDA C++ ("CUDA Tile" path per `ARCHITECTURE_PRINCIPLES.md:3-4` and `PERFORMANCE_TARGETS.md`'s `cuda-tile` enum). Lower-level than D; included so the bakeoff covers the "manual tile in shared memory" paradigm independent of Kokkos abstraction cost.

**CUDA Fortran** is **not** in the v0 bakeoff *unless* a research-scout sprint produces a concrete NVIDIA-only Canary-v0 justification before M2 dispatch. The plan does not declare it disproven; it declares the v0 bakeoff bandwidth bounded at six candidates and routes the question to the human arbiter (see §11) per `.agent/rules/architecture-decision-policy.md`. This avoids pre-judging the GPT-5.5 brief's Plan B without evidence.

**Shared bakeoff problems** (defined by M1 fixtures):
- **Problem 1 (stencil):** 3D advection-diffusion update on a staggered grid, 4th-order horizontal, 2nd-order vertical, with halo exchange placeholder. Tests fusion ergonomics and memory layout.
- **Problem 2 (column):** A condensed column-physics workload exercising many local prognostics, deep conditional branching, and no horizontal coupling. Tests register pressure and SRAM tiling. We do **not** port any real WRF physics scheme in M2 — too large. We synthesize a representative analog with the same data-flow shape and branching depth as schemes like Thompson microphysics or MYNN PBL, but the analog is **not** a commitment to those schemes as the first physics suite (the M5 decision gate selects the actual operational first suite — see §8).

**Per-candidate proof object:**
- Correctness comparison vs. fixture, in the schema of `PERFORMANCE_TARGETS.md`.
- `ncu`/`nsys` JSON with: kernel launches, occupancy, registers/thread, local-memory bytes, achieved bandwidth, host/device transfer count.
- Maintainability narrative (≤300 words): build complexity, error legibility, debugger story.
- Agent-success log: how many sprints / how many reviewer rejections / how many escalation events the implementation cost.
- If the candidate fails to build, compile, or run on the target hardware: a **candidate-failure artifact** with build log, blocker category (toolchain, license, hardware, agent-success), and reviewer decision. A failed candidate is an evidence outcome, not a bakeoff deadlock — ADR-001 may still proceed with documented coverage gaps.

**ADR-001 (backend lock) requires:**
- Profiler JSON or candidate-failure artifact for both problems in all six candidates.
- Reviewer cross-model challenge per `.agent/rules/cross-model-review-policy.md`.
- Explicit human approval per `.agent/rules/architecture-decision-policy.md` (irreversible).

## 6. Validation architecture (binds M1–M6)

The four-tier pyramid in `VALIDATION_STRATEGY.md` is correct but underspecified for *tooling families*. This plan binds the **family** for each tier and the acceptance behavior, not specific named tools. Concrete tools are chosen by ADR after a research-scout sprint with implementation evidence — they may turn out to be the named candidates below or alternatives.

| Tier | Purpose | Tooling family / acceptance behavior | Candidate tools (non-binding) | First binding milestone |
|---|---|---|---|---|
| 1 | Micro fixture parity | Pre-/post-subroutine savepoints from a trusted CPU baseline; Python comparison harness; per-variable / per-tier tolerances with documented rationale | Serialbox, hand-rolled NetCDF savepoints | M1 (schema + first fixture); M2 (consumed) |
| 2 | Physical invariants | Mandatory: mass-conservation residual, tracer positivity, bounds, NaN/Inf. Optional / scenario-specific: KE spectrum slope ($k^{-5/3}$ check), water/moisture budgets, momentum budgets. | NumPy / project-owned Python | M4 (mandatory subset); M6 (optional scenario-specific) |
| 3 | Short-run / timestep convergence | GPU drift vs. CPU-baseline timestep-sensitivity envelope on a reduced/idealized case where the envelope is computable | TSC1.0 method | M4 → M6 |
| 4 | Probabilistic ensemble consistency | Either a PyCECT-style PCA projection from a perturbed CPU ensemble *or* a probtest-style per-variable tolerance derivation, or both — selection deferred to the M6 ADR, which also sets ensemble size based on actual storage/runtime budget (see risk in §10) | PyCECT, probtest, custom | M6 (small-ensemble prototype) → M7 (production ensemble after budget approval) |

Additional cross-cutting validators required from M3 onward:
- **Transfer audit** (CUPTI or backend equivalent) on every coupled-loop test run.
- **Backend-specific fault-isolation mode** allowing bit-equal compare on small cases for debugging. The exact mechanism — NVHPC `-Mnofma` / `--gpu=math_uniform` (cited by the GPT-5.5 brief for the MPAS/NVHPC stack), XLA determinism flags, Triton-level controls, or equivalent — is **documented in ADR-001 / ADR-002 for the chosen backend**. Never a production target. Not assumed universal across candidates.
- **Compute Sanitizer** (memcheck/racecheck/initcheck/synccheck) on every kernel ADR.

## 7. Proposed stricter milestone gates (pending patch into per-milestone files)

These are **proposals**. The constitution and `MILESTONES.md` remain authoritative. After human approval of this plan, the manager patches the per-milestone files via the normal repository process — this plan does not silently override them.

- **M1 exit (proposed tightening):** At minimum one analytic stencil fixture *and* one analytic column-physics fixture *and* one Canary WRF-derived fixture, plus the manifest schema, plus the storage policy, plus the comparison harness command. **All five are required to close M1.** There is no placeholder-data bypass — if the Canary fixture is missing, M1 remains open. (See §10 for the parallel-work alternative under human approval.)
- **M2 exit (proposed tightening):** Each of the six backend candidates produces either profiler JSON for both bakeoff problems *or* a candidate-failure artifact in the schema of §5. ADR-001 includes a maintainability narrative *and* an agent-success log per candidate. Human approval required.
- **M3 exit (proposed tightening):** Transfer audit shows **zero** host/device transfers inside a 1000-step dummy timestep loop. Not "low," **zero**. The single allowed exception (output) must be cited in ADR-002. `GridSpec` includes named, machine-readable fields for: map projection, terrain/geog static-field provenance, vertical coordinate metadata, halo width, and boundary-condition metadata (deferred from M7).
- **M4 exit (proposed tightening):** Tiers 1 + 2 (mandatory subset) + 3 must all pass for the reduced dycore on at least one idealized case (`em_hill2d_x` or equivalent). Tier 3 envelope is computed from the CPU baseline, not asserted by hand.
- **M5 exit (proposed addition):** Includes an M5-S0 decision-gate sprint that selects the *first physics suite* from the Canary operational target stack, with a recorded rationale. The M2 column-physics analog is **not** by itself a commitment to any specific WRF scheme.
- **M6 exit (proposed addition):** Includes the Tier-4 small-ensemble prototype; full ensemble deferred to M7. Includes a separate METplus-or-equivalent verification-tooling research-scout sprint that completes *before* M7 dispatch — see §10 risk on "METplus learning curve."
- **M7 exit (proposed addition):** Falsifiable I/O / restart compatibility — minimal `wrfinput` / `wrfbdy` / `wrfout` / `wrfrst` compatibility matrix (or an explicit deviation document) and a restart-continuity test. Adds an IC/BC mapping proof object covering source datasets, update cadence, boundary-field variables, interpolation policy, and restart interaction. Adds a surface/land/SST/static-geog proof object (earliest emergence may be M5 or M6 depending on first-suite choice).

## 8. Sprint sequence (v0 horizon)

This is the recommended ordering, not a frozen Gantt. Re-evaluated after each milestone closeout.

```
M1   ──►  S1: fixture-storage-policy + manifest-schema + extraction harness skeleton
M1   ──►  S2: analytic stencil micro fixture + comparison harness CLI
M1   ──►  S3: analytic column micro fixture + WRF variable-map seed
M1   ──►  S4: first Canary WRF-derived fixture (single timestep slice, single column subset)
[ M1 closeout + Codex review of fixtures → green-light M2 ]
M2   ──►  S5..S9: five-way backend bakeoff, one sprint per candidate, frozen interface
[ M2 ADR-001 → human approval → backend lock ]
M3   ──►  S10: GridSpec + State contract draft + ADR-002 state layout
M3   ──►  S11: device-resident state implementation + dummy loop + transfer audit
M4   ──►  S12..S14: advection, pressure/acoustic, RK coupling on reduced dycore
M5   ──►  S15..S17: microphysics subset, then PBL subset, then radiation decision
M6   ──►  S18..S19: coupling + drift envelope + ensemble prototype
M7   ──►  S20..S23: 3 km pipeline, then 1 km memory audit, then post-processing, then ops verification
M8   ──►  S24..S26: docs hygiene, packaging, release validation
```

The first sprint to execute is **S1**. Its contract lives at `.agent/sprints/<date>-m1-fixture-storage-policy/sprint-contract.md` and is the deliverable accompanying this plan.

## 9. Operating rules for the manager (delta on top of existing roles)

- Spawn every subordinate agent (Claude worker, Codex review, profiler runs) in a **named tmux window inside the user's session**. Close the window once deliverables are captured to the sprint folder. Already memoryized.
- Never dispatch a worker without: (a) a sprint contract on disk, (b) reviewer pre-approval when lifecycle requires it, (c) explicit file-ownership statement to avoid parallel edits to shared core files.
- Status reports to the user follow `docs/user-status-report-format.md`: current milestone, what changed, proof objects, blocked decisions, next sprint recommendation.

## 10. Risks added on top of `RISK_REGISTER.md`

(These extend, not replace, the official register. Items the manager believes should be patched into `RISK_REGISTER.md` are marked **[patch]**.)

| Risk | Impact | Mitigation |
|---|---|---|
| M2 bakeoff degenerates into a JAX-vs-Kokkos religious war | Project paralysis, ADR-001 escalation deadlock | Maintainability narrative + agent-success log are first-class metrics, not opinions. Cross-model review per `.agent/rules/cross-model-review-policy.md`. Human-arbiter tie-break. Candidate-failure artifact schema (§5) avoids "all-or-nothing" deadlock |
| M1 fixtures take the entire project budget (deepthink's "Phase 0: Weeks 1-3" is optimistic) | M2 starvation; bakeoff blind | Hard cap of 10 sprints applies to **analytic** fixture work only. The Canary fixture is mandatory for M1 closure. If schedule pressure requires earlier M2 motion, the manager may authorize an M2 *read-only scout sprint* (research-only, no implementation files touched) running in parallel with the remaining Canary fixture work. No M2 implementation sprint dispatches until M1 closes |
| Local RTX 5090 is unavailable or under-provisioned during bakeoff | M2 cannot collect profiler artifacts | Mitigation deferred: smoke-test on smaller GPU first, escalate to cloud H100 only if profiler artifacts are blocked. **[patch]** add to RISK_REGISTER as "RTX 5090 toolchain maturity for all bakeoff candidates" — JAX/Triton/Kokkos/GT4Py compatibility on Blackwell is not guaranteed at M2 entry |
| WRF baseline access for fixture extraction unresolved | M1 S4 blocked | Manager raises this as human decision before S1 starts (see §11) |
| **[patch]** IC/BC dataset availability and licensing for Canary | M7 blocked, possibly unrecoverable | Add explicit M3 ADR proof object naming the source (GFS / ERA5 / ECMWF AIFS / other), licensing terms, and refresh cadence. Manager raises as decision before M3 dispatch |
| **[patch]** Terrain / geog / static-field correctness | Physics on a wrong topography invalidates all Canary v0 evidence | Add M3 proof object: provenance file, projection, transform, checksum, and a sanity check (max elevation, coastline alignment) |
| **[patch]** Observation source for METplus-style verification | M7 verification not falsifiable | Research-scout sprint at the end of M6 (codex dissent on METplus timing). Identifies station network, satellite/radar feeds, license terms, and storage |
| **[patch]** Full-ensemble runtime and storage cost (100-member CPU PyCECT) | M7 ensemble can blow up storage budget, blocks the operational target | Tier 4 prototype at M6 with a *small* ensemble (e.g. 10 members) establishes the per-member cost. Full ensemble at M7 only after the cost model is approved by the human arbiter |
| **[patch]** S1→S2 ordering and parallel work | S2 implementing fixtures against a schema that is still drafting | S2 implementation does not dispatch until S1 schema review passes. A read-only S2 scout (no edits to fixture files) may run in parallel after S1 freezes the schema field list |

## 11. Operational decisions (manager-owned, recorded here)

Per the manager-autonomy directive of 2026-05-19, the user delegates all operational, design, fixture-source, tooling, sprint-sequencing, and architecture-bakeoff decisions to the manager. The user is consulted only at milestone closure or on genuine blockers. The seven items below were operational, not constitutional, and are now manager decisions recorded for audit. For ADRs that the constitution still requires human approval on (backend lock at M2, state layout at M3, public release at M8), the manager runs a Codex cross-model critical review and records the call here in lieu of pre-asking the user.

### Recorded manager decisions (2026-05-19)

1. **WRF baseline source — DECIDED.** WRF v4.7.1 binary at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` (sha `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`). Build environment: `source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`. Compiler: NVIDIA HPC SDK `nvfortran` 26.3. **Source of truth for Tier-1 fixture extraction.**

2. **Fixture storage location — DECIDED.** `/mnt/data/wrf_gpu2/` (1.1 TB free as of 2026-05-19), symlinked at `./data/` in repo. Subdirectories: `data/fixtures/` (binary fixture payloads), `data/runs/` (large WRF or candidate outputs), `data/profiler_artifacts/` (ncu/nsys binary dumps), `data/cache/` (working scratch). `data` is gitignored; only manifests in `fixtures/manifests/` are tracked in git.

3. **CUDA Fortran in M2 — DECIDED: excluded from v0 bakeoff.** No scout sprint scheduled unless M2 candidates A–F all underperform or fail. The previous attempt's `../wrf_gpu/` README documents the CUDA-Fortran-with-tile-resident-restructure path as the *fallback* architecture if Kokkos/JAX/Triton/CUDA-Tile/GT4Py all fail M2. Honored as fallback, not promoted to v0 candidate.

4. **CUDA Tile / explicit CUDA C++ as M2 candidate F — DECIDED: included.** Already reflected in `PROJECT_PLAN.md §5` and `ROADMAP.md M2`.

5. **Analytic-fixture sprint cap — DECIDED: 10 sprints.** Canary WRF fixture still mandatory for M1 closure. See §10.

6. **IC/BC source dataset — DECIDED: AIFS** for M3+ production path (already adopted by Gen2 — see `/home/enric/src/canairy_meteo/Gen2/README.md` §2 "AIFS core"). For M1 fixture work where IC/BC don't bind decisions: reuse existing Gen2 WRF runs from `/mnt/data/canairy_meteo/runs/` rather than triggering new CPU WRF jobs. The S4 (Canary fixture) sprint prefers slicing from a pre-existing Gen2 run over launching a new run.

7. **M5-S0 first-physics-suite decision-gate sprint — DECIDED: approved** as a distinct sprint. Per Gen2's operational stack (Thompson microphysics, MYNN-EDMF PBL, Noah-MP land, RRTMG radiation), the leading candidate for "first physics suite" is microphysics-first (Thompson). The decision-gate sprint will confirm or revise based on M2 backend ergonomics.

### Constitutional gates that remain (manager-exercised, with Codex critical review, reported to user post-decision)

- **ADR-001 (M2 backend lock).** Manager writes; Codex cross-model review; manager finalizes; reports to user at M2 closeout.
- **ADR-002 (M3 state layout).** Same pattern.
- **ADR-003 (M4 dycore precision).** Same pattern.
- **Public release naming/legal (M8).** Reported to user before public posting (the user is the publisher of record).

## 12. What this plan does not do

- Does not pick a backend. M2 does.
- Does not freeze interface contracts. `INTERFACE_CONTRACTS.md` is explicitly placeholders until M3 ADR-002.
- Does not promise a speedup number. `PROJECT_SPEC.md` already states the bar correctly ("beats the operational baseline enough to justify the rewrite").
- Does not change governance, validation strategy, precision policy, performance targets, or scope. Those are constitutional.
