# Milestone Roadmap (v0)

Companion to `MILESTONES.md` and the per-milestone files. This document lists, for every milestone, the **proof objects** required to close it, the **ADRs** it produces, and the **dependencies** that gate the next milestone. It is the merge-gate checklist for the manager.

Status legend: ◯ pending · ◐ in progress · ● complete.

## M0 — AgentOS bootstrap                                  ●
- **Proof objects committed:**
  - Governance files (constitution, scope, spec, architecture principles, validation strategy, precision policy, performance targets, risk register, interface contracts placeholder).
  - Role files for manager, gpt-kernel-worker, sonnet-test-engineer, opus-reviewer, profiler-bot, research-scout, human-arbiter.
  - 13 skill skeletons under `.agent/skills/`.
  - Sprint template, milestone-plan template, fixture manifest template.
  - `scripts/validate_agentos.py`, `pytest -q`, `scripts/repo_status_snapshot.py` all green.
- **Produces no ADR.**
- **Gates:** unblocks everything.

## M1 — WRF Oracle & Fixtures                              ◯
- **Required proof objects** (all five):
  1. `fixtures/manifests/schema.yaml` — pinned manifest schema validated by `scripts/validate_agentos.py`.
  2. `docs/fixture-storage-policy.md` — naming, checksums, external-storage paths, git-exclusion rules.
  3. At least **one analytic stencil micro-fixture** committed as manifest + (≤100 KB) sample slice + comparison-harness CLI invocation.
  4. At least **one analytic column micro-fixture** with the same triplet.
  5. At least **one Canary WRF-derived fixture** (single timestep slice, single column subset; bulk data external; manifest in git).
- **ADRs:** none yet (M1 is intentionally schema-only).
- **Independent review:** required, per `.agent/milestones/M1-wrf-oracle-fixtures-plan.md`.
- **Gate to M2:** all five proof objects present; reviewer signs M1-PLAN "Reviewer Decision = Accepted"; human-arbiter confirms WRF baseline source.

## M2 — Backend Bakeoff                                    ◯
- **Candidate families** (locked at M2 entry, see `PROJECT_PLAN.md §5`): A JAX/XLA, B Triton, C GT4Py/DaCe, D Kokkos/C++, E CuPy or Numba, F Explicit CUDA C++ ("CUDA Tile"). CUDA Fortran only by human-arbiter decision pending scout output.
- **Required proof objects** (per candidate × 2 problems = 12 minimum, plus aggregates):
  - For each candidate A–F:
    - `artifacts/m2/<candidate>/stencil_profile.json` matching `PERFORMANCE_TARGETS.md` schema, **or** `artifacts/m2/<candidate>/stencil_failure.json` in the candidate-failure schema below.
    - `artifacts/m2/<candidate>/column_profile.json` matching the same schema, **or** `artifacts/m2/<candidate>/column_failure.json`.
    - `artifacts/m2/<candidate>/correctness.json` against the M1 fixtures (must run unless the candidate-failure artifact explains why not).
    - `artifacts/m2/<candidate>/maintainability.md` (≤300 words).
    - `artifacts/m2/<candidate>/agent_success.json`: sprint count, reviewer rejections, escalation events.
  - `.agent/decisions/ADR-001-backend-selection.md` — decision, evidence, dissent, and documented coverage gaps from any failed candidate.
- **Candidate-failure artifact schema:**
  ```json
  {
    "candidate": "kokkos|jax|triton|gt4py|cupy|cuda-tile|cuda-fortran|other",
    "problem": "stencil|column",
    "blocker_category": "toolchain|license|hardware|agent_success|build|runtime|other",
    "blocker_summary": "≤200-word description",
    "logs": ["artifacts/m2/<candidate>/build.log", "..."],
    "reviewer_decision": "covered|excluded|escalated",
    "remediation": "what would need to change to retry"
  }
  ```
- **Independent review:** cross-model review (two reviewers) per `.agent/rules/cross-model-review-policy.md`.
- **Human approval:** required (irreversible architecture decision).
- **Gate to M3:** ADR-001 merged; M1 fixtures still passing; profiler artifacts archived; agent-success log accepted; any failed candidate accounted for with a candidate-failure artifact and an explicit "covered / excluded / escalated" reviewer decision.

## M3 — GPU State & Grid Skeleton                          ◯
- **Required proof objects:**
  - `src/gpuwrf/contracts/grid.py` (or equivalent) implementing `GridSpec` from `INTERFACE_CONTRACTS.md`, with **named, machine-readable** fields for:
    - map projection (Lambert, Mercator, polar — selected per Canary v0 target),
    - terrain / geog static-field provenance: source file, shape, units, checksum, projection transform, sanity check (max elevation, coastline alignment),
    - vertical coordinate metadata (eta/sigma/hybrid, level count, top pressure),
    - halo width and staggering,
    - boundary-condition metadata: BC field names, update cadence, source dataset, interpolation policy.
  - `src/gpuwrf/contracts/state.py` implementing device-resident `State`.
  - Dummy 1000-step timestep loop + transfer-audit JSON showing **zero** host/device transfers (output excepted, cited in ADR).
  - `.agent/decisions/ADR-002-state-layout.md` covering state layout *and* the IC/BC source decision (or referencing a sibling ADR if scope balloons).
- **Independent review:** required.
- **Gate to M4:** transfer audit clean; ADR-002 merged; halo abstraction stub committed (implementation deferred); BC metadata schema frozen.

## M4 — Minimal Dycore                                     ◯
- **Required proof objects:**
  - Reduced RK + advection + acoustic kernels implemented in the chosen backend.
  - Tier 1 fixture parity passing on the M1 stencil fixture (within documented tolerance).
  - Tier 2 invariants: mass-conservation residual ≤ tolerance; positivity holds; no NaN/Inf.
  - Tier 3 short-run convergence: drift envelope computed from the CPU baseline; GPU drift inside it for ≥1 idealized case (`em_hill2d_x`-class).
  - `artifacts/m4/dycore_profile.json`.
  - `.agent/decisions/ADR-003-dycore-precision.md`.
- **Gate to M5:** Tiers 1–3 green; ADR-003 merged.

## M5 — First Physics Suite                                ◯
- **M5-S0 decision-gate sprint (NEW)**: a research+manager sprint that selects the *first physics suite* from the Canary operational target stack with a recorded rationale. Output: short ADR or decision memo naming the scheme (e.g. Thompson, WSM6, MYNN, RRTMG, Noah-MP — whichever the operational target actually uses) and the reason it goes first. The M2 column-physics analog is **not** by itself a commitment to any scheme.
- **Required implementation proof objects** (after S0):
  - Selected physics scheme subset implemented as column kernels.
  - Tier 1 fixture parity within tolerance on the M1 column fixture.
  - Tier 2 invariants: tracer positivity, water budget, no NaN/Inf.
  - Register pressure & spill report from profiler (matching `PERFORMANCE_TARGETS.md` schema).
  - Edge-case test suite owned by sonnet-test-engineer.
  - If the selected suite is a surface/land/SST-coupled scheme (or the first suite requires SST/static-geog inputs), a minimal **surface/land/SST/static-geog proof object**: data provenance, ingestion path, schema, and a unit-test using a frozen Canary slice.
- **Gate to M6:** S0 decision merged; Tier 1+2 green; profiler shows no register-spill regressions vs. the corresponding M2 column candidate.

## M6 — Coupled Short Forecast                             ◯
- **Required proof objects:**
  - Short-run driver coupling M4 dycore and M5 physics.
  - Drift envelope vs. CPU baseline documented.
  - Tier 3 convergence on at least one idealized case.
  - **Tier 4 small-ensemble prototype** (e.g. 10 members) using probtest-style per-variable tolerance derivation, PyCECT-style PCA projection, or both — selection driven by the M6 ADR with implementation evidence. Establishes the per-member runtime and storage cost. Full ensemble at M7 only after the cost model is approved by the human arbiter.
  - Transfer audit on coupled run.
  - Surface/land coupling validated end-to-end on a coupled short run if M5 selected a surface-coupled first suite.
- **Required closing-sprint deliverable (NEW, codex dissent #5):** A **verification-tooling research-scout sprint** that produces a brief recommending METplus, EVS, or an alternative for M7 forecast-vs-observation verification. Output: scout report + recommended toolchain ADR draft. The scout sprint completes *before* M7 dispatch so M7 implementation does not block on tooling discovery.
- **Gate to M7:** Tier 3 green; Tier 4 prototype interpretable + cost model approved by human arbiter; coupled transfer audit clean; verification-tooling ADR draft on disk.

## M7 — Canary Operational v0                              ◯
- **Required proof objects:**
  - End-to-end 3 km daily pipeline with the full chain split into named proof objects below.
  - **IC/BC mapping proof object:** source datasets (e.g. GFS / ERA5), update cadence, boundary-field variables (u/v/T/q/p_s/...), interpolation policy, restart interaction, and a test that drives one Canary day from real IC/BC.
  - **I/O / restart compatibility matrix:** minimal `wrfinput` / `wrfbdy` / `wrfout` / `wrfrst` field-by-field compatibility table (or an explicit *deviation document* listing every intentional difference and why).
  - **Restart-continuity test:** run an N-step forecast, stop, restart from the checkpoint, run another N steps, compare against the unbroken run within tier-1 tolerance.
  - WRF baseline forecast vs. GPU forecast comparison on at least one full Canary day, on the surface/land/SST/static-geog setup frozen by M3/M5.
  - Forecast-vs-observation verification using the M6-selected verification toolchain (METplus or alternative), classical scores: T2, wind, precip BIAS/RMSE, plus at least one neighbourhood or object-based score for precip.
  - **Full Tier-4 ensemble** sized per the cost model approved at M6 closeout (target 100 members but the exact number is bounded by the approved storage/runtime budget); pass/fail call documented.
  - 1 km memory audit and operational gaps document.
  - Wall-clock evidence vs. CPU operational baseline.
- **Gate to M8:** repeatable daily run; IC/BC reproducible; restart-continuity test green; meteorological verification within accepted bounds; 1 km feasibility known.

## M8 — Public/Forkable Release                            ◯
- **Required proof objects:**
  - Public docs (architecture, fork guide, extension guide).
  - License and naming review (human-arbiter approved).
  - Reproducible example for an external user.
  - Release validation script green.
- **Gate to next phase (out of v0 scope):** clean public release.

## Cross-cutting checks (every milestone after M3)

These run as part of merge gates regardless of which milestone the PR targets:
- Transfer audit JSON for any change to the timestep loop.
- Compute Sanitizer (memcheck/racecheck/initcheck/synccheck) for any GPU kernel change.
- Profiler JSON for any performance claim.
- Memory-patch validation for any change to stable memory/skills/rules.

## Recomputed risks per milestone (delta on `RISK_REGISTER.md`)

- **M2** — bakeoff stalemate. Mitigation: maintainability narrative + agent-success log are first-class evidence; candidate-failure artifact schema avoids all-or-nothing deadlock; human-arbiter tiebreak by design.
- **M2** — RTX 5090 toolchain maturity (codex finding #19). Mitigation: scout candidate-by-candidate on Blackwell support before profiler sprints; document blockers in candidate-failure artifacts.
- **M3** — IC/BC and terrain ingestion deferred too long collapses M7 (codex findings #7, #9). Mitigation: hoisted into M3 proof objects as named fields in `GridSpec`.
- **M4** — Tier 3 envelope is not well defined for a *reduced* dycore. Mitigation: use a simpler analytic problem (1D advection or 2D hill flow) where the envelope is computable analytically.
- **M5** — first-suite picked by analogy instead of operational need (codex finding #21). Mitigation: M5-S0 decision-gate sprint with recorded rationale.
- **M5** — physics in isolation hides coupling bugs. Mitigation: explicit "isolation harness" in tester report; coupling bugs are M6's responsibility, not M5's.
- **M6** — verification tooling chosen too late (codex dissent #5). Mitigation: closing research-scout sprint produces ADR draft before M7 dispatch.
- **M6/M7** — full-ensemble runtime/storage cost (codex finding #5). Mitigation: small-ensemble prototype establishes per-member cost; full ensemble gated on cost-model approval by the human arbiter.
- **M7** — verification observation-source availability and licensing. Mitigation: handled by the M6 verification-tooling scout.
- **M7** — I/O / restart compatibility unfalsifiable (codex finding #10). Mitigation: explicit `wrf{input,bdy,out,rst}` compatibility matrix and restart-continuity test as named proof objects.
