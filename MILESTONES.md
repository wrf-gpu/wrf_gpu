# Milestones

Before any implementation work in M1-M8, the manager must write a milestone plan and obtain independent review. The first sprint in a milestone is normally a planning/review sprint unless the milestone plan already exists and passes review.

## M0 - AgentOS / Factory Bootstrap

Goal: create governance, memory, skills, sprint templates, scripts, and smoke tests.
Deliverables: required files, validation scripts, skill skeletons, initial commit.
Acceptance gates: `python scripts/validate_agentos.py`, `pytest -q`, repo snapshot.
Likely sprints: bootstrap only.
Blockers: none after repo creation.

## M1 - WRF Oracle And Fixtures

Goal: build trusted WRF-derived and analytic fixtures.
Deliverables: fixture schema, extraction plan, variable mapping, tolerance metadata.
Acceptance gates: fixture manifests validate; at least one idealized and one Canary fixture.
Likely sprints: WRF runner contract, fixture schema, tolerance seed.
Blockers: source WRF access, storage policy.

## M2 - Backend Bakeoff

Goal: choose the primary stack using evidence.
Deliverables: same stencil and column kernel in candidate backends.
Acceptance gates: correctness comparison, profiler JSON, maintainability review, ADR.
Likely sprints: JAX stencil, Triton/custom kernel, GT4Py/DaCe spike, CuPy/Numba/CUDA Tile spike.
Blockers: CUDA/JAX/Triton installation and stable fixture.

## M3 - GPU State And Grid Skeleton

Goal: define device-resident state and grid metadata.
Deliverables: `GridSpec`, `State`, halo contracts, transfer audit.
Acceptance gates: no hidden host/device transfers in dummy timestep loop.
Likely sprints: state object, grid layout, halo abstraction.
Blockers: M2 backend decision.

## M4 - Minimal Dycore

Goal: prove a reduced dycore path with analytic and WRF-like tests.
Deliverables: RK/advection/acoustic subset, invariants, profiler report.
Acceptance gates: tier 1-3 validation passes for chosen reduced cases.
Likely sprints: advection, pressure/acoustic, RK coupling.
Blockers: fixture coverage and precision policy.

## M5 - First Physics Suite

Goal: implement one WRF-compatible physics column subset.
Deliverables: isolated column kernels, fixture tests, edge-case tests.
Acceptance gates: tier 1-2 validation, no register-spill claim without profiler evidence.
Likely sprints: microphysics subset, PBL subset, radiation decision.
Blockers: fixture extraction and backend ergonomics.

## M6 - Coupled Short Forecast

**2026-05-24 split per external consultation (`.agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24-post-consultation.md`).** "Finite because guarded" no longer counts. Gates run sequentially; no progress claim without passing all upstream gates.

### M6a - WRF small-step savepoint parity

Goal: prove the JAX dycore reproduces WRF's small-step operator-by-operator on real Canary d02 inputs.
Deliverables: CPU-WRF savepoint extractor (`module_small_step_em` instrumented), savepoint manifest schema with stagger/units/RK-stage/acoustic-substep metadata, JAX comparator that fails loudly on deliberate perturbation, per-operator parity proofs (coefficient construction, tridiagonal solve, ww/MUTS/t_2ave, advance_w, pressure/geopotential restoration).
Acceptance gates: **sanitizer-off**. 1/2/5/10-step replay matches CPU WRF savepoints within strict per-tier tolerances. No clamps, no caps, no tanh sanitizer.

### M6b - Honest 1-hour Canary d02

Goal: a sanitizer-free 1h coupled forecast bounded by Gen2 noise floor envelope.
Deliverables: physics-on + boundary-on full 1h run with WRF-savepoint-validated operator.
Acceptance gates: no nonfinite at any step. Theta physically bounded. T2/U10/V10 RMSE inside a pre-declared envelope (typically ≤5× Gen2 noise floor: ≈3 K / 7.5 m/s). Interior error not dominated by single boundary or terrain artifact.

### M6c - 6h/24h Gen2 probabilistic consistency

Goal: Tier-4 statistical consistency vs Gen2 backfill on full Canary 3km domain.
Deliverables: 6h and 24h runs across the 17-pair Gen2 noise-floor sample.
Acceptance gates: GPU-vs-Gen2 RMSE bounded by Gen2-vs-Gen2 floating-point divergence envelope (AceCAST-style framing) rather than bitwise equality.

Likely sprints: M6B0 savepoint harness; M6B1-B6 bottom-up port ladder; M6b honest replay; M6c Tier-4 comparator.
Blockers: M4/M5 interfaces; CPU WRF Fortran instrumentation patch authorization.
Validation source: `wrf_l3/`, `wrf_l2/` daily backfill in `/mnt/data/canairy_meteo/runs/` per `.agent/references/cpu-wrf-baseline.md`. Pin run-IDs into the sprint contract; do not wildcard the directory.

## M7 - Canary Operational v0

Goal: daily-run pipeline for Canary 3 km then 1 km.
Deliverables: I/O, restart, post-processing, operational verification.
Acceptance gates: repeatable run, WRF baseline comparison, wall-clock evidence.
Likely sprints: 3 km pipeline, 1 km memory audit, post-processing.
Blockers: GPU memory and I/O throughput.
WRF baseline comparator: the Gen2 daily CPU run for the same AIFS input, per `.agent/references/cpu-wrf-baseline.md` — not a fresh external WRF installation.

## M8 - Public/Forkable Release

Goal: release-ready project with clean governance.
Deliverables: docs, license notices, contribution guide, extension guide.
Acceptance gates: reproducible examples, public naming review, no large artifact leakage.
Likely sprints: docs hygiene, packaging, release validation.
Blockers: legal/name review and stable v0 evidence.
