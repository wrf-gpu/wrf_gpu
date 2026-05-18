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

Goal: couple dycore and physics for short windows.
Deliverables: short forecast driver, conservation checks, drift envelope.
Acceptance gates: tier 3 and initial tier 4 consistency.
Likely sprints: coupling, timestep controls, diagnostics.
Blockers: M4/M5 interfaces.

## M7 - Canary Operational v0

Goal: daily-run pipeline for Canary 3 km then 1 km.
Deliverables: I/O, restart, post-processing, operational verification.
Acceptance gates: repeatable run, WRF baseline comparison, wall-clock evidence.
Likely sprints: 3 km pipeline, 1 km memory audit, post-processing.
Blockers: GPU memory and I/O throughput.

## M8 - Public/Forkable Release

Goal: release-ready project with clean governance.
Deliverables: docs, license notices, contribution guide, extension guide.
Acceptance gates: reproducible examples, public naming review, no large artifact leakage.
Likely sprints: docs hygiene, packaging, release validation.
Blockers: legal/name review and stable v0 evidence.
