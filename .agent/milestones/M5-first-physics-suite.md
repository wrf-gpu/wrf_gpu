# M5 - First Physics Suite

Goal: implement one constrained WRF-compatible physics package subset.

Deliverables:

- isolated column fixtures
- implementation in selected backend
- edge-case tests
- profiler artifacts

Acceptance gates:

- **M5-S0 decision-gate sprint** (mandatory first sprint of M5) selects the first physics suite from the Canary operational target stack with recorded rationale; leading candidate per `PROJECT_PLAN.md §11.7` is Thompson microphysics
- fixture parity within tolerance on the M1 column fixture
- invariants pass (tracer positivity, water budget, NaN/Inf)
- register pressure and spill evidence recorded; no regressions vs. corresponding M2 column candidate
- if selected suite is surface/land/SST-coupled: surface/land/SST/static-geog proof object with frozen Canary slice unit test

See `.agent/milestones/ROADMAP.md` M5 for full proof-object list.
