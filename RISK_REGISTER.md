# Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Architecture lock-in too early | Multi-year wrong turn | M2 bakeoff plus ADR |
| WRF fixture extraction complexity | No trusted oracle | Start with smallest fixtures |
| GPU memory insufficient for 1 km domain | v0 blocked | 3 km first, memory audit before expansion |
| Physics validation false confidence | Wrong forecasts | Four-tier validation and independent tester |
| Skill/memory bloat | Agent drift | Patch protocol and hygiene passes |
| Agent drift | Incoherent implementation | Sprint contracts and proof objects |
| Hidden CPU/GPU transfers | Performance collapse | Transfer audit and profiler gate |
| Public naming/licensing confusion | Release risk | License notes and human legal review |
| Over-parallelization | Merge conflicts | File ownership and branch policy |
| IC/BC dataset availability and licensing for Canary | M7 blocked, possibly unrecoverable | M3 proof object names AIFS (per `PROJECT_PLAN.md §11.6`) with licensing terms, refresh cadence, and a Gen2-shared fixture. Manager owns the call |
| Terrain / geog / static-field correctness | Physics on wrong topography invalidates all Canary v0 evidence | M3 proof object: provenance file, projection, transform, checksum, sanity check (max elevation, coastline alignment) |
| Observation source for METplus-style verification | M7 verification not falsifiable | M6 closing research-scout sprint identifies station network, satellite/radar feeds, license, storage |
| Full-ensemble runtime and storage cost (100-member Tier-4) | M7 ensemble blows storage/compute budget | M6 small-ensemble prototype establishes per-member cost; M7 full ensemble gated on manager-approved cost model |
| RTX 5090 toolchain maturity for all M2 bakeoff candidates | Bakeoff cannot collect profiler artifacts on Blackwell | Scout candidate-by-candidate Blackwell support; record blockers in candidate-failure artifacts (schema in ROADMAP.md M2) |
| Sprint S1→S2 ordering and parallel work | S2 implements against drafting schema | S2 implementation dispatch gated on S1 schema review pass; read-only S2 scout allowed earlier |
