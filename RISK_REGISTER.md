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
| d02 replay hangs before producing real baseline evidence | M6 S2/S2.1 cannot provide the unchanged ADR-023 1h Gen2 baseline, so S3-real would lack a trustworthy before/after comparison | S2.2 is dedicated to root-causing `scripts/m6_d02_boundary_replay_1h.py --duration-s 1` hanging with zero stdout/stderr after the 120s and 1800s S2/S2.1 probes |
| Warm-bubble operator sanity currently `FAIL_PHYSICAL_BOUNDS` | The current ADR-023 path is finite but still violates physical bounds through mass-coupling/stabilizer behavior; accepting the dycore now would hide unresolved physics risk | ADR-024 makes this a diagnostic failure, not an amplitude failure; S3-real must remove or source-ratify load-bearing stabilization before ADR-023 acceptance |
| ADR-021 carry expansion is non-viable without unphysical clamps | Treating ADR-021 as a clean fallback could reintroduce target-shaped warm-bubble passes and huge hidden state blowups | The clamp-strip honest test is the controlling evidence: `FAIL_FINITENESS` at step 2 with catastrophic theta and vertical-velocity growth; any ADR-021 revival requires a new sourced stabilization plan and reviewer approval |
| Both ADR-023 and ADR-021 require sourced stabilization before acceptance | The project can burn more sprints alternating architectures without closing Tier-3/Tier-4 | Follow the HYBRID sequence: baseline first, then source-backed mu/metric cleanup, then Tier-3 and Tier-4; if gates still fail, S6 writes an explicit architecture blocker instead of silently extending M6 |
| M6 close gate can be misread as warm-bubble amplitude | Agents may tune to a single unsourced `[5, 10] m/s` target rather than the validation pyramid | ADR-024 records warm-bubble as operator sanity only. Binding M6 evidence remains Tier-3 convergence, initial Tier-4 Gen2/observation consistency, conservation/bounds, and clean transfer audit |
| About 20 experiment-backed stabilizer findings remain after S3-narrow | Hidden tuning can survive behind source-backed improvements and contaminate Tier-3/Tier-4 evidence | Continue provenance scanning after real d02 baseline exists. S3-narrow reduced 28→20 experiment-backed findings and 8→37 source-backed findings; remaining hits in limiter/hyperdiffusion/orchestrator paths must be classified before closeout |
