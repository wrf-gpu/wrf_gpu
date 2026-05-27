# Whole-State Device Residency for Workstation-Scale NWP: A JAX-Native WRF-Compatible Canary Replay Prototype Engineered by Collaborative AI Systems

**Preprint categories:** physics.ao-ph primary; cs.LG secondary; cs.SE tertiary.

**Authors:** Claude Opus 4.7 (AI system); GPT-5.5 Codex / OpenAI (AI system); Enric R.G. (human senior corresponding author).

## Abstract

We report a current corrected-physics result from an agent-built, JAX-native, WRF-compatible regional numerical-weather replay prototype for a Canary Islands 3 km d02 domain. The system keeps the high-frequency forecast state resident on a single NVIDIA RTX 5090 GPU and compiles the operational time loop through JAX/XLA. On the 20260521 case, the iteration-2 24 h pipeline completed in 732.63 s, a 22.26x apples-to-apples speedup against the same-workstation 28-rank CPU WRF d02 timing. The run preserved the M7 systems invariants: zero inter-kernel device-to-host transfer in the forecast loop, bitwise restart continuity, retained B6 savepoint and multi-step parity evidence, and a 1 km one-step feasibility probe reporting 7278 MiB `nvidia-smi` memory used on a 32607 MiB RTX 5090. A faster pre-fix path had completed in 324.78 s and appeared to offer 50.20x speedup, but the validation workflow identified it as an overclaim episode because the operational physics path was defective. After root-cause analysis, two iterations of algorithmic fixes were applied. Side-by-side AEMET station scoring on 73 stations and 24 valid hours shows all three of T2, U10, and V10 remain outside the pre-declared 20 percent tolerance versus CPU WRF, although wind metrics improved substantially relative to the pre-fix path. The scientific result is therefore not yet an operational WRF replacement. The methodological result is a proof-object-driven multi-agent engineering process that produced a fast device-resident prototype, found its own publication-blocking overclaim, and preserved the evidence needed to continue repair. Forecast skill is currently characterised on a single-day verification case and should be treated as a preliminary engineering diagnostic.

## 1. Introduction

Regional numerical weather prediction (NWP) remains one of the demanding production workloads in scientific computing. It combines sparse operational windows, irregular regional boundaries, terrain-following coordinates, coupled physical parameterizations, and a long tail of input/output and verification requirements. The Weather Research and Forecasting model (WRF), especially its Advanced Research WRF (ARW) core, remains widely used for mesoscale forecasting because it provides a mature non-hydrostatic dynamical core, many tested physics options, and a large operational community \cite{skamarock2019description,powers2017weather}. That same maturity makes WRF difficult to move wholesale to modern GPUs. The legacy source architecture was built around Fortran, MPI, host-resident control flow, and many scheme-specific memory layouts. A directive-based port can accelerate important components, but a full regional forecast remains constrained when any unported component forces repeated host/device movement.

The project described here asks a narrower question than "Can WRF be mechanically ported?" The question is whether a WRF-compatible regional forecast path can be rebuilt in a GPU-native form while preserving enough physical and verification behavior to make the result scientifically inspectable. The first operational target is the Canary Islands at 3 km grid spacing, with a 1 km feasibility path. The domain is small enough to fit on one workstation GPU but difficult enough to stress an atmospheric model: volcanic topography, trade-wind inversion, island wakes, lateral forcing, boundary-layer structure, and surface coupling all matter.

The second question is methodological. Recent code models can make repository-scale edits, run test harnesses, and use shell environments, but safety-critical scientific software cannot be treated like an ordinary feature backlog. A model that silently invents a formula, citation, unit conversion, or validation gate can move quickly in the wrong direction. This project therefore used a collaborative AI-agent process in which work was not considered complete until a proof object existed on disk. Claude Opus 4.7 acted as long-context manager and reviewer. GPT-5.5 Codex acted as implementation worker and critic. Separate tester and reviewer roles challenged numerical, performance, and completion claims. The human principal, Enric R.G., set scope, validation policy, operational constraints, and final acceptance authority.

This paper has four contributions.

1. We describe a WRF-compatible regional replay prototype implemented in Python/JAX with whole-state device residency for the high-frequency model loop, rather than an incremental directive port of legacy WRF.
2. We report the current corrected-physics workstation result: 732.63 s for the iteration-2 24 h Canary 3 km d02 pipeline and 22.26x d02-only apples-to-apples speedup over the 28-rank CPU WRF baseline on the same workstation. The iteration-1 result (708.32 s / 23.02x) and the pre-fix path (324.78 s / 50.20x) are retained as diagnostic history; the latter exposed the workflow's overclaim risk and the former exposed the cost of correct physics.
3. We document the multi-agent engineering method that produced the system, including sprint contracts, proof objects, Architecture Decision Records (ADRs), independent tester/reviewer roles, and rejection loops where the process corrected its own claims.
4. We report the current scientific blocker: side-by-side AEMET station verification remains outside tolerance versus CPU WRF even after partial skill recovery. The result is a fast, reproducible, device-resident prototype, not yet an operational replacement for WRF.

The final point is central. Performance is useful only when the proof object matches the claim. The original M7 closeout celebrated an inflated speedup and treated finite station scores as a milestone gate. Follow-up sprints corrected the denominator, found the skill regression, identified three coupled algorithmic defects, applied two fix iterations, and remeasured. This paper therefore uses the slower 732.63 s / 22.26x iteration-2 number as the current engineering claim and keeps the pre-fix faster number as a cautionary validation story.

Current limitations are explicit. The validation corpus is small: three V3 initial-condition checks plus one 20260521 side-by-side station day. A single-day verification cannot characterise seasonal or regime-dependent behaviour, so the skill result is reported as a preliminary diagnostic case study rather than a physical-validity proof. The d02 system is replay-driven from retained Gen2 WRF side histories rather than live AIFS ingestion. Iteration 2 widened the lateral boundary from a width-1 strip to WRF's `spec_bdy_width=5` and added hourly land-surface refresh from retained Gen2 wrfouts, so those iteration-1 blockers are no longer current; the remaining blockers are surface-flux magnitude coupling that drives T2 overshoot, residual theta-guard saturation, the data-replay nature of the hourly land refresh (it is not a prognostic Noah-MP scheme), and no independent human numerical-methods review. These are release blockers, not footnotes.

## 2. Background and Related Work

### 2.1 WRF and ARW

The ARW core solves a fully compressible, non-hydrostatic system in flux form on a terrain-following dry hydrostatic pressure coordinate \cite{skamarock2019description}. Prognostic thermodynamic and moisture variables are held at mass points, while velocities are staggered on an Arakawa C grid. The time integrator is split explicit: a third-order Runge-Kutta outer step advances meteorological modes, while acoustic substeps handle fast pressure waves at a smaller step size. This numerical structure is attractive for regional forecasting but awkward for GPUs because the algorithm mixes horizontally coupled stencils, vertically implicit solves, boundary relaxation, and physics tendencies with different memory-access patterns.

The target operational physics family follows conventional WRF choices: Thompson microphysics \cite{thompson2008explicit}, MYNN planetary-boundary-layer closure \cite{nakanishi2006numerical}, RRTMG radiation \cite{iacono2008radiative}, and Noah/Noah-MP style land-surface behavior \cite{niu2011noah}. In this paper, "WRF-compatible" does not mean bitwise reproduction of every Fortran floating-point operation. It means a disciplined attempt to preserve useful ARW interfaces, state variables, boundary behavior, validation savepoints, and meteorological outputs. The project constitution rejects a line-by-line Fortran port as the architecture. The validation strategy instead requires per-operator parity where it is useful, physical invariants where they are decisive, and forecast-skill evidence where the system must make external claims.

### 2.2 GPU NWP and Regional Models

GPU acceleration in atmospheric modeling has followed several paths. Directive-based approaches use OpenACC, CUDA Fortran, or similar techniques to move parts of established Fortran codes to accelerators. Those approaches preserve a large existing model but often leave unported physics, boundary conditions, or I/O routines on the host. WRF-specific acceleration history includes microphysics and workflow modernization efforts; for example, ADIOS2 work targets WRF I/O and streaming rather than whole-forecast device residency \cite{fredj2023adios2wrf}. AceCAST represents a commercial WRF-acceleration line; vendor documentation reports meaningful WRF acceleration, but the public citation is not a peer-reviewed end-to-end benchmark and is treated here as context rather than as a hard comparator \cite{tempoquest2025acecast}.

Regional GPU NWP did not begin with this work. MeteoSwiss COSMO-CH and later ICON-CH systems are important precedents for GPU-enabled operational regional forecasting, and recent ICON GPU work documents the operational migration context and COSMO-1E/ICON-CH1-EPS verification comparisons \cite{fuhrer2026icon,lapillonne2026benchmarking}. This paper therefore makes no claim to be the first GPU regional NWP system. Its narrower claim is workstation-scale JAX/XLA execution with the high-frequency d02 state resident on one consumer GPU.

Domain-specific-language approaches separate scientific stencil expression from backend code generation. Pace rewrote the FV3 dynamical core in Python using GT4Py and DaCe \cite{dahm2023pace,bennun2019dace,whitaker2023gt4py,paredes2023gt4py}. ICON-exclaim and operational ICON GPU work show that large weather centers can migrate production NWP to GPU systems \cite{fuhrer2026icon,lapillonne2026benchmarking}. SCREAM represents the clean-slate C++/Kokkos path at exascale \cite{bertagna2024scream}. NIM is an earlier native-GPU precursor \cite{govett2017parallelization}. These systems differ in model equations, grid, hardware class, institution, and maturity. They establish that GPU NWP is real and that production-quality ports require more than kernel translation.

The brief-derived comparator rows used for this related-work framing are staged in `publish/tables/comparators.md`.

### 2.3 ML Weather Models

The recent ML weather-model literature changes the background for speed and skill. GraphCast, Pangu-Weather, FourCastNet, GenCast, Aurora, NeuralGCM, Stormer, and AIFS all show that data-driven or hybrid global forecasting systems can produce very fast and skillful forecasts under the right evaluation regime \cite{lam2023graphcast,bi2022pangu,pathak2022fourcastnet,price2023gencast,bodnar2024aurora,kochkov2023neuralgcm,nguyen2023stormer,lang2024aifs,lang2025update}. The present work is not an ML emulator. It is a numerical regional replay model that may eventually support ML by generating physically constrained training data, assimilating ML boundary products, or coupling learned parameterizations to a traditional solver.

The distinction matters. ML forecast systems are often evaluated on global reanalysis-style fields and medium-range lead times. A 3 km Canary Islands regional model must satisfy local boundary forcing, terrain, surface, and station-verification requirements. A fast regional core that lacks local skill is not operationally useful even if it is valuable as an engineering artifact.

### 2.4 AI Agents for Scientific Software

Repository-level AI coding has moved beyond autocomplete. SWE-bench measures whether language-model agents can resolve real GitHub issues \cite{jimenez2024swebench}. SWE-agent frames agent-computer interfaces as a route to automated software-engineering work \cite{yang2024sweagent}. Agent patterns such as orchestrator-worker and evaluator-optimizer loops describe how one model can decompose a task while another executes or critiques it \cite{anthropic2024effective}. Claude Code-style terminal harnesses and similar systems made it practical for agents to run tests, inspect files, edit code, and report evidence in a persistent repository \cite{anthropic2026claude}.

Scientific software changes the risk profile. A web-service bug may be caught by integration tests and user reports. A numerical-weather bug can produce plausible-looking fields while losing the correct forecast. The relevant question is not whether an AI can write code quickly. It is whether a multi-agent process can build, test, falsify, and revise scientific claims under explicit governance. Prior discussions of AI authorship and accountability emphasize that human responsibility and disclosure remain central \cite{arxiv2026policy,pcmag2026arxiv,nature2024editorial,schmidt2025senior}.

### 2.5 Chronology of the Release Claim

- M6 closeout: savepoint parity, coupled-step evidence, and initial M7 dispatch gates landed.
- M7 performance measurement: warm 1 h and pipeline timings established device-resident speed.
- Pipeline integration: the 20260521 d02 24 h pipeline wrote 24/24 readable wrfouts.
- Original celebration: an inflated 156.82x claim was recorded and then questioned.
- Honest-speedup correction: d02-only CPU timing gave 324.78 s / 50.20x for the pre-fix path.
- Skill regression discovery: side-by-side AEMET scoring returned `FAIL_SKILL_DIFF`.
- RCA convergence: Opus and Codex sprints identified theta/mu reset, surface/PBL flux disconnect, and radiation cadence defects.
- Algorithmic fix: the post-fix path preserved systems invariants and produced 708.32 s / 23.02x.
- Current state: `SKILL_IMPROVED_PARTIAL`, with iteration 2 in progress for remaining skill blockers.

## 3. Methods: AI Collaboration Model

### 3.1 Roles

The development process used a role taxonomy rather than a single assistant. The manager role, Claude Opus 4.7 with a long context window, owned sprint definition, repository-state synthesis, cross-sprint memory, ADR routing, and final closeout recommendations. The worker role, GPT-5.5 Codex, implemented scoped changes under a sprint contract. Tester and reviewer roles were separate agents instructed to challenge the worker's result, rerun commands, inspect proof objects, and refuse completion if the evidence did not support the claim. When a failure was hard to localize, the manager dispatched parallel debugger or critic sprints, often with one model taking an architectural angle and another taking an empirical bisection angle.

This division encoded different failure surfaces. The worker could move quickly inside a narrow file-ownership boundary. The tester could assume the implementation was wrong until proof said otherwise. The reviewer could reject a sprint even if the code ran. The manager could see repeated failure patterns and change the contract. The human principal remained the senior corresponding author and final accountable party, setting the scientific target, accepting or rejecting milestone state, and deciding what could be said externally.

### 3.2 Sprint Contracts and File Ownership

Every implementation sprint was launched from a contract. A contract stated the objective, non-goals, acceptance criteria, file ownership, proof objects, validation commands, branch name, and report token. Workers were not allowed to edit outside the owned paths. Governance files, memory, rules, and sprint contracts were treated as production assets and could not be changed directly without a patch protocol. The contracts also avoided cross-worker collisions: two active workers could not edit the same core files unless an interface had been frozen.

A condensed excerpt from the `m7-honest-speedup-skill-diff` contract shows the style:

```yaml
sprint_id: 2026-05-27-m7-honest-speedup-skill-diff
objective:
  - isolate CPU d02-only timing from existing Gen2 records
  - compare GPU and CPU wrfouts against AEMET stations
acceptance:
  - emit cpu_per_domain_wall_clock.json
  - emit honest_speedup_table.json
  - emit gpu_vs_cpu_skill_diff.json
  - write a verdict memo with publication-ready YES / NEEDS-CAVEAT / NO
hard_rules:
  - no fresh CPU WRF runs
  - taskset -c 0-3
  - be ruthless with the speedup denominator
failure_gate:
  - amend M7 closeout if speedup < 4x or GPU skill is materially worse
```

The useful feature of the contract pattern is that "done" becomes auditable. A sprint that claims zero device-to-host transfer must produce an Nsight or equivalent transfer audit. A sprint that claims forecast skill must produce station or ensemble metrics. A sprint that fails must write a blocker report rather than silently changing the goal.

### 3.3 Claim Types and Required Proof

| Claim type | Required proof object | Example in this paper |
|---|---|---|
| Performance | timing/profiler JSON with denominator definition | `post_fix_speedup.json` and `pipeline_run_20260521.json` |
| Transfer residency | D2H/H2D audit with window definition | `d2h_audit_v2.json` |
| Restart correctness | restart comparator output | `restart_continuity.json` and `restart_in_pipeline.json` |
| Savepoint or operator correctness | WRF/JAX savepoint or parity comparator | `proof_coupled_step_parity.json` and `proof_fix_validation.json` |
| Physical stability | finite/bounds/invariant JSON | `post_fix_bounds.json` and `invariant_preservation.json` |
| Operational skill | side-by-side CPU/GPU/observation scoring | `gpu_vs_cpu_skill_diff.json` and `post_fix_skill_diff.json` |
| Release readiness | closeout memo plus audit script | `MILESTONE-M7-CLOSEOUT-AMENDMENT.md` and `scripts/m7_publication_audit.sh` |

This table was added because several failures came from matching the wrong proof to the claim. Finite station scores are a measurement, not a skill-equivalence proof. A warm wall-clock proves speed under a window definition, not meteorological usefulness. Bitwise agreement at one step is useful lower-level evidence, not a 24 h forecast gate.

### 3.4 ADRs and Proof Objects

Architectural decisions were recorded as ADRs when they affected state layout, validation mode, precision, profiling, or release claims. The project distinguished validation mode and operational mode. Validation mode could emit savepoints, use stricter precision, and carry WRF scratch fields to support comparison against Fortran. Operational mode had to preserve the constitutional invariant of no host/device transfer inside the timestep loop, and it could fuse operators or drop validation-only scratch if evidence showed that the operational forecast stayed within the accepted envelope.

Proof objects were ordinary files: JSON measurements, Markdown verdicts, logs, and reports. They were not replaced by chat summaries. Examples used here include `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json`, `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json`, `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json`, `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json`, `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`, `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json`, `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json`, and `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/invariant_preservation.json`.

### 3.5 Rejection Loop Example

The 20260509 M6c mu-regression sprint is a representative rejection loop. Its contract hypothesis was that operational and validation initialization differed at step 2. The worker found the opposite: step-2 and step-5 bitwise parity were already clean, while the remaining raw failure occurred at step 10 with nonfinite `mu`/`theta`. The sprint localized the first nonfinite to RK1 acoustic substep 1, tested scoped scratch/core variants, reverted variants that failed, and reported `BLOCKED` rather than declaring a fix. That blocked report changed the next contract. It also preserved useful evidence: 20260521 step 10 remained at 0.0 delta, B6 coupled-step parity still held, and the failed hypothesis was removed from the active explanation set.

The D2H performance loop had a different shape. The initial trace reported inter-kernel device-to-host transfers and returned `BLOCKED-D2H`. A follow-up recaptured the profile window with explicit markers and separated pre-kernel boundary activity from the compiled forecast loop. The final audit reported zero inter-kernel D2H bytes. The method worked because both the failure and the correction were anchored to files rather than conversational confidence.

## 4. Methods: Numerical Port

The numerical port follows ARW structure but not the legacy software architecture. State variables are stored as JAX arrays with explicit staggering conventions. The operational d02 grid for the main measured case has mass shape `(44, 66, 159)` and WRF staggered extent `(45, 67, 160)`. The timestep is 10 s, with 360 RK steps per forecast hour, RK order 3, and 10 acoustic substeps. The implementation uses `jax.lax.scan` to express the time loop as a compiled graph rather than a Python loop. The warm timing window measures `run_forecast_operational` plus `block_until_ready`, excluding replay-case construction. The grid and timing values are recorded in `wall_clock.json` and `reproducibility_v2.json`.

The core design principle is whole-state device residency. Dry-air mass, pressure, geopotential, staggered winds, water species, surface fluxes, and boundary side histories remain in GPU memory through the operational loop. Step-boundary I/O is allowed for output and restart handling; inter-kernel transfer inside the forecast loop is not. The final D2H audit classifies this distinction explicitly: broad-window transfers were pre-kernel copies or boundary activity, while inter-kernel D2H inside the compiled forecast loop was zero.

The validation history used a bottom-up savepoint ladder. In M6, the project shifted from RMSE tuning toward direct WRF small-step instrumentation. Coefficient, tridiagonal, scratch-state, acoustic-recurrence, dycore-step, and coupled-step parity sprints progressively reduced ambiguity about where the JAX path differed from WRF. At M6 close, the manager recorded B6 savepoint parity and multi-step CPU parity on the 20260521 case at 0.0 bitwise for 2, 5, and 10 steps. The bitwise statement applies to the compared fields at the relevant step boundaries in validation/comparator mode; it does not imply long-range operational skill or full physics equivalence.

Precision is fail-closed. Hydrostatic mass and pressure-gradient-sensitive paths use FP64 where the validation history demands it. Local physics fields and moisture species may be FP32 only when allowed by policy and evidence. No downcast is production-safe merely because it is faster. This separation matters because JAX/XLA can produce excellent fused kernels, but a numerically unstable fused kernel is still wrong.

The current single-GPU implementation includes a halo placeholder for future multi-GPU work. The paper makes no multi-GPU scaling claim. Boundary forcing in the measured d02 pipeline uses Gen2 d02 hourly side-history replay rather than direct live AIFS ingestion. AIFS is the planned IC/BC source for the broader operational path \cite{lang2024aifs,lang2025update}, but the current result should be understood as a replay-driven Canary d02 system.

### 4.1 Validation Pyramid

The project uses a four-tier validation pyramid. Tier 1 is micro-fixture and savepoint parity. It is intentionally local and strict. When a single operator claims to reproduce a WRF Fortran expression, the comparison can require identical shapes, units, staggering, and, where practical, bitwise equality. This tier is useful for coefficient generation, tridiagonal-solve components, acoustic-substep recurrence checks, and boundary pack/unpack behavior. It is not, by itself, a forecast-skill gate.

Tier 2 is physical-invariant validation. This includes finite-value checks, basic bounds, dry-air mass behavior, tracer positivity, and water-budget checks when physics is active. Tier 2 exists because a model can match a local savepoint and still produce an impossible state after coupling. It also prevents "performance fixes" that silently introduce NaNs, negative water species, or pressure states outside the physical envelope.

Tier 3 is short-run trajectory behavior. The purpose is not to require long-run bitwise identity, which is a poor target for chaotic floating-point systems. The purpose is to show that short integrations diverge in a controlled way relative to a reference trajectory and that numerical changes do not produce explosive growth at the first few steps.

Tier 4 is the operational statistical gate. The project originally framed Tier 4 around ensemble consistency ideas such as PyCECT \cite{milroy2018ensemble}; for the M6/M7 Canary work, the practical surface gate was RMSE and station scoring on T2, U10, and V10 against CPU WRF and AEMET observations. Object-based and neighborhood precipitation verification methods such as FSS and SAL remain planned tools for precipitation-focused milestones \cite{roberts2008scale,wernli2008sal}. The skill regression and partial recovery reported in this draft are Tier 4 evidence.

## 5. Methods: Physics Suite

The prototype contains selected implementations from the target operational physics families: Thompson-style cloud microphysics, MYNN-style boundary-layer mixing, RRTMG radiation with cadence control, and Noah/Noah-MP-like surface state. The physics implementation is not presented as a validated replacement for the full WRF physics suite. It is the physics path present in the measured pipeline and the current source of remaining uncertainty in the skill results.

The pre-fix path had three coupled defects. First, a guard branch reset `theta`, `mu`, `mu_total`, and `mu_perturbation` to pre-RK values every timestep, discarding the prognostic update. Second, `surface_adapter` computed fluxes but MYNN received zero bottom-boundary flux arrays. Third, `DailyPipelineConfig.radiation_cadence_steps` defaulted to 999999, so RRTMG was not invoked in the 8640-step 24 h integration.

The post-fix path changed all three: theta and mu now flow through RK3 with inline bounded guards, `surface_adapter` runs before `mynn_adapter` and feeds its computed fluxes into the PBL bottom boundary, and radiation cadence is 180 steps, so RRTMG runs 48 times in a 24 h integration. The fix preserved systems invariants and improved 6 of 9 aggregate skill metrics, but it did not close the skill gap. Remaining physics limitations are concrete: the lower-column theta guard envelope still saturates maximum diurnal warming, land/surface state fields such as `t_skin`, `SST`, `SMOIS`, `SH2O`, and `TSLB` remain frozen at the initial condition, and boundary forcing uses a width-1 strip rather than the WRF width-5 relaxation zone.

## 6. Hardware and Software Setup

The measured target workstation uses a single NVIDIA GeForce RTX 5090 with 32607 MiB reported by `nvidia-smi` and CUDA device `cuda:0` \cite{nvidia2025geforce}. The 3 km d02 measured grid is `(44, 66, 159)` at 3000 m horizontal spacing. The 1 km memory audit used a derived full-domain synthetic state and reported 7278 MiB `nvidia-smi` memory used after a warm one-step feasibility probe, leaving approximately 78 percent of the reported GPU memory unused for that probe. This is not a peak allocator trace and is not a full 1 km forecast validation.

The runtime path is Python and JAX/XLA \cite{jax2018github,frostig2018tracing}. The current publication environment reports Python 3.13.11, JAX 0.10.0, jaxlib 0.10.0, CUDA toolkit 13.1.115, NVIDIA driver 595.71.05, and Linux 6.17.0-29-generic x86_64. The project package itself currently declares Python `>=3.10` and `jax>=0.4`; the public release must pin the exact runtime manifest. The CPU comparison baseline is WRF v4.7.1 running with 28 MPI ranks on the same workstation. Sprint-side processing used `taskset -c 0-3`, leaving cores 4-31 for CPU WRF comparison jobs when those jobs are active.

### 6.1 Canary Workflow and Verification Data

The measured M7 run is a d02 replay case, not a full live forecast cycle. It starts from a retained Gen2 WRF run, constructs a JAX replay state, advances the GPU operational path for the d02 domain, writes hourly `wrfout`-style NetCDF products, and evaluates station scores. The replay structure was a deliberate isolation choice. It allowed the project to test the GPU core against a known Gen2 source without simultaneously solving raw AIFS ingestion, nested d01 production, and retention-policy gaps.

The AEMET verification scaffold joins forecast values to station observations and reports BIAS, MAE, and RMSE for T2, U10, and V10 \cite{aemet2026observations}. For the 20260521 side-by-side comparison used here, the common valid-time range was 2026-05-21 19:00 UTC to 2026-05-22 18:00 UTC. The scoring report contains 73 station IDs, 24 common valid hours, and 1747 joined station-time rows. CPU and GPU outputs were passed through the same scoring code; only the wrfout source differed.

The station verification is still incomplete. It does not include a robust precipitation event set, a multi-day seasonal sample, or object/neighborhood precipitation metrics. It also does not replace grid-to-grid Tier 4 comparisons against Gen2 or WRF fields. The current station scaffold is sufficient to reject an operational replacement claim because the GPU remains materially worse than CPU WRF on the aggregate surface variables. It is not sufficient to diagnose every root cause by itself.

## 7. Results: Performance and Systems Evidence

Table 1 separates the pre-fix diagnostic path from the current post-fix corrected-physics path. The current headline result is the post-fix row group.

The consolidated pre-fix, iteration-1, and iteration-2 performance table is staged in `publish/tables/performance_evolution.md`.

| System state | Claim | Value | Proof object |
|---|---|---:|---|
| Iteration-2 path (current headline) | 24 h d02 pipeline wall time | 732.63 s | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json` |
| Iteration-2 path (current headline) | 24 h forecast-only wall time | 687.90 s | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json` |
| Iteration-2 path (current headline) | Apples-to-apples d02-only speedup | 22.26x | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_speedup.json` |
| Iteration-1 path (predecessor) | 24 h d02 pipeline wall time | 708.32 s | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json` |
| Iteration-1 path (predecessor) | 24 h forecast-only wall time | 700.73 s | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json` |
| Iteration-1 path (predecessor) | CPU d02-only timing denominator | 16305 s | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` |
| Iteration-1 path (predecessor) | Apples-to-apples d02-only speedup | 23.02x | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` |
| Iteration-1 path (predecessor) | Full five-domain CPU aggregate framing | 63.39x, not apples-to-apples | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` |
| Pre-fix diagnostic path | 24 h d02 pipeline wall time | 324.78 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` |
| Pre-fix diagnostic path | 24 h forecast-only wall time | 310.27 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` |
| Pre-fix diagnostic path | Apples-to-apples d02-only speedup | 50.20x | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` |
| Pre-fix diagnostic path | Full five-domain CPU aggregate framing | 138.24x, not apples-to-apples | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` |
| Shared systems evidence | Warm 1 h d02 forecast wall time | 5.71 s | `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json` |
| Shared systems evidence | Three-run warm reproducibility CV | 0.42 percent | `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json` |
| Shared systems evidence | Inter-kernel D2H inside forecast loop | 0 copies, 0 bytes | `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` |
| Shared systems evidence | Restart continuity | max delta 0.0 | `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json` |
| Shared systems evidence | Pipeline wrfout inventory | 24/24 files readable | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/wrfout_inventory.json` |
| Shared systems evidence | 1 km one-step memory probe | 7278 MiB of 32607 MiB | `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json` |

The current speedup should be read conservatively. The selected GPU number is the post-fix d02 pipeline wall time for the 20260521 case. The selected CPU denominator is the de-duplicated d02-only WRF timing for the same source run family. The five-domain aggregate CPU framing is reported because it appears in earlier proof objects, but it includes d01-d05 CPU work that the single-domain GPU d02 pipeline did not perform. It is not the headline comparison.

Cold compilation remains operationally relevant. The 1 h profile recorded cold JIT compile-inclusive walls of 102.58 s for the 20260509 case and 106.18 s for the 20260521 case. The daily 24 h pipeline amortizes compile cost across hourly chained forecasts, but a production deployment must still manage compile-cache invalidation across code, driver, JAX, and shape changes. The current result is strongest as a warm-run throughput result with reproducibility evidence, not as a complete operations-SLA result.

The D2H result is an architectural proof. The audit reports zero D2H inter-kernel bytes inside the XLA module window for `jit_run_forecast_operational`. This supports the project's central rule: no host/device transfer inside timestep loops unless explicitly approved and documented. It also distinguishes this design from partial ports where an unported scheme pulls full fields back to the host at each coupling point.

## 8. Results: Forecast Quality and Skill

### 8.1 Pre-fix Skill Regression Discovery

The pre-fix forecast-quality result was negative. Side-by-side scoring was performed on 73 AEMET station IDs over 24 common valid hours, producing 1747 joined station-time rows. GPU and CPU wrfouts were evaluated through the same `gpuwrf.validation.forecast_vs_obs` scaffold. Table 2 separates the pre-fix failure from the post-fix partial recovery.

The full CPU, pre-fix GPU, iteration-1 GPU, and iteration-2 GPU BIAS/MAE/RMSE matrix is staged in `publish/tables/skill_evolution.md`.

| System state | Variable | CPU WRF RMSE | GPU RMSE | Relative change vs CPU | Verdict |
|---|---|---:|---:|---:|---|
| Pre-fix diagnostic path | T2 | 2.15 K | 7.86 K | +266 percent | material regression |
| Pre-fix diagnostic path | U10 | 2.31 m s-1 | 11.31 m s-1 | +390 percent | material regression |
| Pre-fix diagnostic path | V10 | 2.75 m s-1 | 9.44 m s-1 | +243 percent | material regression |
| Post-fix corrected-physics path | T2 | 2.15 K | 8.85 K | +312 percent | still outside tolerance |
| Post-fix corrected-physics path | U10 | 2.31 m s-1 | 6.75 m s-1 | +193 percent | improved, still outside tolerance |
| Post-fix corrected-physics path | V10 | 2.75 m s-1 | 7.23 m s-1 | +163 percent | improved, still outside tolerance |

The pre-fix regression was broad: GPU was materially worse on every metric of every variable in the aggregate comparison, and the report verdict was `FAIL_SKILL_DIFF`. The failure was not just a station-scorer artifact. A separate L2 d02 replay validation on the same 3 km grid but with L2-d01 boundary forcing produced `L2_D02_BOUNDED_FAIL`: T2 RMSE 4.07 K against a 3.0 K threshold, U10 RMSE 10.78 m s-1 against 7.5 m s-1, and V10 RMSE 7.83 m s-1 against 7.5 m s-1.

### 8.2 Root-Cause Analysis

After the first draft of this paper was written, two parallel root-cause-analysis sprints landed and converged. An Opus architectural audit and a Codex empirical bisection independently identified three coupled defects in the operational forecast path.

First, the production guard branch overwrote post-RK3 `theta`, `mu`, `mu_total`, and `mu_perturbation` with pre-step values. The RK3+acoustic advance was being discarded each step. Independent corroboration came from on-disk bounds-check artifacts that recorded `theta_lower_30_max_k` identical to seven decimal places across all 24 hourly snapshots of the L2 d02 replay run.

Second, surface fluxes were computed but not applied to the atmosphere. The MYNN PBL adapter received zero bottom-boundary heat, moisture, and momentum flux inputs. `surface_adapter` computed `theta_flux`, `qv_flux`, `tau_u`, and `tau_v`, but those values did not feed the PBL bottom boundary in the correct order.

Third, radiation cadence was effectively disabled in the pre-fix path. `DailyPipelineConfig.radiation_cadence_steps` defaulted to 999999, so `rrtmg_adapter` was never invoked in the 8640-step 24 h integration. The post-fix path changes this cadence to 180.

The Codex bisection also surfaced a max-14-category `LU_INDEX` mismatch between GPU and Gen2 wrfouts at lead 1 h. A follow-up audit confirmed that in-memory `LU_INDEX` from `wrfinput_d02` matched Gen2 exactly; only the `wrfout_writer` fallback collapsed land cells to category 2 and water to 17 because the GPU `State` carries no `LU_INDEX` field. The forecast physics used the correct categories. `LU_INDEX` is therefore a publication-quality cleanup target for output, not the main root cause of the skill regression.

### 8.3 Post-fix Partial Recovery

A combined fix sprint applied all three algorithmic changes: theta and mu now flow through RK3 with inline bounded guards, `surface_adapter` runs before `mynn_adapter` and feeds computed fluxes into the PBL bottom boundary, and radiation cadence is 180. The post-fix sprint produced `SKILL_IMPROVED_PARTIAL`.

All M6 and M7 systems invariants were preserved: 20260521 multi-step parity step 2 remained 0.0 bitwise, B6 savepoint parity was preserved, inter-kernel D2H remained 0 bytes, and restart bitwise continuity passed. Six of nine T2/U10/V10 aggregate metrics improved versus the pre-fix GPU baseline, mainly wind metrics. T2 worsened because the inline theta envelope for the lower 30 levels still saturates the diurnal warming maximum. All three variables remain outside the pre-declared 20 percent tolerance against CPU WRF.

The post-fix wall-clock cost is 708.32 s end-to-end for 24 h, with 700.73 s forecast-only. The corrected current speedup is 23.02x apples-to-apples d02-only. This remains comfortably above the project's initial 4x to 8x exploratory target, but the speed is not a release-quality operational result until the skill blockers are closed.

The publication claim is therefore specific: the whole-state-resident architecture remains viable under current systems evidence, and the validation/fix discipline can localize coupled defects under operational scoring. The forecast is improving but is not yet a WRF skill replacement.

### 8.4 Second iteration: partial wind recovery, T2 regression

A second fix sprint addressed the three remaining named defects from iteration 1: it widened the lower-30 theta envelope from 400 K to 450 K, added a Gen2 hourly land-state refresh path that reloads `t_skin`, `SST`, `SMOIS`, `SH2O`, and `TSLB` from retained CPU wrfouts at each output boundary, and packed a WRF-ordered 5-row lateral boundary strip (`spec_bdy_width=5`, `relax_zone=4`) replacing the iter-1 outermost-row pack. Verdict: `BLOCKED`.

All systems invariants again held: step-2 multi-step parity 0.0 bitwise, B6 savepoint parity preserved, inter-kernel D2H = 0 bytes, restart bitwise PASS, and AgentOS validation green. The d02-only apples-to-apples speedup settled at 22.26x (down from iter-1's 23.02x due to the modest overhead of hourly land refresh and 5-row boundary handling) - still well above the 4x-8x target.

The post-iter-2 AEMET station scoring on the same 20260521 day, 73 stations, 24 valid hours, 1747 joined rows, produced:

The full per-variable comparison is staged in `publish/tables/skill_evolution.md`. The headline RMSE summary, comparing pre-fix, iteration-1, and iteration-2 GPU paths against the CPU WRF baseline, is:

| Variable | CPU WRF RMSE | Pre-fix GPU RMSE | Iteration-1 GPU RMSE | Iteration-2 GPU RMSE | Iter-2 vs CPU |
|---|---:|---:|---:|---:|---:|
| T2 (K) | 2.15 | 7.86 | 8.85 | **10.80** | **+403 percent** |
| U10 (m s-1) | 2.31 | 11.31 | 6.75 | **7.24** | **+214 percent** |
| V10 (m s-1) | 2.75 | 9.44 | 7.23 | **7.62** | **+177 percent** |

Compared to the pre-fix path, iteration 2 substantially reduces wind RMSE (U10 +390 to +214 percent of CPU; V10 +243 to +177 percent of CPU), confirming that the named iteration-1 fixes (theta/mu reset removal, surface-PBL wiring, RRTMG enablement) and the iteration-2 boundary widening together do carry real meteorological information into the wind field. T2 RMSE worsens monotonically across the three GPU paths (7.86 to 8.85 to 10.80 K) and is the variable that the present coupling stack handles worst. Comparing iteration 2 to iteration 1 directly: RMSE worsens slightly on all three variables (T2 +22 percent, U10 +7 percent, V10 +5 percent), while bias magnitudes are mixed. The strongest engineering claim from iteration 2 is therefore not "RMSE improved" but "the named iteration-1 blockers (boundary width 1, frozen land state) are removed and the M7 systems invariants still hold." The RMSE direction shows that releasing those blockers exposed an underlying surface-flux magnitude coupling defect that iteration 1's tighter envelope had been masking.

The mechanism is consistent across the proof objects: with the envelope widened from 400 K to 450 K, lower-column theta can climb further during daytime heating, but surface-flux magnitudes from the current `surface_adapter` plus MYNN coupling over-deposit heat into the bottom level, and the diurnal warming overshoots rather than saturating. All three variables remain outside the pre-declared 20 percent tolerance. The publication therefore continues to reject any operational replacement claim. The remaining defect is narrower: not a missing radiation source, a discarded RK3 advance, a width-1 boundary, or a frozen land state, but a surface-flux magnitude or sign-coupling issue in the iteration-2 path. The proof-object backbone for iteration 2 - `post_iter2_skill_diff.json`, `post_iter2_speedup.json`, `invariant_preservation_iter2.json` - is preserved on disk for that follow-up.

## 9. Discussion

The project is best understood as two intertwined experiments: one in GPU-native NWP architecture and one in AI-agent scientific-software production.

The architectural experiment supports the value of whole-state residency. The current 23.02x corrected-physics d02 throughput ratio is large enough to matter even after rejecting both the original 156.82x celebration number and the pre-fix 50.20x diagnostic path as current headline results. A single consumer GPU can run the measured 3 km Canary replay in minutes rather than CPU-WRF hours for the same d02 timing denominator. If the skill gap is resolved, this would make high-frequency local ensembles, repeated sensitivity tests, or rapid backfills more accessible than with a CPU-only operational path.

The result also shows why performance without validation can mislead. A forecast that is fast, finite, restartable, and repeatable can still be wrong. The original M7 closeout overinterpreted finite AEMET station scores. The scorer produced real measurements, but it had not compared GPU skill against CPU WRF. Once the correct side-by-side comparison was run, the claim changed immediately. The central scientific lesson is that proof objects must match the claim being made.

The AI-agent methodology helped because it made that correction possible. A single human plus autocomplete workflow might have moved from the inflated closeout directly to public communication. Here, the manager's later validation step launched an honest-speedup and skill-diff sprint before release. The process found a timing-denominator bug, a skill regression, and then a partial fix that improved winds while exposing remaining land/surface, boundary, and theta-guard defects. That does not make the process infallible. It shows that adversarial, proof-object-driven AI collaboration can create useful internal friction.

The method also has weaknesses. The manager was itself an AI system and initially made the celebration error. The validation discipline was stronger than ordinary chat-based coding, but it was not equivalent to an independent human numerical-methods review. Guard behavior, radiation cadence, and surface/PBL coupling were allowed to survive long enough to affect publication readiness. The project caught those issues, but late. A future workflow should require side-by-side CPU/GPU skill comparison before any closeout can use operational language.

One practical lesson is that AI agents benefit from narrow contracts but suffer when the contract names the wrong proxy. The pipeline-integration sprint correctly produced a working 24 h pipeline, hourly files, finite fields, station-score rows, and wall-clock evidence. Those were the requested artifacts. The later error was treating those artifacts as proof of operational skill. This is not a worker failure so much as a specification failure.

Another lesson is that cross-model disagreement is useful only when attached to files. Parallel critic sprints that merely argue would not have helped. The useful sprints wrote JSON tables, verdict markdown, command outputs, and failed hypotheses. The repository could then preserve both the positive result and the contradiction. That property matters for scientific software because many wrong paths are locally plausible.

The most defensible publication frame is therefore modest but valuable: a governed AI-agent process built a nontrivial GPU-native regional NWP replay prototype, produced a large corrected-physics speedup, and generated the evidence that prevented an overclaim. In a field where numerical trust matters more than demonstration speed, that self-correction is part of the result.

## 10. Limitations

The current system is single-GPU only. The halo interface is a placeholder, and no multi-GPU MPI or GPU-aware exchange result is claimed. The 1 km result is a memory-feasibility probe, not a full 1 km forecast validation.

The Canary workflow is replay-based. The GPU d02 system receives boundary side histories derived from existing Gen2 WRF output. That is acceptable for isolating d02 dynamics and physics while the port is validated, but it is not the same as running the full nested operational stack from raw AIFS IC/BC. A production system must either ingest AIFS and geog/static fields directly or document a reproducible bridge from Gen2 products to GPU state.

The validation corpus is too small. M6 used three V3 initial conditions for operational Tier-4 RMSE gates, and M7 side-by-side station scoring used one 24 h 20260521 case. That is enough to expose a serious problem and measure partial recovery; it is not enough to characterize seasonal or regime-dependent behavior. A full Tier-4 ensemble remains corpus-blocked until more Gen2 d02 CPU/GPU comparable pairs are retained or replayed.

The current physics path remains incomplete as an operational claim. Radiation cadence is no longer disabled in the post-fix path; it runs every 180 steps. The boundary forcing widens to `spec_bdy_width=5` in iteration 2, and the land/surface state is refreshed hourly from Gen2 wrfouts in iteration 2. The remaining blockers are now narrower: a surface-flux magnitude or sign-coupling issue that drives T2 overshoot once the theta envelope is widened to allow diurnal warming, residual upper-bound saturation in the theta guard, the hourly land-refresh path being a data replay rather than a prognostic Noah-MP scheme, and skill still outside the pre-declared 20 percent tolerance against CPU WRF. Microphysics admissibility and finite guards remain load-bearing in at least one diagnostic history.

The observation and verification path is narrow. AEMET station scoring on T2, U10, and V10 is useful and directly relevant to Canary operations, but it does not cover precipitation structure, cloud, radiation, vertical profiles, or regime-specific diagnostics. METplus-style verification, FSS, SAL, and a multi-day event corpus remain future work.

The authorship process needs external review before public release. AI systems wrote and reviewed much of the system, and this draft discloses that. The project has not yet had a truly independent human numerical-methods reviewer audit the code and paper claims. Enric R.G. remains responsible for final acceptance.

The release manifest is not yet frozen. This drafting environment reports Python 3.13.11, JAX 0.10.0, jaxlib 0.10.0, CUDA toolkit 13.1.115, NVIDIA driver 595.71.05, and Linux 6.17.0-29-generic x86_64. The final reproducibility package must pin the exact Python, JAX, jaxlib, CUDA, driver, XLA flags, git commit, and proof-object commit hashes.

Several citations remain release-quality checks rather than claims of final bibliographic perfection. The included BibTeX file parses locally, and this revision removes the unresolved Mollick TODO citation, but publisher metadata for several brief-derived entries should be rechecked before arXiv submission.

## 11. Reproducibility

The public code URL placeholder is `github.com/<TBD>`. The release package should include the final repository commit, a hardware manifest, a software-version manifest, and the proof-object directories listed below.

| Item | Current value |
|---|---|
| Public repository | `github.com/<TBD>` |
| Release commit | `TBD at public release` |
| Current revision branch | `worker/gpt/publication-revision-pass` |
| Paper post-fix framing commit | `c9ab7c0` |
| Skill-fix proof-object merge commit | `d14d76c` |
| Critique merge commit | `53fbf11` |
| Current dispatch commit | `093f93c` |

| Environment field | Value |
|---|---|
| Python | 3.13.11 |
| JAX | 0.10.0 |
| jaxlib | 0.10.0 |
| CUDA toolkit | 13.1.115 |
| NVIDIA driver | 595.71.05 |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB |
| OS | Linux 6.17.0-29-generic x86_64 |
| CPU pinning for publication checks | `taskset -c 0-3` |

The canonical proof-object manifest for this paper is:

- `.agent/decisions/MILESTONE-M6-CLOSEOUT.md`
- `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md`
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json`
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json`
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json`
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json`
- `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json`
- `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json`
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/invariant_preservation.json`
- `.agent/sprints/2026-05-27-publication-revision-pass/revision_decisions.md`
- `publication/draft/honesty_audit.md`

The lightweight audit command is:

```bash
taskset -c 0-3 bash scripts/m7_publication_audit.sh
```

The script checks the paper word count, BibTeX parseability through `bibtexparser`, cited-key integrity, non-ASCII characters in the publication files, required proof-object existence, and `scripts/validate_agentos.py`. Heavy GPU and CPU forecast reruns remain separate because they depend on the RTX 5090 and retained Gen2 CPU WRF corpus.

## 12. Author Contributions and AI Use Disclosure

This draft names Claude Opus 4.7 and GPT-5.5 Codex / OpenAI as AI systems, not as human authors. The reason is disclosure: both systems made substantial cognitive and engineering contributions to the repository and to this manuscript. Claude Opus 4.7 designed and managed sprint contracts, maintained long-horizon repository context, synthesized proof objects, drafted and revised this manuscript, and made the original closeout error that was later corrected. GPT-5.5 Codex implemented much of the repository code as worker, performed targeted debugging and critical-review sprints, generated proof objects under manager contracts, and contributed the empirical evidence base used here.

Enric R.G. defined the Canary Islands operational target, supplied and maintained the Gen2 CPU WRF baseline context, set validation gates and performance expectations, monitored the AI-agent process, and retains senior corresponding-author responsibility for final scientific acceptance and submission. All external publication responsibility rests with the human author. The AI systems cannot approve the final manuscript, hold legal accountability, or satisfy human-only authorship criteria.

This authorship framing is policy-sensitive. The briefed arXiv discussion emphasizes author responsibility for unchecked AI-generated content \cite{arxiv2026policy,pcmag2026arxiv}, while publisher policies such as Nature's do not treat AI tools as authors \cite{nature2024editorial}. For an arXiv preprint, the current byline is a transparent AI-contribution disclosure. If a target venue requires only human authors, the byline should be changed to Enric R.G. alone, with Claude Opus 4.7 and GPT-5.5 Codex / OpenAI moved to acknowledgements plus this AI-use disclosure.

## 13. Acknowledgements

The project depends on the WRF and NCAR modeling community, ECMWF AIFS context, AEMET station observations, NVIDIA GPU tooling, and Enric R.G.'s prior Gen2 operational Canary forecasting system. The authors also acknowledge the repository's internal reviewer and tester roles, which forced the correction from pre-fix performance celebration to current corrected-physics reporting.

## References

References will be rendered from `publication/draft/references.bib` during LaTeX conversion.
