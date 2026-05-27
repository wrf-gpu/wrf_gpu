# Whole-State Device Residency for Workstation-Scale NWP: A JAX-Native WRF v4 Port Engineered by Collaborative AI Agents

**Preprint categories:** physics.ao-ph primary; cs.LG secondary; cs.SE tertiary.

**Authors:** Claude Opus 4.7; GPT-5.5 Codex / OpenAI; Enric R.G.

## Abstract

We report a first complete draft result from an agent-built, JAX-native rewrite of the Weather Research and Forecasting (WRF) model's regional dynamical-core and operational-physics path for a Canary Islands 3 km domain. The system keeps the high-frequency forecast state resident on a single NVIDIA RTX 5090 GPU and compiles the operational time loop through JAX/XLA. On the 20260521 case, a warm 1 h forecast ran in 5.71 s, and the 24 h d02 pipeline completed in 324.78 s. Against the same-workstation 28-rank CPU WRF d02 timing, the corrected apples-to-apples throughput ratio is 50.20x, with zero inter-kernel device-to-host transfer in the forecast loop, bitwise restart continuity, repeatability pass, and a 1 km memory audit showing 7278 MiB peak process memory on a 32607 MiB RTX 5090. The methodological contribution is the development process: a manager-orchestrated, multi-agent AI software-engineering system using sprint contracts, proof objects, ADRs, cross-model review, and explicit failure reports. This same validation discipline found a publication-blocking error after an initial performance celebration: side-by-side AEMET station scoring showed material skill regression relative to CPU WRF on T2, U10, and V10, with RMSE increases of 243-390 percent depending on variable. We therefore frame the result as a fast, reproducible, device-resident prototype whose forecast skill is not yet operationally acceptable.

## 1. Introduction

Regional numerical weather prediction (NWP) is still one of the demanding production workloads in scientific computing. It combines sparse operational windows, irregular regional boundaries, terrain-following coordinates, coupled physical parameterizations, and a long tail of input/output and verification requirements. The Weather Research and Forecasting model (WRF), and in particular its Advanced Research WRF (ARW) core, remains widely used for mesoscale forecasting because it provides a mature non-hydrostatic dynamical core, many tested physics options, and a large operational community \cite{skamarock2019description,powers2017weather}. The same maturity makes WRF difficult to port wholesale to modern GPUs. Its source architecture was built around Fortran, MPI, host-resident control flow, and many scheme-specific memory layouts. A directive-based port can accelerate important pieces, but the full regional forecast remains constrained when any unported component forces repeated host/device movement.

The project described here asks a narrower question than "Can WRF be mechanically ported?" The question is whether a WRF-compatible regional forecast path can be rebuilt in a GPU-native form while preserving enough physical and verification behavior to make the result scientifically inspectable. We target a single operational domain first: the Canary Islands at 3 km grid spacing, with a 1 km feasibility path. The domain is small enough to fit on one workstation GPU but hard enough to stress the atmospheric model. Its volcanic topography, trade-wind inversion, island wakes, and intermittent Saharan Air Layer episodes are precisely the kind of phenomena for which a regional model must handle steep terrain, lateral forcing, boundary-layer structure, and surface coupling.

The second question is methodological. Recent code models can make repository-scale edits, run test harnesses, and use shell environments, but a safety-critical scientific codebase cannot be treated like an ordinary feature backlog. A model that silently invents a formula, a citation, a unit conversion, or a validation gate can move quickly in the wrong direction. This project therefore used a collaborative AI-agent process in which work was not considered complete until a proof object existed on disk. Claude Opus 4.7 acted as 1M-context manager and reviewer. GPT-5.5 Codex acted as the implementation worker and critic. Additional tester and reviewer roles challenged numerical claims, performance claims, and completion claims. The human principal, Enric R.G., set scope, validation policy, operational constraints, and final acceptance authority.

This paper has four contributions.

1. We describe a WRF-compatible regional forecast prototype implemented in Python/JAX with whole-state device residency for the high-frequency model loop, rather than an incremental directive port of legacy WRF.
2. We report corrected workstation-scale performance evidence: 324.78 s for a 24 h Canary 3 km d02 pipeline and a conservative 50.20x d02-only speedup over the 28-rank CPU WRF baseline on the same workstation.
3. We document the multi-agent engineering method that produced the system, including sprint contracts, proof objects, Architecture Decision Records (ADRs), independent tester/reviewer roles, and failure-mode examples where the process corrected its own claims.
4. We explicitly report the current scientific blocker: side-by-side AEMET station verification shows that the GPU forecast is materially less skillful than the CPU baseline. The result is not yet an operational replacement for WRF.

The final point is central. The system is fast and reproducible, but fast is not sufficient in NWP. The original M7 closeout celebrated an inflated speedup and treated finite station scores as a milestone gate. A follow-up honest-speedup and skill-difference sprint corrected the speedup denominator to 50.20x for the best d02 apples-to-apples comparison and showed that finite station scores were not evidence of skill parity. This paper keeps the engineering result and the validation failure in the same frame, because the ability to find and publish that failure is part of the methodological claim.

## 2. Background and Related Work

### 2.1 WRF and ARW

The ARW core solves a fully compressible, non-hydrostatic system in flux form on a terrain-following dry hydrostatic pressure coordinate \cite{skamarock2019description}. Prognostic thermodynamic and moisture variables are held at mass points, while velocities are staggered on an Arakawa C grid. The time integrator is split explicit: a third-order Runge-Kutta outer step advances meteorological modes, while acoustic substeps handle fast pressure waves at a smaller step size. This numerical structure is attractive for regional forecasting but awkward for GPUs because the algorithm mixes horizontally coupled stencils, vertically implicit solves, boundary relaxation, and physics tendencies with different memory-access patterns.

The operational physics suite used by the target system follows a conventional WRF family: Thompson microphysics \cite{thompson2008explicit}, MYNN planetary-boundary-layer closure \cite{nakanishi2006numerical}, RRTMG radiation \cite{iacono2008radiative}, and Noah/Noah-MP style land-surface behavior \cite{niu2011noah}. In this first draft, "WRF-compatible" does not mean bitwise reproduction of every Fortran floating-point operation. It means a disciplined attempt to preserve useful ARW interfaces, state variables, boundary behavior, validation savepoints, and meteorological outputs. The project constitution explicitly rejects a line-by-line Fortran port as the architecture. The validation strategy instead requires per-operator parity where it is useful, physical invariants where they are decisive, and forecast-skill evidence where the system must make external claims.

### 2.2 GPU NWP and Climate Models

GPU acceleration in atmospheric modeling has followed several paths. Directive-based approaches use OpenACC, CUDA Fortran, or similar techniques to move parts of an established Fortran code to accelerators. Those approaches preserve a large existing model but often leave unported physics, boundary conditions, or I/O routines on the host. AceCAST represents the commercial WRF-acceleration line and has reported 5-14x speedup ranges in the provided research brief \cite{tempoquest2025acecast}. That result is important context: it shows that WRF acceleration is real, but it also motivates whole-forecast residency when the target is an end-to-end regional pipeline rather than a kernel subset.

Domain-specific-language approaches separate scientific stencil expression from backend code generation. Pace rewrote the FV3 dynamical core in Python using GT4Py and DaCe, reporting approximately 3.5-4x speedups over a highly optimized CPU baseline in the briefed comparison \cite{dahm2023pace,bennun2019dace,whitaker2023gt4py}. ICON-exclaim and operational ICON GPU work report a 5.5x socket-to-socket speedup in the briefed benchmark, showing that large weather centers can migrate production NWP to GPU systems \cite{fuhrer2026icon,lapillonne2026benchmarking}. SCREAM represents the clean-slate C++/Kokkos path, reaching 1.26 simulated years per day at 3.25 km global cloud-permitting resolution on Frontier-class GPU resources \cite{bertagna2024scream}. NIM is an earlier native-GPU precursor, with up to 34x dynamics-only speedups in the cited brief \cite{govett2017parallelization}.

These systems are not direct competitors to the present work. They are different models, grids, hardware classes, and organizational contexts. They establish that GPU NWP is not speculative and that production-quality ports require more than kernel translation. Our narrower claim is workstation-scale regional throughput with complete high-frequency state residency in a Python/JAX implementation. We do not claim priority as the first GPU regional NWP system, the first Python weather model, or the first differentiable atmospheric code.

### 2.3 ML Weather Models

The recent ML weather-model literature changes the baseline for speed and skill. GraphCast, Pangu-Weather, FourCastNet, GenCast, Aurora, NeuralGCM, Stormer, and AIFS all show that data-driven or hybrid global forecasting systems can produce very fast and skillful forecasts under the right evaluation regime \cite{lam2023graphcast,bi2022pangu,pathak2022fourcastnet,price2023gencast,bodnar2024aurora,kochkov2023neuralgcm,nguyen2023stormer,lang2024aifs,lang2025update}. The present work is not an ML emulator. It is a numerical regional model that may eventually support ML by generating physically constrained training data, assimilating ML boundary products, or coupling learned parameterizations to a traditional solver.

This distinction matters for claims. ML forecast systems are often evaluated on global reanalysis-style fields and medium-range lead times. A 3 km Canary Islands regional model must satisfy local boundary forcing, terrain, surface, and station-verification requirements. A fast regional core that lacks local skill is not operationally useful, even if it is useful as an engineering artifact.

### 2.4 AI Agents for Scientific Software

Repository-level AI coding has moved beyond autocomplete. SWE-bench measures whether language-model agents can resolve real GitHub issues \cite{jimenez2024swebench}. Agent patterns such as orchestrator-worker and evaluator-optimizer loops describe how one model can decompose a task while another executes or critiques it \cite{anthropic2024effective}. Claude Code-style terminal harnesses and similar systems made it practical for agents to run tests, inspect files, edit code, and report evidence in a persistent repository \cite{anthropic2026claude}.

Scientific software changes the risk profile. A web-service bug may be caught by integration tests and user reports. A numerical-weather bug can produce plausible-looking fields while losing the correct forecast. We therefore treat AI-agent development itself as an object of study. The relevant question is not whether an AI can write code quickly. It is whether a multi-agent process can build, test, falsify, and revise scientific claims under explicit governance. Prior discussions of AI authorship and accountability emphasize that human responsibility and disclosure remain central \cite{arxiv2026policy,pcmag2026arxiv,nature2024editorial,schmidt2025senior}. The sprint contract also requested a Mollick et al. citation for this point, but the provided research briefs did not include enough bibliographic detail, so the manager review pass should add it as \cite{TODO_Mollick_AI_authorship}.

## 3. Methods: AI Collaboration Model

### 3.1 Roles

The development process used a small role taxonomy rather than a single assistant. The manager role, Claude Opus 4.7 with a 1M-token context window, owned sprint definition, repository-state synthesis, cross-sprint memory, ADR routing, and final closeout recommendations. The worker role, GPT-5.5 Codex, implemented scoped changes under a sprint contract. The tester and reviewer roles were separate agents instructed to challenge the worker's result, rerun commands, inspect proof objects, and refuse completion if the evidence did not support the claim. When a failure was hard to localize, the manager dispatched parallel debugger or critic sprints, often with one model taking an architectural angle and another taking an empirical bisection angle.

This division was not cosmetic. It encoded different failure surfaces. The worker could move quickly inside a narrow file-ownership boundary. The tester could assume the implementation was wrong until proof said otherwise. The reviewer could reject the sprint even if the code ran. The manager could see repeated failure patterns and change the contract. The human principal remained the senior corresponding author and final accountable party, setting the scientific target, accepting or rejecting milestone state, and deciding what could be said externally.

The author list follows that cognitive division. Claude Opus 4.7 is first because it orchestrated the project and drafted this manuscript. GPT-5.5 Codex / OpenAI is middle because it executed the bulk implementation and critical-review work. Enric R.G. is senior corresponding author because he defined the operational problem, supplied the Gen2 CPU baseline context, set the validation gates, and retains human responsibility for publication.

### 3.2 Sprint Contracts and File Ownership

Every implementation sprint was launched from a contract. A contract stated the objective, non-goals, acceptance criteria, file ownership, proof objects, validation commands, branch name, and report token. Workers were not allowed to edit outside the owned paths. Governance files, memory, rules, and sprint contracts were treated as production assets and could not be changed directly without a patch protocol. The contracts also avoided cross-worker collisions: two active workers could not edit the same core files unless an interface had been frozen.

The useful feature of the contract pattern is that "done" becomes machine-auditable. A sprint that claims zero device-to-host transfer must produce an Nsight or equivalent transfer audit. A sprint that claims forecast skill must produce station or ensemble metrics. A sprint that fails must write a blocker report rather than silently changing the goal. This mattered several times. The D2H performance sprint initially looked blocked by transfer counts, but parallel probes found a profiler-window placement bug rather than a model-residency violation. Without the proof-object discipline, the project would likely have launched code-fix sprints against the wrong problem.

### 3.3 ADRs and Proof Objects

Architectural decisions were recorded as ADRs when they affected state layout, validation mode, precision, profiling, or release claims. The project distinguished validation mode and operational mode. Validation mode could emit savepoints, use stricter precision, and carry WRF scratch fields to support comparison against Fortran. Operational mode had to preserve the constitutional invariant of no host/device transfer inside the timestep loop, and it could fuse operators or drop validation-only scratch if evidence showed that the operational forecast stayed within the accepted envelope.

Proof objects were ordinary files: JSON measurements, Markdown verdicts, logs, and reports. They were not replaced by chat summaries. Examples used in this paper include `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json`, `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json`, `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json`, `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json`, and `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`.

### 3.4 Failure Modes Caught by the Process

The process caught four important failure modes.

First, the D2H false alarm showed why profiler artifacts need careful window definitions. The initial trace appeared to violate the no-transfer rule. A follow-up sprint recaptured the window with explicit NVTX markers and classified pre-kernel boundary copies separately from inter-kernel forecast-loop transfer. The final audit found zero D2H inter-kernel bytes.

Second, the 20260509 theta-growth problem showed that bitwise agreement is not sufficient if both compared paths share a hidden guard or common error. A sprint designed to prove one hypothesis instead reported itself blocked when the evidence contradicted the contract. That prevented a premature close.

Third, an inflated speedup was corrected after the manager dispatched an honest-speedup sprint. The original denominator double-counted mirrored CPU WRF timing records from an unsuitable path. The corrected comparison uses de-duplicated CPU d02 timing from sibling `rsl.error.0000` and `rsl.out.0000` records, yielding 50.20x.

Fourth, the publication-blocking skill regression was caught by side-by-side AEMET scoring before public release. The pipeline sprint had only shown finite station scores for the GPU output. The amendment recognized that finite scores are measurements, not a skill claim. The honest-skill-diff sprint ran CPU and GPU outputs through the same scorer and showed large GPU degradation on T2, U10, and V10.

## 4. Methods: Numerical Port

The numerical port follows the ARW structure but not the legacy software architecture. State variables are stored as JAX arrays with explicit staggering conventions. The operational d02 grid for the main measured case has mass shape `(44, 66, 159)` and WRF staggered extent `(45, 67, 160)`. The timestep is 10 s, with 360 RK steps per forecast hour, RK order 3, and 10 acoustic substeps. The implementation uses `jax.lax.scan` to express the time loop as a compiled graph rather than a Python loop. The warm timing window measures `run_forecast_operational` plus `block_until_ready`, excluding replay-case construction.

The core design principle is whole-state device residency. Dry-air mass, pressure, geopotential, staggered winds, water species, surface fluxes, and boundary side histories remain in GPU memory through the operational loop. Step-boundary I/O is allowed for output and restart handling; inter-kernel transfer inside the forecast loop is not. The final D2H audit classifies this distinction explicitly: 25 D2H operations in the broad window were pre-kernel copies or boundary activity, while inter-kernel D2H was zero.

The validation history used a bottom-up savepoint ladder. In early M6 work, the project shifted from RMSE tuning toward direct WRF small-step instrumentation. Coefficient, tridiagonal, scratch-state, acoustic-recurrence, dycore-step, and coupled-step parity sprints progressively reduced ambiguity about where the JAX path differed from WRF. At M6 close, the manager recorded B6 savepoint parity at 0.0 bitwise and multi-step CPU parity on the 20260521 case at 0.0 bitwise for 2, 5, and 10 steps. Those facts are not enough for operational skill, but they are useful evidence that the port had passed several lower-level correctness gates.

Precision is fail-closed. Hydrostatic mass and pressure-gradient-sensitive paths use FP64 where the validation history demands it. Local physics fields and moisture species may be FP32 when allowed by policy and evidence. No new downcast is considered production-safe merely because it is faster. This separation is important because JAX/XLA can produce excellent fused kernels, but a numerically unstable fused kernel is still wrong.

The current single-GPU implementation includes a halo placeholder for future multi-GPU work. The paper makes no multi-GPU scaling claim. Boundary forcing in the measured d02 pipeline uses Gen2 d02 hourly side-history replay rather than direct live AIFS ingestion. AIFS is the planned IC/BC source for the broader operational path \cite{lang2024aifs,lang2025update}, but the current result should be understood as a replay-driven Canary d02 system.

### 4.1 Validation Pyramid

The project uses a four-tier validation pyramid. Tier 1 is micro-fixture and savepoint parity. It is intentionally local and strict. When a single operator claims to reproduce a WRF Fortran expression, the comparison can require identical shapes, units, staggering, and, where practical, bitwise equality. This tier is useful for coefficient generation, tridiagonal-solve components, acoustic-substep recurrence checks, and boundary pack/unpack behavior. It is not, by itself, a forecast-skill gate.

Tier 2 is physical-invariant validation. This includes finite-value checks, basic bounds, dry-air mass behavior, tracer positivity, and water-budget checks when physics is active. Tier 2 exists because a model can match a local savepoint and still produce an impossible state after coupling. It also prevents "performance fixes" that silently introduce NaNs, negative water species, or pressure states outside the physical envelope.

Tier 3 is short-run trajectory behavior. The purpose is not to require long-run bitwise identity, which is a poor target for chaotic floating-point systems. The purpose is to show that short integrations diverge in a controlled way relative to a reference trajectory and that numerical changes do not produce explosive growth at the first few steps. In this project, several failures were caught at this tier before they became performance claims.

Tier 4 is the operational statistical gate. The project originally framed Tier 4 around ensemble consistency ideas such as PyCECT \cite{milroy2018ensemble}; for the M6/M7 Canary work, the practical surface gate was RMSE and station scoring on T2, U10, and V10 against CPU WRF and AEMET observations. Object-based and neighborhood precipitation verification methods such as FSS and SAL remain planned tools for precipitation-focused milestones \cite{roberts2008scale,wernli2008sal}. The skill regression reported in this draft is a Tier 4 failure. It overrides lower-level successes for any operational claim.

The pyramid prevents one metric from doing too much. Bitwise parity is appropriate for a coefficient routine, not for claiming forecast equivalence. A fast wall-clock is appropriate for performance, not for claiming meteorological usefulness. Finite station scores are measurements, not evidence that the GPU model matches CPU WRF skill. The closeout amendment is essentially a validation-pyramid correction: the project had enough Tier 1, Tier 2, and performance evidence to celebrate an engineering milestone, but it lacked the side-by-side Tier 4 comparison needed for an operational statement.

## 5. Methods: Physics Suite

The prototype contains the operational physics families selected for the Canary target: Thompson-style cloud microphysics, MYNN-style boundary-layer mixing, radiation with cadence control, and Noah/Noah-MP-like surface state. The physics implementation is not presented here as a validated replacement for the full WRF physics suite. It is presented as the physics path present in the measured pipeline and as a current source of uncertainty in the skill regression.

The radiation cadence is a visible example of why this first draft must avoid operational overclaiming. The measured namelist records `radiation_cadence_steps=999999`. That means radiation is effectively disabled over the 24 h operational loop unless separately forced by initialization or diagnostics. This may or may not be a root cause of the skill regression, but it is a plausible suspect that must be resolved before an operational-replacement claim can stand.

Microphysics and surface guards also remain load-bearing in at least one diagnostic path. M6 closeout records microphysics admissibility and finite-or-origin guards as defense-in-depth, with a recommendation to verify later whether they become unreachable after deeper fixes. The project therefore reports stability and performance with guards on, but it does not claim that the underlying dycore/physics system has fully removed every stabilizing scaffold.

## 6. Hardware and Software Setup

The measured target workstation uses a single NVIDIA GeForce RTX 5090 with 32607 MiB reported by `nvidia-smi` and CUDA device `cuda:0` \cite{nvidia2025geforce}. The 3 km d02 measured grid is `(44, 66, 159)` at 3000 m horizontal spacing. The 1 km memory audit used a derived full-domain synthetic state and reported peak process memory of 7278 MiB, leaving approximately 78 percent of the reported GPU memory free for that one-step feasibility probe.

The runtime path is Python and JAX/XLA \cite{jax2018github,frostig2018tracing}. The publication sprint environment reports Python 3.13.11 and JAX 0.10.0. Earlier sprint text referred to JAX 0.4.x, so the release package must pin the exact runtime manifest before submission. The project package itself currently declares Python `>=3.10` and `jax>=0.4`. The CPU comparison baseline is WRF v4.7.1 running with 28 MPI ranks on the same workstation. CPU pinning for sprint-side processing used cores 0-3, leaving the project convention of reserving cores 4-31 for the CPU WRF comparison jobs.

### 6.1 Canary Workflow and Verification Data

The Canary Islands case was chosen because it is operationally useful and technically unforgiving. The domain contains steep volcanic terrain, sharp land-sea contrasts, marine boundary-layer structure, trade-wind acceleration channels, and island wakes. These features are sensitive to horizontal pressure-gradient treatment, surface fluxes, lateral boundary relaxation, and boundary-layer coupling. A model that performs well on smooth synthetic tests can still fail in this setting.

The measured M7 run is a d02 replay case, not a full live forecast cycle. It starts from a retained Gen2 WRF run, constructs a JAX replay state, advances the GPU operational path for the d02 domain, writes hourly `wrfout`-style NetCDF products, and evaluates station scores. The replay structure was a deliberate isolation choice. It allowed the project to test the GPU core against a known Gen2 source without simultaneously solving raw AIFS ingestion, nested d01 production, and retention-policy gaps.

The AEMET verification scaffold joins forecast values to station observations and reports BIAS, MAE, and RMSE for T2, U10, and V10 \cite{aemet2026observations}. For the 20260521 side-by-side comparison used in this paper, the common valid-time range was 2026-05-21 19:00 UTC to 2026-05-22 18:00 UTC. The scoring report contains 73 station IDs, 24 common valid hours, and 1747 joined station-time rows. CPU and GPU outputs were passed through the same scoring code; only the wrfout source differed. That symmetry is what makes the skill regression evidence stronger than the earlier finite-score pipeline check.

The station verification is still incomplete. It does not yet include a robust precipitation event set, a multi-day seasonal sample, or object/neighborhood precipitation metrics. It also does not replace grid-to-grid Tier 4 comparisons against Gen2 or WRF fields. The current station scaffold is sufficient to reject an operational replacement claim because the GPU is much worse than CPU WRF on all three aggregate surface variables. It is not sufficient to diagnose the root cause by itself.

For that reason, the next validation pass should pair station aggregates with hourly spatial error maps, boundary-zone versus interior masks, and physics-on/off brackets. Those artifacts would turn the current rejection into a localized repair target instead of another broad tuning loop.

## 7. Results: Performance and Systems Evidence

Table 1 summarizes the performance and systems claims that survive the post-closeout amendment.

| Claim | Value | Proof object |
|---|---:|---|
| Warm 1 h d02 forecast wall time | 5.71 s | `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json` |
| Three-run warm reproducibility CV | 0.42 percent | `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json` |
| 24 h d02 pipeline wall time | 324.78 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` |
| 24 h forecast-only wall time | 310.27 s | same |
| CPU d02-only wall timing | 16305 s | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` |
| Pre-fix apples-to-apples speedup | 50.20x | same |
| Post-fix apples-to-apples speedup (with correct physics) | 23.02x | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` |
| 24 h post-fix pipeline wall time | 708.32 s | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json` |
| Full five-domain CPU aggregate framing (pre-fix) | 138.24x, not apples-to-apples | same |
| Inter-kernel D2H inside loop | 0 copies, 0 bytes | `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` |
| Restart continuity | max delta 0.0 | `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json` |
| Pipeline wrfout inventory | 24/24 files readable | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/wrfout_inventory.json` |
| 1 km memory probe | 7278 MiB of 32607 MiB | `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json` |

The corrected speedup should be read conservatively. The GPU number is the d02 pipeline wall time for the 20260521 case. The selected CPU denominator is the de-duplicated d02-only WRF timing for the same source run family. That is the fairest domain-level comparison available in the proof object and the lowest defensible headline ratio. The five-domain aggregate CPU framing gives 138.24x, but it includes d01-d05 CPU work that the single-domain GPU d02 pipeline did not perform. This paper therefore uses 50.20x as the headline and only mentions 138.24x with an apples-to-oranges caveat.

Cold compilation remains operationally relevant. The 1 h profile recorded cold JIT compile-inclusive walls of 102.58 s for the 20260509 case and 106.18 s for the 20260521 case. The daily 24 h pipeline amortizes that cost across hourly chained forecasts, but a production deployment must still manage compile-cache invalidation across code, driver, JAX, and shape changes. The current result is strongest as a warm-run throughput result with reproducibility evidence, not as a complete operations-SLA result.

The D2H result is an important architectural proof. The audit reports zero D2H inter-kernel bytes inside the XLA module window for `jit_run_forecast_operational`. This supports the constitution's central rule: no host/device transfer inside timestep loops unless explicitly approved and documented. It also distinguishes this project from partial ports where an unported scheme pulls full fields back to the host at each coupling point.

## 8. Results: Forecast Quality and the Skill Regression

The honest forecast-quality result is negative. Side-by-side scoring was performed on 73 AEMET station IDs over 24 common valid hours, producing 1747 joined station-time rows. GPU and CPU wrfouts were evaluated through the same `gpuwrf.validation.forecast_vs_obs` scaffold. Table 2 reports aggregate RMSE.

| Variable | CPU WRF RMSE | GPU RMSE | Relative change | Verdict |
|---|---:|---:|---:|---|
| T2 | 2.15 K | 7.86 K | +266 percent | material regression |
| U10 | 2.31 m s-1 | 11.31 m s-1 | +390 percent | material regression |
| V10 | 2.75 m s-1 | 9.44 m s-1 | +243 percent | material regression |

The regression is broad: GPU was materially worse on every metric of every variable in the aggregate comparison, and the report verdict was `FAIL_SKILL_DIFF`. The failure is not just a station-scorer artifact. A separate L2 d02 replay validation on the same 3 km grid but with L2-d01 boundary forcing produced `L2_D02_BOUNDED_FAIL`: T2 RMSE 4.07 K against a 3.0 K threshold, U10 RMSE 10.78 m s-1 against 7.5 m s-1, and V10 RMSE 7.83 m s-1 against 7.5 m s-1.

### 8.1 Root-cause analysis and fix iteration

After the initial draft of this paper was written, two parallel root-cause-analysis sprints landed and converged. An opus architectural audit (`.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md`) and a codex empirical bisection (`.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/worker-report.md`) independently identified three coupled defects in the operational forecast path:

1. **Dycore state reset every timestep.** The `disable_guards=False` production branch of `_physics_boundary_step` was overwriting the post-RK3 `theta`, `mu`, `mu_total`, and `mu_perturbation` with the pre-step values. The RK3+acoustic advance was being discarded each step. Independent corroboration came from on-disk bounds-check artifacts that recorded `theta_lower_30_max_k` identical to seven decimal places across all 24 hourly snapshots of the L2 d02 replay run.
2. **Surface fluxes computed but not applied.** The MYNN PBL adapter received `jnp.zeros_like(theta_columns)` as its bottom boundary heat, moisture, and momentum flux inputs. `surface_adapter` computed `theta_flux`, `qv_flux`, `tau_u`, and `tau_v` but wrote them only into the State record without advancing the atmospheric profile. The ordering in `_physics_boundary_step` was also inverse to WRF's convention.
3. **Radiation effectively disabled.** `DailyPipelineConfig.radiation_cadence_steps` defaulted to 999999, which meant `rrtmg_adapter` was never invoked in the 8640-step 24 h integration. The surface energy balance had no shortwave forcing during the day and no longwave forcing at night.

The codex bisection also surfaced a max-14-category `LU_INDEX` mismatch between GPU and Gen2 wrfouts at lead 1 h. A follow-up audit confirmed that the in-memory `LU_INDEX` from `wrfinput_d02` matched Gen2 exactly; only the `wrfout_writer` fallback collapsed land cells to category 2 and water to 17 because the GPU `State` carries no `LU_INDEX` field. The forecast physics used the correct categories. `LU_INDEX` is therefore a publication-quality cleanup target for the output writer rather than a root cause of the skill regression itself.

### 8.2 Partial fix outcome

A combined fix sprint applied all three algorithmic changes: theta and mu now flow through RK3 with inline bounded guards instead of being reset, `surface_adapter` runs before `mynn_adapter` and feeds its computed fluxes into the PBL bottom boundary, and `radiation_cadence_steps` was reduced to 180 (so RRTMG runs 48 times in a 24 h integration). The post-fix sprint produced `SKILL_IMPROVED_PARTIAL` (`.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/worker-report.md`):

- All M6 and M7 invariants are preserved: 20260521 multi-step parity step 2 = 0.0 bitwise, B6 savepoint parity preserved, inter-kernel D2H = 0 bytes, restart bitwise PASS.
- Six of nine T2/U10/V10 aggregate metrics improved versus the pre-fix baseline. U10 RMSE and V10 RMSE saw real recovery; T2 RMSE worsened because the inline theta envelope `[200 K, 400 K]` for the lower 30 levels saturates the diurnal warming maximum. Three metrics worsened overall.
- All three variables remain outside the pre-declared ±20 percent tolerance against CPU WRF. The post-fix forecast is still not operationally skill-equivalent.
- The wall-clock cost of running correct physics is 700.89 s for 24 h end-to-end (compared to 324.78 s for the pre-fix path that was running incorrectly). The corrected speedup is 23.02x apples-to-apples d02-only, still 3x above the 4-8x target.

Remaining defects named for the next iteration: (a) the theta guard envelope is too tight and pins the lower-column maximum, suppressing diurnal evolution; (b) land surface state (`t_skin`, `SST`, `SMOIS`, `SH2O`, `TSLB`) is still frozen at the IC, with no time evolution from the Gen2 reference or from a prognostic Noah-MP-style scheme; (c) the lateral boundary application uses only the outermost parent row (`bdy_width=1`) instead of the WRF `spec_bdy_width=5` strip.

The publication claim is therefore: the fast, whole-state-resident, restart-bitwise architecture is correct, and the validation and fix discipline can localize coupled defects under operational scoring. The forecast is improving but is not yet a WRF skill replacement. The next iteration is in progress.

The negative result reframes the publication claim. The engineering result is that a JAX-native, whole-state-resident WRF-compatible pipeline can run very fast on a single workstation GPU and can satisfy restart, repeatability, and transfer-audit gates. The meteorological result is that the current forecast is improving but not yet skill-equivalent to CPU WRF; it cannot be advertised as an operational replacement.

## 9. Discussion

The project is best understood as two intertwined experiments: one in GPU-native NWP architecture and one in AI-agent scientific-software production.

The architectural experiment supports the value of whole-state residency. The 50.20x corrected d02 throughput ratio is large enough to matter even after rejecting the original 156.82x celebration number, which came from a timing-denominator error and is not used as a result in this paper. It also changes the economic picture for regional forecasting: a single consumer GPU can run the measured 3 km Canary forecast in minutes rather than hours. If the skill gap is resolved, this would make high-frequency local ensembles, repeated sensitivity tests, or rapid backfills much more accessible than with a CPU-only operational path.

The result also shows why performance without validation can be misleading. A forecast that is fast, finite, restartable, and repeatable can still be wrong. The original M7 closeout overinterpreted finite AEMET station scores. The scorer had produced real measurements, but it had not compared GPU skill against CPU WRF. Once the correct side-by-side comparison was run, the claim changed immediately. This is the central scientific lesson of the sprint: proof objects must match the claim being made. A finite-output proof object is not a skill proof object.

The AI-agent methodology helped because it made that correction possible. A single human plus autocomplete workflow might have moved from the inflated closeout directly to public communication. Here, the manager's later validation step launched an honest-speedup and skill-diff sprint before release. The process found a timing-denominator bug and a skill regression against its own preferred narrative. That does not make the process infallible. It shows that adversarial, proof-object-driven AI collaboration can create useful internal friction.

The method also has weaknesses. The manager was itself an AI system and initially made the celebration error. The validation discipline was stronger than ordinary chat-based coding, but it was not equivalent to an independent human numerical-methods review. Some local policies, such as guard behavior and radiation cadence, were allowed to survive long enough to affect publication readiness. The project caught those issues, but late. A future workflow should require side-by-side CPU/GPU skill comparison before any closeout can use the word "operational."

One practical lesson is that AI agents benefit from narrow contracts but suffer when the contract names the wrong proxy. The pipeline-integration sprint correctly produced a working 24 h pipeline, hourly files, finite fields, station-score rows, and wall-clock evidence. Those were the requested artifacts. The later error was in treating those artifacts as proof of operational skill. This is not a worker failure so much as a specification failure. The next version of the process should make "claim type" explicit: performance claims require profiler and timing proof; numerical-equivalence claims require comparator proof; operational-skill claims require CPU/GPU/observation proof on the same cases.

Another lesson is that cross-model disagreement is useful only when it is attached to files. Parallel critic sprints that merely argue would not have helped. The useful sprints wrote JSON tables, verdict markdown, command outputs, and failed hypotheses. The repository could then preserve both the positive result and the contradiction. That property is especially important for scientific software because many wrong paths are locally plausible. A WRF-compatible formula may have the right variable names and still use a wrong mass basis. A profiler may show copies but use the wrong time window. A station scorer may produce thousands of rows but answer the wrong skill question.

Finally, the result suggests a more modest but still valuable publication framing. The strongest claim is not "AI agents built an operational WRF replacement." They did not. The strongest claim is that a governed AI-agent process built a nontrivial GPU-native regional NWP prototype, produced a large performance gain, and then generated the evidence that prevented an overclaim. In a field where numerical trust matters more than demonstration speed, that self-correction is part of the result.

Compared with traditional scientific-software development, the wall-time pace was unusual. The project produced a large amount of code, tests, proof objects, and documentation in less than two weeks of focused agent work. The cost of that speed is that the repository must treat validation artifacts as first-class state. Without contracts, proof objects, and independent review, the same speed would amplify errors.

## 10. Limitations

The current system is single-GPU only. The halo interface is a placeholder, and no multi-GPU MPI or GPU-aware exchange result is claimed. The 1 km result is a memory-feasibility probe, not a full 1 km forecast validation.

The Canary workflow is currently one-way and replay-based. The GPU d02 system receives boundary side histories derived from existing Gen2 WRF output. That is an acceptable way to isolate d02 dynamics and physics while the port is being validated, but it is not the same as running the full nested operational stack from raw AIFS IC/BC. A production system must either ingest AIFS and geog/static fields directly or document a reproducible bridge from Gen2 products to GPU state. Until then, the model should be described as a validated replay pipeline, not a standalone live-cycle replacement.

The current validation corpus is too small. M6 used three V3 initial conditions for operational Tier-4 RMSE gates, and M7 side-by-side station scoring used one 24 h 20260521 case. That is enough to expose a serious problem; it is not enough to characterize seasonal or regime-dependent behavior. A full Tier-4 ensemble remains corpus-blocked until more Gen2 d02 CPU/GPU comparable pairs are retained or replayed.

The physics path is incomplete as an operational claim. Radiation cadence is effectively disabled in the measured namelist. Microphysics admissibility and finite guards are load-bearing in at least one diagnostic history. Surface and boundary coupling require root-cause review. The current system should be called a fast prototype with known skill regression, not a validated WRF replacement.

The data path is replay-driven. The measured d02 forecast uses Gen2 d02 side-history replay. AIFS is the planned IC/BC source, and AIFS is relevant to the project's future operational workflow, but direct AIFS ingestion is not the measured claim in this paper.

The authorship process needs external review before public release. The AI agents wrote and reviewed much of the system, and this draft discloses that. However, the project has not yet had a truly independent human reviewer audit the numerical decisions, code, and paper claims. The senior human author remains responsible for final acceptance.

The release manifest is not yet frozen. This drafting environment reports Python 3.13.11 and JAX 0.10.0, while earlier sprint text referenced JAX 0.4.x. The final reproducibility package must pin the exact Python, JAX, jaxlib, CUDA, driver, XLA flags, git commit, and proof-object commit hashes.

The paper itself is a first draft. Several references in the research briefs were extracted from PDFs and secondary web pages rather than final publisher metadata. The included BibTeX file is parseable, but the manager review pass should verify bibliographic details before submission. The Mollick et al. citation requested by the sprint contract remains marked as a TODO because it was not present in the provided brief text.

## 11. Reproducibility

The public code URL is `github.com/<TBD>`. The release package should include the final repository commit, a hardware manifest, a software-version manifest, and the proof-object directories listed in this paper. The current proof-object backbone is:

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

The final artifact should also include a script that verifies the BibTeX file, computes the paper word count, checks that the honesty audit covers all quantitative claims, and reruns the light-weight repository documentation checks. Heavy GPU and CPU forecast reruns should remain separate because they depend on the RTX 5090 and retained Gen2 CPU WRF corpus.

## 12. Author Contributions

Claude Opus 4.7 designed and managed the sprint process, maintained long-horizon repository context, wrote and reviewed sprint contracts, synthesized proof objects, drafted this manuscript, and made the original closeout error that was later corrected by the honest-speedup and skill-diff sprint.

GPT-5.5 Codex / OpenAI implemented much of the repository code as worker, performed targeted debugging and critical-review sprints, generated proof objects under manager contracts, and contributed the empirical evidence base used in this draft.

Enric R.G. defined the Canary Islands operational target, supplied and maintained the Gen2 CPU WRF baseline context, set the validation gates and performance expectations, monitored the AI-agent process, and retains senior corresponding-author responsibility for final scientific acceptance and submission.

## 13. Acknowledgements

The project depends on the WRF and NCAR modeling community, ECMWF AIFS context, AEMET station observations, NVIDIA GPU tooling, and Enric R.G.'s prior Gen2 operational Canary forecasting system. The authors also acknowledge that this draft is not a final release manuscript: root-cause analysis of the skill regression, release-manifest pinning, independent review, and public repository preparation remain open.

## References

References will be rendered from `publication/draft/references.bib` during LaTeX conversion.
