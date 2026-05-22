# GPU Dycore Architecture Scout Report

Date: 2026-05-22

Decision: recommend C, a new dycore architecture sprint, with the scope framed as a small milestone rather than a single bundled c1 patch. Do not choose A, because bundling map factors, hybrid-eta, damping, diffusion, sponge, and limiters into the existing c1 branch would recreate the exact debugging failure mode that the methodology review warned against. Do not choose B as the technical path if the goal remains a WRF-compatible GPU dycore; B is only a defensible administrative close if the user explicitly decides that M6 should end as a throughput demonstration and that M7 real-case dycore correctness is no longer a near-term objective.

The manager consensus I recommend is: close the current c1 work as evidence of throughput and short-time structural progress, but open a new architecture sprint before adding any more production dycore mechanisms. The new sprint should port the architecture, not the code, of the best existing GPU dycore precedents into our JAX/WRF-compatible system. Pace/FV3 is the best primary reference for the architecture of a production Python/GPU dycore with split acoustic dynamics, explicit grid metrics, damping, hyperdiffusion, Rayleigh damping, and tracer fixing. ICON4Py/ICON-exclaim is the best secondary reference for explicit metric/interpolation/config state and controlled nonhydrostatic solve organization. NeuralGCM/Dinosaur is the best JAX style reference, especially for coordinate systems, pytrees, filters, and scan-friendly pure functions. HOMMEXX/SCREAM are valuable GPU and package-architecture evidence but are not the right numerical model to imitate for a regional, WRF-compatible, Arakawa C-grid, hybrid-eta dycore.

## Evidence Base

I used the project constitution, AGENTS instructions, the sprint role prompt, the local prior-art and disagreement-resolution skills, the current morning report, methodology review, bughunt4 meta-analysis, empirical bisection closeout, A7 and A11 c1 closeouts, ADR-001, ADR-002, ADR-003, the current `src/gpuwrf/contracts` and `src/gpuwrf/dynamics` modules, and the local WRF `dyn_em` source. I treated the source as read-only, as required by the sprint contract.

Public sources checked:

- HOMMEXX paper: Bertagna et al., "HOMMEXX 1.0: a performance-portable atmospheric dynamical core for the Energy Exascale Earth System Model", Geoscientific Model Development, 2019, https://gmd.copernicus.org/articles/12/1423/2019/
- E3SM HOMME source tree: https://github.com/E3SM-Project/E3SM/tree/master/components/homme
- E3SM SCREAM/EAMxx source tree: https://github.com/E3SM-Project/E3SM/tree/master/components/eamxx
- Pace repository: https://github.com/ai2cm/pace
- ICON4Py repository: https://github.com/C2SM/icon4py
- NeuralGCM paper: Kochkov et al., "Neural general circulation models for weather and climate", Nature, 2024, https://www.nature.com/articles/s41586-024-07744-y
- Dinosaur repository: https://github.com/neuralgcm/dinosaur

The most important local fact is that c1 has already achieved the throughput target, around 44x in the morning report, and has shown a structurally plausible warm-bubble response for the first few hundred seconds. It has not closed real-case stability: the same morning report records that the sanitize acceptance path failed badly, with a max-ratio RMSE around 21x against a 1.5x target, saturated or nonfinite final state, and warm-bubble failure by 600 seconds. A11 fixed one diagnostic-pressure bug and exposed the next layer of missing WRF mechanisms: map factors, hybrid-eta coefficients, smdiv divergence damping, sixth-order hyperdiffusion, Rayleigh damping or sponge, and positive-definite or monotonic limiting. The methodology review had already warned that c1 likely had a structural coordinate and operator mismatch, not just a single local bug. The empirical bisection was still useful: it showed that earlier failures were triggered in advection/coupling rather than boundary forcing, physics replacement, or acoustic subcycling alone. That narrows the problem but does not make the missing stabilizers optional.

## Q1. What Existing GPU Dycore Solutions Teach Us

### HOMMEXX and SCREAM

HOMMEXX is a performance-portable C++/Kokkos rewrite of HOMME, the High-Order Methods Modeling Environment dycore used in E3SM. The paper describes a port designed to keep one code base efficient on CPUs, many-core processors, and Nvidia GPUs. The important architectural lesson is not that we should copy spectral elements into our code. The lesson is that a successful GPU dycore port treats the dycore as a product with explicit kernels, layout choices, initialization reuse, regression tests, and performance validation. HOMMEXX did not become successful by sprinkling device pragmas over an unstable port; it rewrote the on-node parallel structure around Kokkos views, execution policies, and data layout.

That said, HOMMEXX is a poor direct numerical reference for our current decision. Its dynamical core family is global spectral element, not WRF's finite-difference or finite-volume regional Arakawa C-grid on terrain-following hybrid eta. HOMME's domain, element structure, metric handling, halo communication, and timestep operators are not close enough to WRF `dyn_em` to answer our specific c1 problem. Our constraints include WRF-compatible `wrfinput` and `wrfbdy`, WRF-style map factors, WRF hybrid-eta mass and face coefficients, regional lateral forcing, and fp64-sensitive acoustic and pressure pathways. HOMMEXX can inform GPU engineering discipline, but using it as the reference architecture would produce a second model family rather than a WRF-compatible one.

SCREAM is also useful but not directly as a dycore solution. The E3SM EAMxx tree describes SCREAM as a next-generation atmosphere component, with a C++ atmosphere driver and process implementations, while E3SM still uses HOMME/HOMMEXX-family dynamics underneath for the relevant global dycore path. SCREAM's value for us is package orchestration: isolate processes, make fields and grids explicit, build tests around each package, and prevent hidden host/device movement. It does not solve the c1 issue of WRF-compatible map factors, hybrid-eta acoustic coefficients, and regional boundary forcing.

Verdict for Q1: HOMMEXX/SCREAM are GPU-port maturity references, not the architecture to port. They argue against B, because mature projects do not stop at throughput when the dycore is unstable. They also argue against A, because they reached GPU viability through explicit architecture and validation, not a bundle of last-mile patches.

### Pace / FV3

Pace is the strongest primary reference for our architectural question. Its repository describes it as a Python/GT4Py implementation of the FV3GFS/SHiELD atmospheric model that can run from a laptop backend to heterogeneous supercomputers. The numerical model is not WRF: FV3 uses a cubed-sphere grid and a different vertical Lagrangian/remap strategy. But the software decomposition is very close to what we need. Pace keeps grid metrics, damping coefficients, vertical coefficients, acoustic dynamics, tracer advection, hyperdiffusion, Rayleigh damping, negative tracer adjustment, halo updates, and checkpointer validation as named parts of the dycore system.

The local Pace source inspection matters more than the README. `fv_dynamics.py` constructs a `DynamicalCore` that owns `grid_data`, `damping_coefficients`, `config`, `state`, `timestep`, and checkpointer hooks. It composes `AcousticDynamics`, `HyperdiffusionDamping`, `LagrangianToEulerian`, tracer advection, and negative tracer adjustment. The step flow makes split acoustic dynamics an explicit sub-loop and applies tracer transport, vertical remap, optional omega hyperdiffusion, and negative tracer repair in named phases. `dyn_core.py` uses a loop over acoustic substeps with C-grid and D-grid shallow-water components, pressure-gradient components, vertical solve, Rayleigh damping, halos, and hyperdiffusive heating. `del2cubed.py` owns sixth-order style hyperdiffusion state and coefficients. `ray_fast.py` owns Rayleigh damping. `fillz.py` owns a mass-aware negative tracer fix. `grid/helper.py` and related generation code make grid metrics and vertical coefficients first-class data, including areas, inverse areas, edge lengths, `ak`, `bk`, `ptop`, reference pressures, and damping coefficients.

This is exactly the architectural mistake in c1: c1 currently represents the dycore as a small set of advection and acoustic functions over a minimal `State` and `GridSpec`, while the WRF-equivalent problem has more required static and semi-static structure than those contracts can express. Pace shows that those structures should not be hidden inside ad hoc operator functions. They should be carried as typed grid, metric, damping, and vertical-coordinate objects that are resident on device and passed through compiled kernels.

Verdict for Q1: Pace is the best primary architecture to adapt, with the caveat that we should not port FV3 numerics. We should port the decomposition pattern: metric-rich grid data, explicit vertical coordinate coefficients, split acoustic scan, named stabilizer modules, limiter modules, halo/boundary grouping, and savepoint-grade validation.

### ICON-exclaim / ICON4Py

The practical public code artifact for ICON-exclaim-style work is ICON4Py, a Python/GT4Py implementation of ICON components and utilities. It is less directly compatible than Pace because ICON's nonhydrostatic dycore is built around an unstructured triangular/icosahedral C-grid rather than WRF's structured regional C-grid. However, ICON4Py is very useful as a secondary architecture reference because it treats metric state, interpolation state, vertical parameters, damping options, and intermediate diagnostic fields as explicit dataclasses rather than hidden globals.

The local ICON4Py inspection found `NonHydrostaticConfig` with explicit choices for Rayleigh damping, divergence damping order and type, off-centering, divergence damping coefficients, damping height bands, nudging coefficients, and extra diffusion. `SolveNonhydro` owns `metric_state_nonhydro`, `interpolation_state`, vertical parameters, geometry, exchange runtime, and stencil program wrappers. `IntermediateFields` preallocates pressure-gradient fields, edge-centered density and virtual potential temperature, kinetic energy, tangential winds, horizontal gradients of normal-wind divergence, and vertical wind derivatives. `dycore_states.py` defines enums for divergence damping type/order, pressure discretization, and advection mode. A specific divergence damping stencil adds vertical-wind derivative terms into the divergence-damping pathway, showing that this is a first-class dycore mechanism, not a late numerical afterthought.

ICON4Py's lesson is that a modern GPU dycore implementation should freeze its state taxonomy before parallel work begins. There should be a difference between prognostic state, static metrics, intermediate diagnostics, solver coefficients, and configuration. Our current `State` and `GridSpec` blur that boundary because `GridSpec` has only idealized terrain and vertical eta levels, while the dycore operators have to infer or simplify missing WRF quantities. ICON4Py says the opposite: name the metric and intermediate fields, allocate them once, and let the solver organization make damping and diffusion choices explicit.

Verdict for Q1: ICON4Py is the best secondary architecture reference. It should not be copied numerically, but it strongly supports C over A because the missing c1 mechanisms cross state, metric, operator, and configuration boundaries.

### NeuralGCM / Dinosaur

NeuralGCM is not a regional WRF-compatible dycore. The Nature paper presents a model that combines a differentiable atmospheric solver with machine-learning components and demonstrates weather, ensemble, and climate results at global scales. The public Dinosaur repository describes itself as NeuralGCM's differentiable dycore. Its code is highly relevant to our backend decision because it is JAX-native and uses coordinate systems, pytrees, sharding constraints, functional filters, time integration, and differentiable model composition.

The mismatch is just as important as the relevance. Dinosaur is a global spectral or spherical-harmonic style dycore, not a structured regional WRF C-grid with lateral boundary forcing. Its filters operate naturally in modal space; our sixth-order WRF-style diffusion, smdiv damping, and monotonic limiter must operate on staggered grid fields and WRF mass-coupled fluxes. Dinosaur's coordinate-system abstraction is valuable, but its equations and grid topology are not the correct target.

Verdict for Q1: NeuralGCM/Dinosaur should guide our JAX coding style and pytree contracts, not our WRF compatibility. It supports C because it shows what a clean JAX dycore architecture looks like; it does not support B, and it does not make A safe.

## Q2. How the Chosen Reference Handles the Missing Mechanisms

The chosen architecture reference is Pace primary, ICON4Py secondary, with WRF `dyn_em` as the physics/numerics source of truth and Dinosaur as the JAX style guide.

Map factors: WRF passes map-scale factors through the small-step and advection code as explicit arrays such as `msfux`, `msfuy`, `msfvx`, `msfvy`, `msftx`, and `msfty`. The local WRF source shows those factors entering small-step mass and pressure updates and horizontal and vertical flux divergences. The A9/A11 evidence already found a real example in WRF v-momentum vertical flux where `msfvy/msfvx` scales the tendency. Pace handles the analogous problem by carrying grid metrics, areas, inverse areas, edge lengths, and related metric coefficients in `GridData`. ICON4Py does the same conceptually through geometry, metric state, and interpolation state. The architecture lesson is direct: WRF map factors must be part of `GridSpec` or a child `DycoreMetrics` pytree, not baked into individual advection patches.

Hybrid eta: WRF's small-step code takes `c1h`, `c2h`, `c1f`, `c2f`, `c3h`, `c4h`, `c3f`, and `c4f`, plus vertical inverse spacings such as `rdn` and `rdnw`. Local WRF `calc_coef_w` uses these coefficients in the tridiagonal vertical acoustic solve and pressure/geopotential coupling. c1's current `GridSpec` can say `kind="hybrid_eta"` and carries eta levels, but it does not carry the WRF coefficient families. Pace carries `ak`, `bk`, `ptop`, reference pressure, and related pressure-grid data as part of the grid/dycore object. Dinosaur carries coordinate systems as explicit pytrees. The architecture lesson is that a string saying "hybrid_eta" is not enough; the coefficients that define the coordinate must be data, loaded from WRF fixtures when running WRF-compatible cases and generated analytically only for idealized tests.

Divergence damping / smdiv: WRF uses small-step divergence damping through `smdiv` and previous-step pressure-like fields, with a pressure update term that damps acoustic divergence. Pace has a dedicated divergence damping module and damping coefficients. ICON4Py makes divergence damping order/type/config explicit and contains stencils that include vertical wind derivative contributions. The architecture lesson is that smdiv belongs in a named damping module with explicit inputs, previous-step carry, and tests. It should not be merged into a generic pressure update without a proof oracle, because that makes conservation and acoustic stability regressions hard to isolate.

Sixth-order hyperdiffusion: WRF has a sixth-order, monotonic, flux-limited numerical diffusion pathway in `module_big_step_utilities_em.F`, with `diff_6th_factor`, `diff_6th_opt`, map-factor scaling, and up-gradient guards. Pace has a named `HyperdiffusionDamping` component with coefficients such as `del6_u` and `del6_v`, area metrics, flux temporaries, and application loops. ICON4Py has a diffusion module with configurable diffusion choices and upper damping. The architecture lesson is that hyperdiffusion must be a separate pure operator over the fields it owns, with coefficient objects, boundary/halo behavior, and monotonicity tests. It should be invoked from the timestep orchestrator at a deliberate point rather than hidden inside advection.

Rayleigh damping / sponge: WRF's `rk_rayleigh_damp` uses damping coefficients in the upper layer, height dependence, and mass/pressure coupling. Pace has a `RayleighDamping` module that damps winds in an upper pressure region while respecting pressure thickness. ICON4Py has Rayleigh type/config in nonhydrostatic configuration and metric state fields such as `rayleigh_w`. The architecture lesson is again modular: the sponge is a named vertical damping operator with vertical-coordinate and metric dependencies. It is not a magic stabilizer to apply after nonfinites appear.

Positive-definite and monotonic limiter: WRF's positive-definite routines repair negative scalar states by redistributing mass. Pace has `FillNegativeTracerValues` and water-species adjustment modules that use pressure thickness/mass to preserve column properties while fixing negatives. Dinosaur has filters as pure functions over pytrees, although its filters are spectral and not directly the WRF limiter. The architecture lesson is that the limiter must be mass-aware and field-aware. It should expose conservation diagnostics and should run at a defined point after tracer/scalar transport, not as a global sanitize step that masks failures.

Together these mechanisms show why A is risky. They are not six independent lines of code. They require new static fields, new vertical coefficients, new timestep carries, new diagnostics, new proof harnesses, and new configuration. Pace and ICON4Py both treat those as architecture.

## Q3. How to Port the Architecture to JAX While Keeping Our Compatibility

The port target should be a WRF-compatible JAX dycore architecture, not a Pace/FV3 port. ADR-001 remains valid: JAX/XLA is the primary backend, and kernels should be expressed as pure functions compiled with `jit` and `lax.scan` where appropriate. ADR-002 remains directionally valid: `State` is an SoA pytree with one fp64 leaf per prognostic field and C-grid staggering. But ADR-002 needs an amendment because the current `GridSpec` and `State` contracts cannot represent the WRF dycore mechanisms now known to be required.

The first change should be to split dycore data into four categories:

1. `State`: prognostic or time-evolving fields, still SoA and fp64 for pressure/geopotential/mu/acoustic-sensitive paths.
2. `BoundaryState`: time-interpolated lateral forcing fields and tendencies from `wrfbdy`, still resident and grouped by side and field.
3. `DycoreMetrics` or extended `GridSpec`: static grid, map-factor, vertical-coordinate, and damping coefficient arrays loaded from `wrfinput` or generated by analytic fixtures.
4. `DycoreDiagnostics` or scan carry: previous-step pressure fields, flux accumulators, intermediate tendencies, and proof counters that are needed inside the timestep but should not be exposed as long-lived model state unless WRF compatibility requires it.

Recommended `GridSpec` extension: add a child dataclass, perhaps `DycoreMetricSpec`, with `msftx`, `msfty`, `msfux`, `msfuy`, `msfvx`, `msfvy`, and derived inverses or ratios only when storing them reduces repeated division and matches WRF behavior. Add vertical arrays for `eta_mass`, `eta_face`, `c1h`, `c2h`, `c3h`, `c4h`, `c1f`, `c2f`, `c3f`, `c4f`, `dn`, `dnw`, `rdn`, and `rdnw`. Add static damping/filter config for `smdiv`, sixth-order diffusion options, Rayleigh damping depth/coefficient, and limiter flags. Because JAX static arguments trigger recompilation when Python objects change, the array-valued parts must be pytree leaves, while small enum/config values can be static only when the sprint deliberately accepts separate compilations.

Recommended `State` changes: keep map factors out of `State`; they are static metrics. Add or separate read-only base-state fields if WRF compatibility requires them, especially base pressure or pressure-base diagnostics that c1 currently reconstructs imperfectly. Consider a `StaticState` or `BaseState` pytree for `pb`, base geopotential, and other WRF base profiles that vary in three dimensions over terrain but do not evolve as prognostic fields. Add missing boundary leaves only after confirming the WRF fixture requires them; likely candidates are `w_bdy`, `p_bdy` or perturbation-pressure boundary data, and base-pressure/geopotential boundary support. The current boundary schema only carries six leaves, which the bughunt showed was not the initial nonfinite trigger but is still too thin for long-run compatibility.

Recommended new or reorganized files under `src/gpuwrf/dynamics`:

- `metrics.py`: WRF map-factor accessors, staggered metric interpolation, safe metric ratios, and map-scaled divergence helpers.
- `hybrid_eta.py`: WRF hybrid-eta coefficient loading/validation, pressure reconstruction helpers, Exner or diagnostic pressure utilities, and vertical finite-difference helpers using `rdn` and `rdnw`.
- `transport.py` or `flux_form.py`: mass-coupled flux-form advection and momentum/scalar tendency assembly with metric factors separated from stencil math.
- `acoustic_wrf.py`: WRF small-step acoustic scan, including pressure/geopotential/mu coupling, vertical tridiagonal solve, and smdiv carry.
- `damping.py`: smdiv divergence damping, Rayleigh damping, upper sponge, and their configuration dataclasses.
- `hyperdiffusion.py`: sixth-order WRF-style flux-limited diffusion with map factors and opt/factor handling.
- `limiters.py`: positive-definite and monotonic scalar correction with mass conservation diagnostics.
- `orchestrator.py` or a revised `rk3.py`: the RK3 and acoustic substep composition point, with no host/device transfers inside the step.

The current `step.py` already has the right outer shape: `run()` uses `lax.scan` over timesteps and `step()` is JIT-compiled with static grid/dt/substep knobs. That should remain. The new architecture should map as follows: the outer forecast loop is one `lax.scan` over model timesteps; each timestep calls a pure `wrf_em_step(state, boundary_state, metrics, config, carry)`; RK3 stages are either static calls or a small `lax.scan` over stage descriptors; acoustic small steps are a nested `lax.scan` over `n_acoustic`; vertical tridiagonal solves are implemented as vectorized column solvers using `vmap` over horizontal columns and `lax.scan` over vertical forward/back substitution; filter and limiter passes are pure functions called at named points in the scan body. Previous-pressure or smdiv memory should be in the scan carry, not in Python globals.

This preserves compatibility with ADR-001 and ADR-002 while fixing the missing representation. It also keeps the current single-GPU-first assumption. The design should not introduce MPI or multi-device decomposition in this sprint. Halo functions can remain no-op for the single-GPU fixture, but their signatures should receive the richer field groups so that boundary and halo correctness are not rediscovered later.

## Q4. Scope Estimate: One Sprint, Five Sprints, or New Milestone

This is not safely a one-sprint c1 bundle. A one-sprint bundle would have to change the contracts, load or synthesize WRF metrics, implement hybrid-eta coefficients, revise advection and acoustic coupling, add smdiv, add hyperdiffusion, add Rayleigh damping, add a limiter, and prove the result on WRF and analytic fixtures. That is too much to do in one c1 continuation without losing the ability to tell which mechanism fixed or broke the model.

The minimum responsible shape is a new dycore-architecture sprint followed by implementation and validation sprints. If the question must be answered as "one sprint, five sprints, or milestone", the answer is: new milestone, with roughly five focused sprints inside it. The first sprint freezes the architecture and lands the data contracts and skeletal modules. The next one or two sprints implement the WRF vertical coordinate and metric-correct flux/acoustic pathways with isolated oracles. The next sprint implements stabilizers and limiters with conservation tests. The next sprint integrates real-case boundary forcing and validates against WRF savepoints and Gen2/AIFS comparisons. The final sprint reruns throughput, transfer audit, profiler artifacts, and long-window stability.

An aggressive but plausible schedule is 3 to 5 working days for the first architecture sprint, then 8 to 14 working days for enough implementation and validation to decide whether the JAX WRF dycore can become the M7 base. A conservative schedule is three weeks. The important point is not the calendar number; it is that the work crosses architectural boundaries. The cost of not doing this is continued c1 churn where every new WRF mechanism is added as a patch and every failure again becomes ambiguous.

## Q5. Recommendation Among A, B, and C

Recommend C.

A, bundle into c1, is the wrong engineering move. It is tempting because c1 is fast and nearly useful. But the evidence says the remaining problems are structural. The current `GridSpec` can represent only idealized eta levels and terrain, not WRF's map-factor and hybrid-coefficient machinery. The current dycore files contain simplified periodic upwind advection and reduced acoustic coupling, not a framework with named metric, damping, hyperdiffusion, sponge, and limiter modules. Adding six mechanisms inside c1 without freezing the contracts would violate the lesson from the methodology review and the project rule to freeze interfaces before parallel work begins.

B, throughput-only close, is honest but strategically weak. It would acknowledge that M6 achieved an important performance result and that c1 short-time warm-bubble behavior is not nonsense. If the only goal were to end the current sprint without more scope, B is defensible. But B does not answer the user's architecture concern and does not advance the system toward a real WRF-compatible GPU dycore. It risks turning a successful throughput prototype into a dead-end artifact.

C, new sprint, is the best path because it changes the unit of work to match the unit of missing complexity. C should not mean "port FV3", "port ICON", or "replace everything with NeuralGCM". It means adopt the architecture patterns that production GPU dycores have converged on: explicit metric and vertical-coordinate data; named acoustic, transport, diffusion, damping, and limiter modules; scan-friendly pure functions; device-resident carries; savepoint and analytic proof gates; and a clean separation between static grid data, prognostic state, boundary forcing, and intermediate diagnostics.

The consensus position to give the manager is therefore:

- Use B only as the administrative closeout label for the old c1 throughput sprint if a closeout is required today.
- Do not authorize A.
- Authorize C as the technical path, with a short architecture sprint first and a clear stop/go gate before implementation expands.

## Q6. Draft Sprint Contract for C

Sprint name: `2026-05-22-m6x-c2-jax-wrf-dycore-architecture`

Objective: freeze and prove the architecture for a JAX/XLA WRF-compatible GPU dycore that can represent WRF map factors, hybrid-eta coefficients, smdiv, sixth-order hyperdiffusion, Rayleigh damping, and positive-definite/monotonic limiting without host/device transfers inside timestep loops. The sprint does not claim operational closure unless the proof gates pass; its primary deliverable is the architecture and the first executable proof harness.

Allowed source ownership:

- `src/gpuwrf/contracts/grid.py`
- `src/gpuwrf/contracts/state.py`
- New files under `src/gpuwrf/dynamics`: `metrics.py`, `hybrid_eta.py`, `damping.py`, `hyperdiffusion.py`, `limiters.py`, and optionally `acoustic_wrf.py` or `orchestrator.py`
- Focused tests under `tests/` for metrics, hybrid coefficients, damping, hyperdiffusion, limiter conservation, and scan residency
- ADR file under `.agent/decisions/` through the patch protocol if architecture approval is required

Explicit non-goals:

- No line-by-line FV3, ICON, HOMME, or NeuralGCM port.
- No MPI or multi-GPU decomposition.
- No hidden sanitize acceptance.
- No physics retuning to compensate for dycore instability.
- No claims of GPU performance unless profiler and transfer-audit artifacts are produced.

Acceptance criteria:

1. ADR/protocol: produce an ADR or architecture patch that amends ADR-002 with `DycoreMetrics`, WRF hybrid-eta coefficients, boundary-state policy, base-state policy, and scan-carry policy. Human approval is required before broad implementation.
2. Metrics proof: load or synthesize `msftx`, `msfty`, `msfux`, `msfuy`, `msfvx`, and `msfvy` for both an analytic flat fixture and a WRF fixture. Prove shapes, staggering, dtype, provenance, and no implicit host callbacks inside `jit`.
3. Hybrid-eta proof: represent `c1h`, `c2h`, `c3h`, `c4h`, `c1f`, `c2f`, `c3f`, `c4f`, `dn`, `dnw`, `rdn`, and `rdnw` as JAX arrays. Compare pressure/geopotential helper output against a WRF savepoint or an analytic oracle.
4. Damping proof: implement smdiv, sixth-order diffusion skeleton, Rayleigh/sponge skeleton, and limiter skeleton as pure JAX functions with disabled-by-default config. Each module must have an isolated test that proves identity when disabled and a nontrivial finite effect when enabled.
5. Scan proof: show an outer timestep `lax.scan` and nested acoustic `lax.scan` can carry the required diagnostics without Python-side mutation or host/device transfer inside the loop. Produce a transfer audit artifact.
6. Conservation and limiter proof: for at least one scalar field, prove the limiter preserves nonnegative mass within tolerance on an analytic fixture and report any deliberate nonconservation.
7. Integration proof: run the warm-bubble or WRF fixture for a short window with all new modules wired but conservative flags, then with stabilizers enabled. The proof object must include finite-state checks, mass/energy-relevant diagnostics, and comparison to the previous c1 result.
8. Decision gate: at closeout, recommend one of: continue C implementation, narrow to B throughput-only, or rollback the architecture if the proof objects show it is incompatible with JAX/XLA or WRF fixtures.

Proof object paths:

- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/architecture.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/metrics.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/hybrid_eta.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/scan_transfer_audit.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/limiter_conservation.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/manager-closeout.md`

Risks to carry into the sprint:

- The current c1 branch may not contain all A7/A11 fixes on the active branch; the architecture sprint must start by reconciling branch state.
- WRF `wrfbdy` pressure and vertical-velocity boundary needs may require a broader boundary schema than current six leaves.
- JAX compilation pressure may grow if all config is static and every fixture recompiles; the design must distinguish static enums from dynamic coefficient arrays.
- Pace and ICON4Py prove architecture patterns, not WRF compatibility. WRF `dyn_em` remains the oracle for numerical choices.
- The first architecture sprint might prove that the JAX representation is viable but still not close the 600-second warm-bubble failure. That is acceptable if the proof objects isolate why.

## Final Consensus Statement

The external architecture survey changes the decision from "maybe close throughput-only because c1 is unstable" to "close the throughput prototype if necessary, but open a new architecture sprint for the technical path." Pace and ICON4Py show that the missing c1 mechanisms are normal dycore architecture, not optional cleanup. NeuralGCM shows that JAX can support clean differentiable dycore composition, but not by ignoring coordinate and grid structure. HOMMEXX/SCREAM show that GPU dycore success requires deliberate architecture and validation, not throughput in isolation.

Therefore the recommendation is C. The manager should reject A as an unsafe bundle and reserve B as an administrative fallback only. The next decision needed from the user/manager is approval to create the C sprint contract and ADR amendment for dycore metrics, WRF hybrid-eta coefficients, stabilizer modules, and scan-carry policy.
