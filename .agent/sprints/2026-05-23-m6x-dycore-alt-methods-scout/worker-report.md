# Worker Report - M6.x Dycore Alternative Methods Research Scout

Summary: External evidence does not cleanly ratify either draft as written. The field has strong evidence for "clean" column-implicit vertical solves, but successful GPU-NWP projects do not generally preserve WRF scratch-state shape line-for-line. They also do not support ADR-022's most aggressive simplification unless it is backed by a column oracle. My recommendation is `RECOMMEND-THIRD-OPTION`: a JAX HEVI column operator that keeps ADR-022's clean carry boundary, but borrows the conservative column coupling and tridiagonal/Newton evidence from ICON4Py, SCREAM/HOMME, MPAS, and Pace/FV3 rather than the simplified phi update currently in ADR-022.

Local access note: Pace/FV3-JAX, Dinosaur, NeuralGCM, ICON4Py, and E3SM/SCREAM were not locally accessible under `/home/enric/src`; their code was inspected through public GitHub raw/API URLs at the requested commits where specified. MPAS was locally accessible at `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/`.

## Section 1 - Comparative Table

| Project | Dycore family | Vertical operator | Backend | Operational status | Tier-4 RMSE reported vs reference |
|---|---|---|---|---|---|
| Pace / FV3-JAX / Vulcan (`Pace@6a46e69`) | FV3 Lagrangian split-explicit acoustic dynamics. `DynamicalCore` owns `AcousticDynamics` (`fv3core/pace/fv3core/stencils/fv_dynamics.py:92-103,257-264`); `AcousticDynamics` is the Fortran `dyn_core` analog (`dyn_core.py:221-225`). | Nonhydrostatic vertical Riemann / semi-implicit solver. `dyn_core.py` imports `NonhydrostaticVerticalSolver` and `NonhydrostaticVerticalSolverCGrid` (`dyn_core.py:35-36`), constructs them (`dyn_core.py:472-479`), and calls C-grid and D-grid vertical solvers inside the acoustic loop (`dyn_core.py:793-805,874-892`). `riem_solver3.py` states the implemented path is semi-implicit and the exact Riemann solver is not yet implemented (`riem_solver3.py:207-234`), then calls `Sim1Solver` (`riem_solver3.py:294-306`). | GT4Py + DaCe Python DSL. README says Pace implements FV3GFS/SHiELD with GT4Py and can run on heterogeneous supercomputers (`README.md:13-21`). | Research/pre-operational port of FV3GFS/SHiELD, not evidence of an operational NOAA forecast system running Pace itself. | No WRF-style Tier-4 RMSE found in inspected source. Evidence is Fortran/FV3 parity and performance-portability work, not Canary-like 24h/72h RMSE. |
| Dinosaur (`Dinosaur@59a0197`) | Global spectral primitive-equation/shallow-water dycore on sigma coordinates. README states spectral methods and primitive equations on sigma coordinates (`README.md:5-10`). Coordinate metadata is explicit in `CoordinateSystem`, including separate dycore and physics sharding (`dinosaur/coordinate_systems.py:85-121,181-189`). | General IMEX ODE abstraction, not a WRF-like column acoustic step. `ImplicitExplicitODE` splits explicit and implicit terms and provides `implicit_inverse` (`dinosaur/time_integration.py:74-114`). It ships Crank-Nicolson/RK2, low-storage RK/CN, RK3/RK4, and SIL3 IMEX (`time_integration.py:193-226,233-405`). | JAX. README says it is rewritten in JAX for AI weather/climate models and GPU/TPU acceleration (`README.md:5-10`). | Experimental research project; README says documentation is still experimental (`README.md:16-26`). | No WRF-style Tier-4 RMSE in inspected source. Dinosaur is a differentiable dycore substrate, not a regional WRF reference comparison. |
| NeuralGCM | Hybrid ML/physics atmospheric model built on Dinosaur. README says weather and climate simulation (`neuralgcm/README.md:5-8`); `model_builder.py` imports `dinosaur.time_integration` (`neuralgcm/legacy/model_builder.py:21-27`); `api.py` advances sigma-level state through an `advance_fn` and unrolls through `time_integration.trajectory_from_step` (`neuralgcm/legacy/api.py:427-446,549-557`). | Uses Dinosaur's trajectory/time-step machinery; no inspected code path resembling WRF split-explicit acoustic substeps. IMEX support comes through Dinosaur's ODE API and SIL3 (`dinosaur/time_integration.py:375-405`). | JAX/Haiku plus Dinosaur. `advance` and `unroll` are JIT/static configured in API (`neuralgcm/legacy/api.py:427-449,475-557`). | Research/productized library with released checkpoints, not operational NWP. | Paper-level weather RMSE exists externally, but no source-level vertical-operator RMSE or WRF reference RMSE was found. No evidence that NeuralGCM needed WRF-style split-explicit substeps for stability. |
| ICON4Py / GT4Py (`ICON4Py@3934f68`) | ICON nonhydrostatic dycore translated into Python/GT4Py. README calls it a work-in-progress Python implementation of ICON and performance-portable GT4Py components (`README.md:7-19`). | Vertically implicit nonhydrostatic solver with explicit predictor/corrector wiring. `NonHydrostaticConfig` exposes sound-wave/off-centering and divergence damping knobs (`solve_nonhydro.py:139-194,307-338`). `SolveNonhydro` sets up predictor and corrector vertical implicit programs (`solve_nonhydro.py:543-621`) and calls the predictor operator with w/rho/exner/theta inputs (`solve_nonhydro.py:1165-1193`). The operator computes tridiagonal coefficients and solves `w` by forward sweep and back substitution (`stencils/vertically_implicit_dycore_solver.py:139-206,283-356`; `stencils/solve_tridiagonal_matrix_for_w_forward_sweep.py:18-68`). | GT4Py / Python DSL with backend selection (`solve_nonhydro.py:338-352`). | Experimental; README says highly experimental and not packaged on PyPI (`README.md:19-21`), though it targets integration with ICON Fortran. | No operational RMSE in inspected source. The project has datatest/stencil parity evidence, not a published Tier-4 forecast RMSE in the inspected files. |
| SCREAM / E3SM-MMF / HOMME-NH | Nonhydrostatic spectral-element atmosphere using HOMME in EAMxx/SCREAM. SCREAM README says it is a next-generation E3SM atmosphere component (`components/eamxx/README.md:1-5`). EAMxx docs and DOE presentation report C++/Kokkos and exascale climate use, including HEVI-IMEX and 1.26 simulated-years/day at 3.25 km on Frontier. | HEVI-IMEX DIRK/Newton column solve. Kokkos `DirkFunctorImpl` computes an initial hydrostatic `phi` guess (`DirkFunctorImpl.hpp:141-199`), runs Newton on `w`/`phi` with nonhydrostatic pressure/exner (`DirkFunctorImpl.hpp:203-394`), forms a tridiagonal Jacobian, and solves it with cyclic reduction on GPU or Thomas on CPU (`DirkFunctorImpl.hpp:707-778`). The explicit RK path calls `dirk.run` at each stage (`prim_advance_exp.cpp:237-263`). Fortran `imex_mod.F90` documents the DIRK stage equations (`imex_mod.F90:63-86`). | C++/Kokkos. | Successful exascale climate model; not WRF-compatible regional NWP. | No WRF-style Tier-4 RMSE in inspected source. Operational-like evidence is throughput and climate validation, not WRF/Canary RMSE. |
| MPAS-A | Split-explicit RK3 nonhydrostatic atmospheric solver. Local `mpas_atm_time_integration.F` states time-split RK3 and nonhydrostatic solver (`MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:117-124`). | WRF-like acoustic small-step with vertically implicit tridiagonal solve. It computes vertical implicit coefficients before RK/acoustic loops (`mpas_atm_time_integration.F:437-493,1461-1656`), prepares small-step perturbation variables, runs `atm_advance_acoustic_step` in a small-step loop (`mpas_atm_time_integration.F:622-731`), and `atm_advance_acoustic_step` explicitly says it follows Klemp et al. MWR 2007 using forward-backward vertically implicit integration (`mpas_atm_time_integration.F:1824-1833`). The solve sweeps up and down the column and then back-substitutes rho/theta (`mpas_atm_time_integration.F:2141-2208`). | Local artifact is Fortran/OpenMP in the inspected dynamics file. Public MPAS GPU docs describe a special OpenACC branch and `OPENACC=true`; this local artifact did not show OpenACC directives in the dycore file inspected. | Community/research/forecast model lineage; GPU-enabled MPAS-A branch is documented, but the local WSM6 artifact is not proof of an operational GPU dycore. | No Tier-4 RMSE in inspected source. MPAS technical documentation and tests are algorithmic, not a Canary-style RMSE proof. |

## Section 2 - Pattern Analysis

1. Split-explicit acoustic loop with vertically implicit column solve.

Projects: MPAS-A, WRF/ERF lineage, Pace/FV3. Canonical reference: Klemp, Skamarock, and Dudhia 2007; MPAS source also names this reference in the acoustic step (`mpas_atm_time_integration.F:1824-1833`). Math summary: update horizontal acoustic/momentum terms explicitly on small time steps; eliminate vertically coupled `w`, density/mass, and thermodynamic perturbations to a per-column tridiagonal system; then back-substitute mass/theta and accumulate averaged mass flux. MPAS makes this concrete with coefficient assembly (`mpas_atm_time_integration.F:1461-1656`) and up/down column sweeps (`mpas_atm_time_integration.F:2172-2208`). Pace/FV3 uses an acoustic loop and Riemann/semi-implicit vertical solver (`dyn_core.py:701-715,793-805,874-892`; `riem_solver3.py:207-234`).

Implication: this pattern supports the physical premise behind ADR-021, but not necessarily the exact line-for-line WRF carry expansion. Pace and MPAS both preserve algorithmic structure while using their own solver abstractions and state layouts.

2. HEVI-IMEX column operator without WRF small-step scratch carry.

Projects: ICON4Py and SCREAM/HOMME. Canonical references: ICON nonhydrostatic dycore; HOMME-NH DIRK/IMEX implementation. Math summary: split horizontal/nonstiff terms into explicit RK stages and solve the stiff vertical acoustic/exner/w coupling implicitly by a per-column tridiagonal or Newton/tridiagonal system. ICON4Py computes beta/alpha coefficients and solves for `w` (`vertically_implicit_dycore_solver.py:139-206,283-356`). SCREAM forms a nonlinear residual in `w`/`phi`, computes a tridiagonal Jacobian, and uses a GPU tridiagonal solver (`DirkFunctorImpl.hpp:344-356,707-778`).

Implication: this is the strongest support for ADR-022's direction, but it argues for a richer conservative column solve than ADR-022's simplified `ph_new = ph_old + dt*g*w_new` statement. Successful projects keep thermodynamic, exner/pressure, and metric couplings explicit in the solver.

3. Spectral JAX IMEX for global primitive equations.

Projects: Dinosaur and NeuralGCM. Canonical reference: Dinosaur/NeuralGCM source at `Dinosaur@59a0197`. Math summary: represent atmospheric state in modal/nodal spectral coordinates and define an ODE split `dx/dt = F(x) + G(x)` with `G_inv = (1 - dt G)^-1`; then use CN/RK or IMEX RK time stepping (`time_integration.py:74-114,193-226,233-405`). Filtering damps high spectral modes (`filtering.py:38-99`).

Implication: this supports "JAX can carry IMEX atmospheric time integration" and contradicts any claim that JAX requires WRF-style split-explicit state. It is weak evidence for our specific regional eta-coordinate vertical acoustic operator because Dinosaur is global, spectral, and not WRF-boundary-compatible.

## Section 3 - What Is Missing From ADR-021 And ADR-022

ADR-021 captures the WRF/MPAS/Klemp split-explicit family, but it underweights the evidence that GPU-native ports usually abstract the vertical solve rather than drag all legacy scratch fields into the public scan carry. Pace does not expose WRF-style `t_2ave`, `_1`, and `_save` families in its table-level API; it wraps the vertical solve as `NonhydrostaticVerticalSolver` and `NonhydrostaticVerticalSolverCGrid` (`dyn_core.py:472-479`). MPAS is closer to WRF, but its state is perturbation variables `ru_p`, `rw_p`, `rtheta_pp`, and `rho_pp` inside a local acoustic step (`mpas_atm_time_integration.F:1824-1833,2141-2208`), not proof that a JAX scan carry should grow by seven WRF scratch families.

ADR-022 mentions Pace, Dinosaur, and ICON4Py, but it misses the strongest external support for a robust hybrid operator: SCREAM/HOMME's DIRK/Newton column solve. SCREAM is the clearest production GPU evidence for HEVI-IMEX at scale, and its implementation is not just a simple geopotential update. It carries nonlinear EOS pressure/exner coupling, forms an analytical tridiagonal Jacobian, and has explicit failure handling for Newton convergence (`DirkFunctorImpl.hpp:344-390,707-778`). ADR-022 should not claim the omitted mass/geopotential weighting is tiny without either a 1-D column oracle or a WRF/MPAS-derived fixture.

Both ADRs miss a useful third option: a JAX HEVI column solve with ADR-022's small carry, but MPAS/ICON/SCREAM-grade conservative coupling. It would keep `AcousticScanCarry` from becoming a WRF scratch container, while replacing ADR-022's simplified phi update with a column solve over `(w, rho/mu, theta, phi/exner)` that uses a tridiagonal or Newton/tridiagonal system. Implementation cost: one sprint to build the analytic column oracle and coefficient builder; one sprint to prototype a JAX Thomas/CR-compatible column solve under `lax.scan`/`vmap`; one reviewer sprint to compare against a WRF or MPAS small-step slice. This is cheaper than a full ADR-021 WRF carry port, but more honest than ratifying ADR-022 as written.

## Section 4 - Risk Signatures

Split-explicit + vertical tridiagonal succeeded in MPAS/WRF-lineage models and appears in Pace/FV3's acoustic loop. I found no explicit "adopted and later regretted" statement in the inspected sources. The risk signature is engineering complexity: MPAS still exchanges `rho_pp` and `rtheta_pp` inside the small-step loop (`mpas_atm_time_integration.F:672-731`) and comments that it solves redundant edge tendencies to minimize communication (`mpas_atm_time_integration.F:622-624,2049-2051`). For our JAX device-residency rules, this pattern must be audited carefully for host/device transfers and halo traffic.

HEVI-IMEX column solves succeeded in SCREAM/HOMME and are the main non-WRF evidence for ADR-022's family. I found no formal regret, but the source exposes the actual failure mode: SCREAM's Newton path can hit max iterations and emit a warning (`DirkFunctorImpl.hpp:384-390`). ICON4Py's README also says the project is highly experimental (`README.md:19-21`). The risk is not the family; it is under-solving or oversimplifying the column physics.

JAX spectral IMEX succeeded as a research substrate for Dinosaur/NeuralGCM, and I found no evidence it needed WRF-style split-explicit substeps. The risk is domain mismatch. Dinosaur uses spectral primitive equations and sigma coordinates (`README.md:5-10`; `coordinate_systems.py:85-121`), so it does not answer whether a Canary regional grid with WRF boundaries remains stable under ADR-022's simplified vertical operator.

GPU directive/OpenACC MPAS evidence is useful but weak for our architecture decision. Public MPAS GPU docs show OpenACC support exists, but the local MPAS dynamics source inspected here contains OpenMP/vector directives and no dycore OpenACC directives in `mpas_atm_time_integration.F`. That means MPAS is strong algorithmic evidence and weak implementation-backend evidence for this sprint.

## Section 5 - Recommendation

`RECOMMEND-THIRD-OPTION`

External evidence supports a clean JAX HEVI operator, but not ADR-022's simplified formulation as written. External evidence also supports WRF/MPAS-style split-explicit vertical implicit integration, but not ADR-021's full WRF scratch-carry expansion as the next best engineering move.

The third option should be ratified as "ADR-022b": keep ADR-022's JAX backend, no broad `AcousticScanCarry` expansion, and no pivot away from the WRF baseline; replace the simplified geopotential update with a conservative column solver patterned after ICON4Py/SCREAM/MPAS. The minimum proof is a 1-D analytic vertical acoustic/gravity-wave oracle plus one WRF or MPAS-derived column slice. If that experiment fails, ADR-021 becomes the fallback because MPAS/WRF evidence shows the small-step family is stable and well understood.

## Commands Run And Output

Validation commands: none; the sprint contract says "None. Cite, don't run."

- `sed -n '1,220p' PROJECT_CONSTITUTION.md`; output confirmed physics correctness, GPU residency, WRF-compatible-where-useful, and no line-by-line WRF port.
- `sed -n '1,260p' AGENTS.md`; output confirmed local `.agent/skills` are authoritative and the old global skill must not be used.
- `sed -n '1,260p' CLAUDE.md`; output confirmed project-local skills and no destructive action.
- `sed -n '1,320p' PROJECT_PLAN.md`; output confirmed ADR-001 JAX primary context, M2/M6 validation posture, and no backend pivot.
- `sed -n '1,260p' .agent/milestones/ROADMAP.md`; output confirmed M6 proof expectations and transfer-audit/validation posture.
- `sed -n '1,220p' .agent/goals/M1-DONE.md`; output confirmed active goal file is read-only.
- `sed -n '1,260p' .agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/sprint-contract.md`; output confirmed no code edits, write-only sprint folder, report sections, no validation commands.
- `sed -n '1,260p' .agent/skills/writing-gpu-kernels/SKILL.md` and `sed -n '1,260p' .agent/skills/writing-execplans/SKILL.md`; output confirmed GPU validation/profiler rules and bounded execution-plan rules.
- `git status --short --branch`; output before branching: `## main...origin/main [ahead 240]`.
- `git switch -c worker/gpt/m6x-dycore-alt-methods-scout`; output: switched to new branch.
- `find /home/enric/src ... pace/dinosaur/neuralgcm/icon4py/scream/e3sm/mpas`; output: no relevant Pace/Dinosaur/NeuralGCM/ICON4Py/SCREAM/E3SM source directories under `/home/enric/src`.
- `find /mnt/data ... pace/dinosaur/neuralgcm/icon4py/scream/e3sm/mpas`; output: MPAS artifact directories found under `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/`; `/mnt/data/lost+found` permission denied.
- Multiple `curl -fsSL <raw-github-url> | rg -n ...` and `curl ... | nl -ba | sed -n ...` commands for Pace, Dinosaur, NeuralGCM, ICON4Py, and E3SM; outputs are the cited line ranges in Sections 1-4.
- `rg -n ... /mnt/data/.../MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F`; output identified MPAS split RK3, coefficient assembly, acoustic step loop, and tridiagonal solve cited above.
- `rg -n "!\\$acc|OpenACC|cuda|device" /mnt/data/.../MPAS-GPU-for_wsm6/src/core_atmosphere/dynamics/mpas_atm_time_integration.F`; output: no matches in the dycore file.

## Files Changed

- `.agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md`

No code files, reviewer report, tester report, manager closeout, memory patch, governance file, rule, skill, contract, or goal file was modified.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md`

## Risks

- Source access was mixed: MPAS local, others public remote. I marked local inaccessibility explicitly.
- SCREAM/E3SM was inspected on public `master` because the contract did not specify a commit.
- Tier-4 RMSE was generally absent from source files; I did not invent numbers.
- The recommendation is architecture-adjacent and should feed the pivot critic, not directly modify ADRs.

## Handoff

Objective: Produce an externally cited scout report on vertical-implicit acoustic/gravity-wave methods relevant to ADR-021 vs ADR-022.

Files changed: `.agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md`.

Commands run: mandatory read-order `sed`; branch/status `git`; local repository discovery `find`; evidence extraction with `curl`, `rg`, `nl`, and `sed`; no validation commands because the sprint contract lists none.

Proof objects produced: this `worker-report.md`.

Unresolved risks: no source-level Tier-4 RMSE for most projects; no local clones for five of six reference projects; no direct experiment yet comparing ADR-021, ADR-022, and the third option on the same column fixture.

Next decision needed: pivot critic should decide whether to ratify ADR-022b as a third option, request one prototype experiment, or force ADR-021 as the conservative fallback.
