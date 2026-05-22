# Sprint Contract — M6.x Dycore Alternative Methods Research Scout

## Objective

The c2 dycore is at a pivot decision (ADR-021 WRF-port vs ADR-022 hybrid JAX IMEX). Before ratifying either, the manager wants a focused **external evidence pass**: how have similar GPU-NWP projects solved the vertical-implicit acoustic step, and what is their operational track record?

The scout's report becomes Appendix A of whichever ADR is ratified, and feeds the pivot critic (`m6x-c2-pivot-critic`).

## Non-Goals

- No code edits anywhere. Read-only.
- Not a market survey; not a Wikipedia tour. Specific to the vertical-implicit acoustic / gravity-wave step.
- Do not recommend pivoting away from JAX or away from the WRF baseline — those are constitutional. Recommendations must be compatible with `ADR-001` (JAX primary) and `PROJECT_CONSTITUTION.md`.

## File Ownership

Write-only to this sprint folder. Read-only everywhere else.

## Inputs

Reference projects to cover (in priority order):

1. **Pace / FV3-JAX (NOAA / Vulcan)**. Repository `Pace@6a46e69`, files `fv3core/pace/fv3core/stencils/fv_dynamics.py`, `dyn_core.py`, `util/pace/util/grid/helper.py`. Operational status, vertical-implicit Riemann solver, AcousticDynamics class, grid metadata pattern.
2. **Dinosaur (Google Research)**. Repository `Dinosaur@59a0197`, files `dinosaur/time_integration.py`, `dinosaur/coordinate_systems.py`, `dinosaur/filtering.py`. IMEX schemes, JAX style.
3. **NeuralGCM (Google)**. Built on Dinosaur. Cite the IMEX choice and whether they ever needed WRF-style split-explicit for stability.
4. **ICON4Py / GT4Py (DLR/MPI-M)**. Repository `ICON4Py@3934f68`, files `model/atmosphere/dycore/solve_nonhydro.py`, `dycore_states.py`. Vertical solver structure, NonHydrostaticConfig, divergence damping.
5. **SCREAM / E3SM-MMF (DOE)**. Kokkos-based, but the algorithmic choices are evidence even though the backend differs. How they handle non-hydrostatic vertical pressure-gradient.
6. **MPAS-A (NCAR)**. Closest WRF analog — hybrid-eta lineage, CUDA Fortran / OpenACC. Their treatment of the vertical-implicit step.

For each project, the report must cite specific file paths and line ranges. If a repository is not locally accessible, say so explicitly — do not fabricate.

## Acceptance Criteria

`worker-report.md` containing five sections:

1. **§1 Comparative table.** Columns: project, dycore family (split-explicit / IMEX / fully implicit / spectral), vertical operator (Riemann / tridiagonal / spectral / other), backend (JAX / Kokkos / GT4Py / native CUDA), operational status (research / pre-operational / operational), Tier-4 RMSE reported vs reference. Rows: the six projects above. Cells with `file:line` citations.

2. **§2 Pattern analysis.** Top 3 algorithmic patterns the field actually uses for the column-vertical-implicit step. For each, name the projects using it, canonical reference paper / commit, one-line math summary.

3. **§3 What's missing from ADR-021 and ADR-022.** Read both ADR drafts in `.agent/decisions/`. For each pattern in §2 that neither ADR mentions, say whether it would be a useful third option, why or why not.

4. **§4 Risk signatures.** For each §2 pattern, cite at least one project that adopted it and later regretted it (instability, performance ceiling) — or report no such evidence exists. Same for projects that adopted it and succeeded.

5. **§5 Recommendation** (exactly one):
   - `RECOMMEND-PROCEED-WITH-ADR-022` — JAX IMEX hybrid well-supported by external evidence; cite strongest support.
   - `RECOMMEND-PROCEED-WITH-ADR-021` — WRF small-step port well-supported; cite strongest support.
   - `RECOMMEND-THIRD-OPTION` — external evidence suggests a path neither ADR considers; describe it with implementation cost.
   - `RECOMMEND-MORE-DATA` — evidence insufficient or contradictory; describe the experiment that would resolve.

## Validation Commands

None. Cite, don't run.

## Performance Metrics

N/A — research sprint.

## Proof Object

`worker-report.md`. Length budget: 2000–4000 words. Time budget: 60–120 minutes.

## Risks

- Vague "Pace uses Riemann" without `file:line` citation — reject and re-cite.
- Confabulating projects not actually accessible. State plainly if a repo is unreachable.
- Bias toward whichever ADR matches training familiarity. Value comes from cited-evidence pattern analysis, not preference.

## Handoff Requirements

When `worker-report.md` is complete, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [scout / m6x-dycore-alt-methods-scout / codex] exit=<ec> report=...` into the manager pane.
