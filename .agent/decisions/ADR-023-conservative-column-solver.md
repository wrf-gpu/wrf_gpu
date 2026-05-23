# ADR-023 — Conservative Column Solver (PROPOSED — RATIFY-ADR-023 with required fixes)

**Status**: **PROPOSED** — ratified 2026-05-23 02:10 UTC after round-2 critic `RATIFY-ADR-023` + prototype `PASS_WARM_BUBBLE_600S` + R7 GREEN. The production-grade sprint folded the critic fixes into code and tests, but reviewer `b2f7a05` rejected the path split. The public-scan unification sprint now routes the public nonhydrostatic scan through the MPAS recurrence, but the unified warm-bubble path does not meet the 600 s warm-bubble acceptance targets. Reviewer concurrence is still required before this ADR moves to ACCEPTED.
**Date**: 2026-05-23
**Author**: Manager (Claude Opus 4.7, 1M-context)
**Ratification evidence**:
- Scout `RECOMMEND-THIRD-OPTION` — `.agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md`
- Critic r2 `RATIFY-ADR-023` — `.agent/sprints/2026-05-23-m6x-adr023-three-way-critic/reviewer-report.md` (commit `c1a3ded`, 26871 bytes, 10 required fixes)
- Prototype proof — `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/` (3/3 R7 GREEN, warm-bubble PASS, 4/4 solver unit, 8/8 c2 horizontal regression, 5/5 transfer audit, 20 kernel launches)

**Scope**: M6.x dycore vertical-acoustic + vertical-theta-transport operator
**Supersedes**: ADR-022-DRAFT (rejected by both critic rounds)
**Triggered by**: scout `RECOMMEND-THIRD-OPTION` + critic r1 `RATIFY-EITHER-WITH-CONDITIONS` (round-1 critic report lost to worktree race)

## Decision

Keep ADR-022's JAX backend, small `AcousticScanCarry` (no expansion to WRF small-step scratch families), and ADR-020 architecture skeleton verbatim. **Replace ADR-022's simplified `ph_new = ph_old + dt * g * w_new` with a conservative tridiagonal column solver over the coupled `(w, mu, theta, phi/exner)` system**, patterned after the strongest external GPU-NWP precedents:

- **SCREAM / HOMME** `DirkFunctorImpl.hpp:344-356, 707-778` — DIRK with Newton outer + cyclic-reduction / Thomas tridiagonal Jacobian solve.
- **ICON4Py** `vertically_implicit_dycore_solver.py:139-206, 283-356`; `solve_tridiagonal_matrix_for_w_forward_sweep.py:18-68` — explicit tridiagonal forward sweep + back substitution for w.
- **MPAS-A** `mpas_atm_time_integration.F:1461-1656, 2141-2208` — Klemp et al. MWR 2007 forward-backward vertically implicit integration, up-sweep + down-sweep + back-substitution.

This is the scout's recommended "ADR-022b" path. The implementation is **smaller than ADR-021** (no carry expansion, no WRF Fortran harness, no `_1`/`_save` field families) and **safer than ADR-022** (uses a conservative column solve rather than a simplified geopotential update that drops `msfty/mu`, `(1+epssm)²` off-centering, and `c2a*alt*t_2ave` buoyancy weighting).

## Rationale

Three converging lines of external evidence (all from `m6x-dycore-alt-methods-scout/worker-report.md`):

1. **No successful GPU-NWP project carries WRF small-step scratch in its scan API.** Pace wraps the vertical solve as `NonhydrostaticVerticalSolver` (`dyn_core.py:472-479`); MPAS uses perturbation variables `ru_p`, `rw_p`, `rtheta_pp`, `rho_pp` inside its acoustic step (`mpas_atm_time_integration.F:1824-1833, 2141-2208`) but as local in-step variables, not as a permanent carry expansion. ADR-021's seven WRF-scratch carry families have no precedent in any operational or pre-operational GPU port the scout found.

2. **SCREAM/HOMME is evidence for the column-implicit GPU pattern, not proof that ADR-023 can omit Newton for a nonlinear HEVI solve.** SCREAM carries nonlinear EOS pressure/exner coupling, forms a tridiagonal Jacobian analytically, and emits explicit Newton-failure warnings (`DirkFunctorImpl.hpp:384-390`). ADR-023 v0 is therefore explicitly a linearized MPAS/WRF-style acoustic-gravity solve; a nonlinear HEVI variant requires a future ADR with Newton residual/Jacobian proof.

3. **ICON4Py + MPAS together cover the entire tridiagonal column solver design space.** ICON4Py's forward-sweep + back-substitution is the canonical JAX-friendly shape; MPAS's Klemp et al. 2007 forward-backward integration is the canonical hybrid-eta-coordinate shape. ADR-023 inherits both — ICON4Py's algorithmic structure under `lax.scan`/`vmap`, MPAS's variable-perturbation form on the eta grid.

## Specification

### 1. Column-solver structure

Per acoustic substep, for each (i, j) column, solve the linearized coupled tridiagonal system:

```
F(w, phi/exner, theta', mu')_implicit + F(...)_explicit = 0
```

with the implicit (column-coupled) terms being:

- **w equation**: `w_new = w_old + dt * (-(c2a*alt)/mu_total * grad_z(p') - g*theta'/theta_base)` with `grad_z` solved implicitly via tridiagonal.
- **phi equation**: `phi_new = phi_old + dt * g * 0.5 * ((1-epssm)*w_old + (1+epssm)*w_new)` — off-centered Crank-Nicolson form following MPAS, NOT the simplified ADR-022 form.
- **theta' equation**: vertical advection of theta' by w_new, solved on faces using `fnm/fnp` interpolation coefficients.
- **mu' equation**: column-integrated continuity, solved by the existing `mu_continuity_tendency` (preserved verbatim from c2-A2).

The tridiagonal coefficient builder uses **per-entry hybrid-eta denominators** matching WRF lines 626/632/637-639/646 — same as ADR-022 §1 and ADR-021 — but the coefficients flow into a Newton/tridiagonal solve rather than a w-only solve.

### 2. Solver implementation in JAX

Algorithm:
1. Build (a, b, c, rhs) tridiagonal entries per column using `jnp.vectorize` / `jax.vmap`.
2. Forward sweep + back substitution (Thomas algorithm) implemented as a `lax.scan` over the vertical dimension.
3. For GPU: cyclic-reduction (CR) tridiagonal solve from the same XLA primitive wrapper already in use (`src/gpuwrf/numerics/tridiagonal_solver.py` from M5-S2).
4. No Newton outer loop in v0. The unknown is the vertically coupled linear perturbation vector, with nonlinear EOS/exner response diagnosed after the solve. SCREAM/HOMME remains a precedent for the GPU column-implicit family and a warning that any nonlinear HEVI extension must add Newton machinery.

### 3. AcousticScanCarry — no expansion

Same six-leaf tuple `(state, previous_pressure, al, alt, cqu, cqv)` as ADR-022. This is the architectural payoff inherited from ADR-022.

The categories are explicit:
- **Public carry**: exactly the six scan leaves `(state, previous_pressure, al, alt, cqu, cqv)`.
- **Per-substep locals**: transient variables inside the scan body or vertical solve, including `rs`, `ts`, `rw_p`, `rho_pp`, `theta_perturbation`, `dmu_dt`, and post-solve pressure diagnostics.
- **Solver scratch**: tridiagonal coefficients (`cofrz`, `cofwr`, `cofwz`, `coftz`, `cofwt`, `a`, `b`, `c`) built and consumed within `_calc_coef_w` / `build_epssm_column_coefficients`.

No per-substep local or solver scratch field may be added to `AcousticScanCarry` without a new ADR.

### 4. Other inherited fixes

- **R3 hybrid-eta denominators**: built into the new tridiagonal coefficient builder.
- **R4 msf-factor in `uncouple_horizontal_pgf_tendency`**: same as ADR-022 §5.
- **R7 analytic oracle**: already in `tests/test_m6x_vertical_acoustic_oracle.py` (3 RED tests). The implementation must turn these green.
- **R8 layer thickness**: operator uses `rdnw`, not `dz_m`.
- **R9 `top_lid`**: honored in coefficient builder; `w(nz) = 0` enforced post-solve.
- **R10**: drop defensive `abs(...)` and clamp bandaids in production paths.

### 5. Off-centering parameter

`epssm` is an `AcousticConfig` field, default 0.1 (MPAS-canonical), wired through the coefficient builder. ADR-022 hard-coded epssm=0 and dropped Crank-Nicolson; ADR-023 restores both. Production sprint sweep evidence keeps 0.1 as the default: `epssm=0.3` improves the MPAS-slice trajectory RMSE but breaks the R7 analytic period gate, so it is not selected.

### 6. Post-solve replacement order

The acoustic scan applies post-solve replacements in this order:

1. `w` from the tridiagonal vertical solve.
2. `theta` from vertical transport or MPAS `rtheta_pp` reconstruction.
3. `ph_perturbation` / `ph_total` from the off-centered geopotential update.
4. `mu_perturbation` / `mu_total` from in-scan `mu_continuity_tendency`.
5. `p_perturbation` / `p_total` diagnostics.
6. `al`.
7. `alt`.

The order is mirrored in `POST_SOLVE_REPLACEMENT_ORDER` and covered by `tests/test_m6x_adr023_production_grade.py`.

## Constraints

- No host/device transfer inside the timestep loop (transfer-audit gate remains binding).
- fp64 for pressure / mass / geopotential; fp32 acceptable for θ' per ADR-007.
- The horizontal PGF path from c2-A2 is preserved verbatim except for the R4 msf-factor multiplication.
- **Tier-1 WRF-savepoint parity is relaxed but not abandoned**: the column solver is conservative and Crank-Nicolson off-centered. MPAS Klemp 2007 is treated as a family-level same conservative off-centered tridiagonal structure, not equation-level equivalence. The current MPAS-derived column-slice trajectory proof is a validation rung, not a percent-level MPAS binary equivalence claim.
- Tier-4 RMSE on U10/V10/T2 at 24h/72h vs Gen2 backfill is the binding acceptance gate.
- The R7 oracle tests in `tests/test_m6x_vertical_acoustic_oracle.py` must turn green before the sprint can close.

## Trade-offs vs ADR-021 and ADR-022

| Dimension | ADR-021 WRF port | ADR-022 hybrid simple | **ADR-023 conservative** |
|---|---|---|---|
| `AcousticScanCarry` expansion | 7 new field families | none | **none** |
| Tier-1 WRF parity | binding | not binding | **softly binding (~5% vs MPAS Klemp 2007)** |
| Tier-4 RMSE binding | yes | yes | **yes** |
| Worker-time to warm-bubble PASS | 5-9 days | 2-4 days | **8-14 days total estimate (5-8d operator proof + 3-6d forecast relevance)** |
| Risk of late "missing-term" discovery | medium-high | medium | **low (conservative column form is well-understood)** |
| Operational GPU-NWP precedent | MPAS (Fortran/OpenACC) | Dinosaur/NeuralGCM (global spectral) | **SCREAM HEVI + ICON4Py + MPAS Klemp 2007 (3 supports)** |
| Future portability beyond WRF baseline | low | high | **high** |
| ADR-007 4× target compat | conditional | conditional | **conditional, same gates** |

## Evidence

All from `.agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md`:
- §1 comparative table (SCREAM, ICON4Py, MPAS, Pace, Dinosaur, NeuralGCM)
- §2 pattern 1 (split-explicit + tridiagonal) and pattern 2 (HEVI-IMEX) both ratify the conservative column form
- §3 "ADR-022 should not claim the omitted mass/geopotential weighting is tiny without either a 1-D column oracle or a WRF/MPAS-derived fixture"
- §5 explicit `RECOMMEND-THIRD-OPTION` text proposing this exact ADR

Plus:
- R7 analytic oracle now on disk (`tests/test_m6x_vertical_acoustic_oracle.py`, commit `f6965be`)
- ADR-022-DRAFT and ADR-021-DRAFT for trade-off comparison

## Open questions for the second critic round

1. **Is the linear single-tridiagonal-pass sufficient, or is a Newton outer required?** Scout cited SCREAM uses Newton; ADR-023 v0 omits it on the linearity argument. Verify against the R7 analytic oracle's amplitude-decay test — if a Newton outer would change the verdict, escalate.
2. **Off-centering parameter `epssm = 0.1` (MPAS default).** Is this defensible for the WRF baseline target, or should it be `epssm = 0.0` (centered, matches ADR-022's implicit choice) or `epssm = 0.3` (WRF Klemp default)?
3. **Tridiagonal solver: forward-sweep `lax.scan` vs cyclic-reduction XLA primitive.** Performance trade-off — `lax.scan` is portable but sequential along z; CR is parallel but more code. ADR-023 v0 specifies CR; reconsider if launch budget breaks.
4. **Does the ADR-023 path require the WRF Fortran harness?** v0 says no — the analytic oracle plus MPAS savepoint (if extractable from local MPAS source) suffices. Verify the local MPAS source `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/` is harness-extractable.

## Status

**PROPOSED** — ratified 2026-05-23 ~02:10 UTC. Moves to ACCEPTED after the next production-grade sprint folds the critic's required fixes below.

## Critic required fixes (must fold into next sprint, per round-2 critic findings)

Folded from `2026-05-23-m6x-adr023-three-way-critic/reviewer-report.md`. The prototype already cleared several of these; the rest are open for the production-grade sprint:

### MAJOR
- **F1 — MPAS equivalence claim**: This ADR's "mathematically equivalent to MPAS Klemp 2007" language is demoted to "family-level same conservative off-centered tridiagonal structure." Equation-level equivalence requires a discrete derivation artifact + one MPAS or WRF column-slice comparison. *Status: CLOSED for ADR text + slice-rung evidence by `tests/test_m6x_mpas_column_slice_oracle.py` and `tests/test_m6x_adr023_production_grade.py`; MPAS binary equivalence remains out of scope.*
- **F2 — Newton-outer justification**: v0 is explicitly framed as a **linearized MPAS/WRF-style acoustic-gravity solve**, not a nonlinear SCREAM HEVI-style. SCREAM is cited only as evidence that nonlinear HEVI requires Newton, not as evidence that single-pass tridiagonal is sufficient. *Status: CLOSED in §Rationale and §Specification; no Newton outer is claimed for nonlinear HEVI.*
- **F3 — R7 oracle red**: Critic noted oracle was red at critic-write time. *Status: CLOSED by prototype and production regression — `tests/test_m6x_vertical_acoustic_oracle.py` remains 3/3 GREEN.*
- **F4 — Tridiagonal solver module path**: ADR previously referenced non-existent `src/gpuwrf/numerics/tridiagonal_solver.py`. The real module is `src/gpuwrf/physics/tridiagonal_solver.py`. *Status: CLOSED — ADR-023 now uses `src/gpuwrf/dynamics/vertical_implicit_solver.py`, including Thomas default, XLA alternative, and the production epssm coefficient builder.*

### MEDIUM
- **F5 — `epssm` default**: not yet swept. Production sprint must sweep `epssm ∈ {0.0, 0.1, 0.3}` against R7 + warm-bubble + slice trajectory, bind default to evidence. *Status: CLOSED pending reviewer acceptance — `proof_epssm_sweep.txt` keeps default 0.1 because 0.3 improves slice RMSE but fails R7.*
- **F6 — Tier-4 acceptance ladder**: Production sprint must implement the staged ladder: analytic-oracle → MPAS column slice → warm-bubble → 1h d02 boundary replay → 24h/72h Gen2 RMSE. No skipping rungs.
- **F7 — Public carry vs locals vs scratch**: ADR must explicitly name the three categories. Production sprint adds this to the ADR + the code structure. *Status: CLOSED in §3 and `AcousticScanCarry` docstring; `tests/test_m6x_adr023_production_grade.py` asserts the six-leaf public carry.*
- **F8 — Cost re-estimation**: critic re-estimated 5-8d operator-proof + 3-6d forecast-relevance. Manager's 3-5d was optimistic. *Status: revised in §Trade-offs to 8-14d total.*
- **F9 — Post-solve replacement order**: critic asked for explicit order for `(w, theta, ph_perturbation, mu_perturbation, p_perturbation, al, alt)`. *Status: CLOSED in §6, `POST_SOLVE_REPLACEMENT_ORDER`, and production-grade tests.*
- **F10 — Performance/residency claims**: prototype reported launch count 20 + 0 transfers + scan jaxpr without host callbacks. *Status: CLOSED for this sprint's evidence by `proof_transfer_audit.txt` and `proof_launch_count_production.txt`; full ncu/nsys remains a later profiler-bot artifact if required by manager.*

## Prototype caveats (production sprint disposition)

From `2026-05-23-m6x-adr023-conservative-column-prototype/worker-report.md` §Risks:

1. **Nonhydrostatic warm-bubble no longer uses the separate buoyancy-column path**: reviewer-reject closure removed the `_wrf_buoyancy_column_update` branch, the named `NONHYDROSTATIC_BUOYANCY_SCALE`, and the positive-only updraft drag. Both `pressure_scale=0.0` and the public `non_hydrostatic=True` scan path now enter the same epssm-aware MPAS recurrence. This closes the path-split defect but exposes that the conservative recurrence alone does not yet satisfy the warm-bubble rung.

2. **Warm-bubble unified-path failure is now explicit evidence**: deleting the mass limiter outright made the public scan nonfinite at step 2. Retaining the documented temporary `mu_continuity` CFL bound keeps the run finite through 600 s but fails the target envelope (`w_max=0.289 m/s` at 300 s and `0.041 m/s` at 600 s in the worker proof). This is not a Tier-4 physics claim and must be resolved before d02 or 24 h replay.

3. **Launch count remains unoptimized**: reviewer-reject closure remeasured the unified `pressure_scale=-1.0` recurrence at 67 kernel launches, matching the prior direct `pressure_scale=0.0` MPAS recurrence path and exceeding the prototype baseline of 20. This is not a host-transfer regression, but it blocks any speed claim until a profiler-bot optimization sprint.

## Fallback trigger (per critic §6 open question 6)

If the production-grade sprint cannot pass R7 + one WRF/MPAS column slice + an honest warm-bubble (without prototype-grade stabilization heuristics) within one sprint plus one fix sprint, **fallback to ADR-021** (full WRF small-step shape vertical port). The architectural step-back §4 third pivot trigger ("needs broad unreviewed state/contract changes") would then re-fire on the production-grade attempt, justifying the ADR-021 carry expansion as the only remaining option.

## Next sprint authority

The next sprint (`2026-05-23-m6x-adr023-production-grade` or similar) is authorized to:
- Modify this ADR to incorporate F1/F2/F5/F6/F7/F9 fixes
- Edit `acoustic_wrf.py` vertical operator further
- Add MPAS/WRF column-slice comparison
- Run `epssm` sweep
- Add full profiler artifact

It is **NOT** authorized to:
- Expand `AcousticScanCarry` beyond the 6-leaf form
- Add a Newton outer loop without explicit ADR amendment
- Modify the R7 analytic oracle or its tests
- Change the c2-A2 horizontal PGF or mu continuity
