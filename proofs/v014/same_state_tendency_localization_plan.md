# V014 Same-State Tendency Localization Plan

## Scope

This is a design/prototype proof object only. It does not edit `src`, does not consume GPU, and does not claim that any dynamic operator is correct. It defines the smallest falsifiable CPU-first strategy to localize the next U/V/P/PH/MU divergence after static/grid/base-state parity is green.

The target question is narrow:

> Given the same WRF state, same static metrics, same namelist switches, same boundary data, and same source tendencies at a selected h8-h14 time, does the JAX implementation compute the same named tendency terms and one-stage update as CPU WRF?

If the answer is no, the first failing named term becomes the next fix target. If the answer is yes for the selected state but full-run divergence remains, the bug is likely in cadence, state construction, save/restore, boundary timing, physics timing, or an untested region/level rather than in the isolated operator formula.

## Prior Evidence Boundary

Inputs read for this design:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/validating-physics/SKILL.md`
- `.agent/sprints/2026-06-08-v014-static-metric-base-parity/sprint-contract.md`
- `proofs/v014/wind_mass_divergence_probe.md`
- `.agent/reviews/2026-06-08-v014-wind-mass-divergence-probe.md`

The wind/mass probe shows broad dynamic divergence by h8-h14, including interior cells, with strong low-level coupling from prognostic winds and pressure/mass to V10/U10/PSFC. Boundary-only and surface-only explanations are disfavored, but not eliminated. The current V014 sprint contract correctly gates dynamic fixes on static/grid/base-state parity first.

## Minimal Falsifiable Harness

### Case

Use the same retained Case 3 paths from the wind/mass divergence probe:

- JAX/GPU retained outputs for failure-cell selection only: `/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- CPU WRF truth history: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`

The same-state term truth must come from CPU WRF source-derived savepoints, not from hourly interpolated WRF history and not from JAX self-comparison.

### First Sample

Start with one lead time, one model step, all three RK stages, and first plus last acoustic substep:

- Lead candidate: h10 first, then h8, h12, h14 if h10 passes or is ambiguous.
- Domain region: ocean or low-terrain interior cells.
- Interior exclusion: at least 8 horizontal cells from the lateral boundary/spec-relax frame.
- Terrain/land filter: prefer `LANDMASK == 0`; otherwise allow `HGT < 300 m` low terrain.
- Cell selection: choose 16 to 32 mass-grid cells ranked by combined h8-h14 `abs(dV10)`, `abs(dU10)`, `abs(dPSFC)`, and nearby low-level `abs(dU/dV/dP/dPH/MU)` from existing CPU-vs-JAX wrfout differences.
- Patch halo: use full-domain arrays if feasible; otherwise crop patches with radius at least 8 around selected mass cells and retain all vertical levels. Compare native staggering:
  - mass/scalar cells for `mu`, `p`, `ph`, `theta`, moisture
  - U faces adjacent to selected mass cells
  - V faces adjacent to selected mass cells
  - W/PH vertical faces over the selected columns

This is the smallest useful sample because it exercises h8-h14 divergence, avoids boundary-frame dominance, preserves enough stencil halo for fifth-order horizontal advection and sixth-order diffusion, and still keeps CPU-only diagnostics compact.

### CPU-Only Execution Contract

The future harness must set CPU-only execution explicitly:

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=src python <same_state_harness>.py ...
```

Any accidental GPU device selection fails the proof. No GPU timing or profiler claims are in scope.

## Current JAX Code Path Inventory

The operational path is `src/gpuwrf/runtime/operational_mode.py`. The current daily/nested operational configuration enables flux advection and fp64 in integration code, while `OperationalNamelist` defaults keep switches explicit.

| Area | Candidate files/functions | Important local lines inspected | Same-state terms to expose |
| --- | --- | --- | --- |
| RK stage orchestration | `src/gpuwrf/runtime/operational_mode.py::_rk_scan_step` | 2159-2262 | stage initial state, RK weights, large-step tendencies, prep state, acoustic result |
| Large-step assembly | `src/gpuwrf/runtime/operational_mode.py::_augment_large_step_tendencies` | 1693-1991 | advective, diffusive, PGF, Coriolis, dry source folded tendencies |
| Physics/source timing | `src/gpuwrf/runtime/operational_mode.py::_physics_step_forcing`, `_dry_physics_tendencies_from_state_delta`, `_apply_physics_non_dry_updates` | 2758-3057 | raw `*_tendf` availability, state-delta physics increments, radiation held-rate path |
| Acoustic state construction | `src/gpuwrf/runtime/operational_mode.py::_acoustic_core_state_from_prep`, `_acoustic_scan`, `_carry_from_finished_stage` | 1194-1687 | frozen `php`, W PGF/buoyancy, PH tendency, boundary targets, final stage state |
| Large-step PGF/Coriolis/addtend | `src/gpuwrf/dynamics/core/rk_addtend_dry.py::large_step_horizontal_pgf`, `large_step_coriolis`, `rk_addtend_dry` | 193-495 | `ru_pgf`, `rv_pgf`, `ru_cor`, `rv_cor`, source-tendency folding into U/V/W/PH/T/MU |
| Acoustic U/V update | `src/gpuwrf/dynamics/core/acoustic.py::advance_uv_wrf`, `acoustic_substep_core` | 456-582, 633-903 | large-step U/V add, small-step pressure-gradient terms, emdiv terms, normal boundary work, `ru_m`, `rv_m` |
| MU/theta update | `src/gpuwrf/dynamics/mu_t_advance.py::advance_mu_t_wrf`, `_advance_mu_t_periodic`, `_advance_mu_t_specified_or_nested` | 81-344 | `dvdxi`, `dmdt`, `mu_tendency`, `mu_work_new`, `mudf`, `muts`, `muave`, `ww`, theta tendency |
| W/PH vertical dynamics | `src/gpuwrf/dynamics/core/advance_w.py::pg_buoy_w_dry`, `advance_w_wrf`; `src/gpuwrf/dynamics/core/rhs_ph.py::rhs_ph_wrf` | 88-380, 68-195 | `rw_tend_pg_buoy`, pressure/buoyancy pieces, `w_next`, PH tendency terms |
| Pressure/rho refresh | `src/gpuwrf/dynamics/core/calc_p_rho.py::calc_p_rho_wrf`, `calc_p_rho_step` | 95-173 | `p`, `alt`, `al`, `rho`, acoustic refresh after MU/theta/W/PH |
| Small-step prep/finish | `src/gpuwrf/dynamics/core/small_step_prep.py::small_step_prep_wrf`; `src/gpuwrf/dynamics/core/small_step_finish.py::small_step_finish_wrf` | 190-325, 22-77 | work-array construction and final perturbation-state reconstruction |
| Primitive advection fallback | `src/gpuwrf/dynamics/advection.py::compute_advection_tendencies` | 262-274 | primitive `u/v/w/theta/mu` advection path before flux replacement |
| Flux advection | `src/gpuwrf/dynamics/flux_advection.py::couple_velocities_periodic`, `advect_u_flux`, `advect_v_flux`, `advect_w_flux`, `advect_scalar_flux` | 171-247, 878-1150 | WRF-like coupled momentum/scalar advection terms |
| Explicit diffusion | `src/gpuwrf/dynamics/explicit_diffusion.py` diffusion helpers | 198-329, 1114-1335 | sixth-order, constant-K, Smagorinsky horizontal diffusion tendencies |
| Boundary/spec-relax | `src/gpuwrf/coupling/boundary_apply.py::apply_lateral_boundaries`, `apply_normal_bdy_work`, `nested_ph_relax_tendency`, `nested_w_relax_tendency`, `spec_bdyupdate_ph_inloop` | 148-230, 760-1033 | end-step boundary deltas, in-loop U/V normal boundary work, PH/W relax tendencies |
| Existing diagnostics | `src/gpuwrf/diagnostics/comprehensive_harness.py`; `tests/savepoint/README.md` | 50-62 and README scope | per-operator deltas exist, but current savepoint path is not a same-state WRF term oracle |

## WRF-Side Truth Savepoints

The term oracle must be produced by a validation-instrumented CPU WRF build or source checkout outside this repository's `src` tree. The instrumentation should be compact and patch-only, writing selected patches/columns for named terms at selected stages.

Candidate WRF routines/files to instrument:

| WRF area | Candidate WRF file/routine | Truth to save |
| --- | --- | --- |
| RK orchestration | `dyn_em/solve_em.F`, `dyn_em/module_em.F` around RK stage loop and `rk_tendency`/`rk_addtend_dry` | stage input state, stage weights, total `ru/rv/rw/ph/t/mu` tendencies before and after source folding |
| Horizontal PGF | `dyn_em/module_big_step_utilities_em.F::horizontal_pressure_gradient` | named `ru_pgf`, `rv_pgf` pieces and final arrays on U/V staggering |
| Coriolis | `dyn_em/module_big_step_utilities_em.F::coriolis` | `ru_cor`, `rv_cor`, optional `rw_cor`/vertical coupling if active |
| Advection | `dyn_em/module_advect_em.F` U/V/W/scalar advection routines | coupled fluxes and final `ru_adv`, `rv_adv`, `rw_adv`, scalar/mass advection |
| Diffusion | `dyn_em/module_diffusion_em.F` and related big-step utilities | sixth-order, constant-K, deformation/Smagorinsky contributions if enabled |
| Small step prep/finish | `dyn_em/module_small_step_em.F` small-step prep and finish routines | work arrays, pressure/rho/alt, final perturbation-state reconstruction |
| Acoustic U/V | `dyn_em/module_small_step_em.F::advance_uv` | large-step add, small-step PGF, emdiv damping, in-loop boundary work, `ru_m`, `rv_m` accumulation |
| MU/theta | `dyn_em/module_small_step_em.F::advance_mu_t` | `dvdxi`, `dmdt`, `mu_tendency`, `mu`, `muts`, `muave`, `mudf`, `ww`, theta tendency |
| W/PH | `dyn_em/module_small_step_em.F::advance_w`; big-step `pg_buoy_w`/`rhs_ph` routines | `rw_tend_pg_buoy`, vertical pressure/buoyancy pieces, `w`, `ph_tend` |
| Boundary/spec-relax | `dyn_em/module_bc_em.F`, `share/module_bc.F`, nested/spec bdy update routines | end-step lateral boundary deltas, in-loop normal-momentum boundary work, PH/W relax deltas |
| Physics source tendencies | `dyn_em/module_first_rk_step_part2.F`, `phys/module_pbl_driver.F`, radiation/microphysics/surface drivers as used by the case | raw WRF `RUBLTEN`, `RVBLTEN`, `RTHRATEN`, `RTHBLTEN`, `RTHCUTEN`, `RQ*`, `MUTTEN`, and any `*_tendf` entering `rk_addtend_dry` |

The WRF savepoint should include both inputs and outputs for each named routine. A compact NetCDF/Zarr/HDF5 artifact is sufficient if it stores:

- global domain shape and native stagger metadata
- selected lead, model step, RK stage, acoustic substep index
- horizontal patch bounds and selected-cell indices
- scalar and staggered input arrays with halo
- named WRF terms and post-update arrays
- namelist switches and timestep weights
- WRF git/build identifiers and instrumentation patch hash

## JAX CPU Harness Strategy

The future CPU harness should be additive and proof-local. It should import the current functions and run them against the WRF savepoint state without modifying production dycore code.

Recommended harness phases:

1. Load the WRF savepoint and reconstruct `State`, `GridSpec`, `DycoreMetrics`, boundary leaves, physics/source-tendency containers, and operational namelist flags exactly for the sampled step.
2. Assert static/base parity proof is green or record the exact static/base mismatch artifact being used as an exclusion.
3. Run the JAX functions on CPU for the same stage state:
   - `compute_advection_tendencies`
   - `_augment_large_step_tendencies`
   - `large_step_horizontal_pgf`
   - `large_step_coriolis`
   - diffusion helpers selected by `diff_opt/km_opt`
   - `rk_addtend_dry`
   - `small_step_prep_wrf`
   - `calc_p_rho_wrf`
   - `_acoustic_core_state_from_prep`
   - `advance_uv_wrf`
   - `advance_mu_t_wrf` through `advance_mu_t_core`
   - `advance_w_wrf`
   - `rhs_ph_wrf`
   - `calc_p_rho_step`
   - `small_step_finish_wrf`
   - boundary helpers if the selected point/substep touches in-loop or end-step boundary paths
4. Compare WRF and JAX term-by-term on native staggering over selected cells/faces and all vertical levels.
5. Emit a proof JSON and Markdown summary containing the first failing named term, stage, substep, level, cell, WRF value, JAX value, absolute error, relative error, and term contribution share.

The current `src/gpuwrf/diagnostics/comprehensive_harness.py` can inform wrapper shape, but it is insufficient by itself because it records JAX operator deltas rather than same-state CPU-WRF source-derived truth.

## Term Groups To Compare

The first pass should expose these named groups in this order:

1. Stage input parity: U/V/W/T/QVAPOR/P/PB/PH/PHB/MU/MUB, metrics, map factors, Coriolis fields, base-state fields, boundary leaves.
2. Mass coupling: `mass_u`, `mass_v`, `mass_h`, `mass_f`, coupled velocities `ru`, `rv`, `rom`.
3. Momentum advection: `ru_adv`, `rv_adv`, `rw_adv`, plus scalar/theta/mu advection used in the stage.
4. Diffusion: sixth-order, constant-K, deformation, and Smagorinsky terms according to active `diff_opt/km_opt`.
5. Large-step horizontal PGF: `ru_pgf`, `rv_pgf`.
6. Coriolis: `ru_cor`, `rv_cor`, vertical coupling terms if present.
7. Source-tendency folding: raw WRF `ru_tendf`, `rv_tendf`, `rw_tendf`, `ph_tendf`, `t_tendf`, `mu_tendf`, and JAX `DryPhysicsTendencies`.
8. Final large-step tendency before acoustic: total `u_tend`, `v_tend`, `w_tend`, `theta_tend`, `mu_tend`, `ph_tend`.
9. Small-step prep: work arrays, `p/alt/al`, `php`, `cqu/cqv/cqw`.
10. Acoustic U/V: large-step add, small-step pressure-gradient components, emdiv damping, normal boundary work, `u_work/v_work`, `ru_m/rv_m`.
11. MU/theta: `dvdxi`, `dmdt`, `mu_tendency`, `mu_work_new`, `muts`, `muave`, `mudf`, `ww`, theta tendency/update.
12. W/PH and pressure refresh: `rw_tend_pg_buoy`, `advance_w` pressure/buoyancy pieces, `ph_tend`, `p/rho/alt` refresh.
13. End-step boundary/spec-relax deltas: state before/after `apply_lateral_boundaries`.

## Predeclared Pass/Fail Rule

For each named term, compare WRF and JAX over the selected native cells/faces:

- all values must be finite
- shape and staggering metadata must match exactly
- zero/near-zero terms use absolute tolerance only
- nonzero terms use both absolute and relative tolerance, with the stricter failure reported

Recommended initial fp64 tolerances for the CPU proof:

- dimensionless/static metrics: exact or `abs <= 1e-12`
- velocity/mass/pressure tendencies: `abs <= max(1e-9, 1e-11 * term_scale)` and `rel <= 1e-9` for nonzero values
- one-substep updated prognostic arrays: `abs <= max(1e-8, 1e-10 * field_scale)` and `rel <= 1e-8`

These are strict enough to flag operator or cadence mistakes while allowing different CPU compiler/FMA ordering. The proof should report full error distributions and not tune tolerances after seeing the result.

## Data Dependencies

Required:

- Static/base parity proof from V014, including exact fields and any unresolved writer-only exclusions.
- CPU WRF history/savepoint source for the target case:
  - `wrfout` h8-h14 for selecting cells and validating post-step context
  - `wrfinput`, `wrfbdy`, and namelist for reconstructing static metrics, boundary leaves, and forcing cadence
  - validation-instrumented CPU WRF term savepoints for selected stage/substep patches
- JAX retained outputs from the prior probe only for selecting high-error cells, not as truth.
- Operational namelist values:
  - `dt_s`
  - `acoustic_substeps`
  - `run_physics`
  - `run_boundary`
  - `diff_opt`
  - `km_opt`
  - `use_flux_advection`
  - `force_fp64`
  - radiation cadence and `rad_rk_tendf`
  - top damping/lid switches
  - specified/nested boundary widths and intervals
- Prognostic and static fields:
  - `U`, `V`, `W`, `T`, `QVAPOR`, other active moisture species
  - `P`, `PB`, `PH`, `PHB`, `MU`, `MUB`
  - `ZNU`, `ZNW`, `DN`, `DNW`, `RDN`, `RDNW`, `FNM`, `FNP`, `DNW`
  - `C1H`, `C2H`, `C3H`, `C4H`, `C1F`, `C2F`, `C3F`, `C4F`
  - `MAPFAC_M`, `MAPFAC_U`, `MAPFAC_V`, reciprocal map factors as used
  - `F`, `E`, `SINALPHA`, `COSALPHA`, `RDX`, `RDY`
  - `HGT`, `LANDMASK`, `LU_INDEX`
  - boundary leaf arrays for specified/relax state and tendencies
  - raw physics/source tendencies if WRF provides them at the sampled step

## Current High-Risk Suspects To Measure, Not Fix

- Static/base fields are already known to have V014 mismatches; dynamic tendency work should wait until those are green or formally excluded as writer-only artifacts.
- `src/gpuwrf/runtime/operational_mode.py::_dry_physics_tendencies_from_state_delta` currently returns empty dry tendencies except the separate radiation-held-rate path. Current physics adapters mostly produce integrated state deltas and non-dry post-dycore updates. WRF commonly folds raw source tendencies into `rk_addtend_dry`; this is a prime cadence/source-folding suspect that must be measured with WRF `*_tendf` truth.
- `src/gpuwrf/dynamics/flux_advection.py` documents a periodic-oriented flux-advection path and boundary-order degradation out of scope. Interior h8-h14 cells reduce, but do not eliminate, the risk of accumulated boundary/stencil effects.
- `src/gpuwrf/coupling/boundary_apply.py::apply_normal_bdy_work` documents a calibrated in-loop normal boundary formulation. The wind/mass probe disfavored pure boundary-frame dominance, but in-loop boundary work should still be measured separately.
- A same-state patch harness may be invalid if cropped halos omit fields needed by WRF/JAX stencils or global boundary logic. Use full arrays first if memory allows; crop only after full-domain CPU execution is too slow.

## Recommended Next Sprint

If V014 static/base parity becomes green but U/V/P/PH/MU divergence remains, open a dedicated sprint:

`V015 same-state CPU tendency localization`

Sprint objective:

> Build a CPU-only same-state WRF term oracle and JAX comparison harness for h8-h14 ocean/low-terrain interior cells, identify the first failing named tendency/update term, and produce a proof object. Do not implement model fixes in the localization sprint.

Required deliverables:

- WRF validation hook patch or external instrumented-WRF note with build hash and exact routines touched.
- Cell-selection manifest for h8-h14 interior/ocean/low-terrain columns.
- CPU-only JAX comparison script and manifest.
- JSON proof containing per-term stats and first-failing term.
- Markdown review with pass/fail, root-cause candidate, and the exact next fix sprint proposal.

Exit criteria:

- If a named term fails, the next sprint fixes only that term/cadence with a new contract and reruns the same-state proof before full h8-h14 reruns.
- If all selected terms pass, broaden the sample before declaring operator formulas correct: additional leads, more columns, steep-terrain cells, boundary-near cells, and physics-radiation cadence cases.

## Inspection Commands Run

```bash
sed -n '1,260p' PROJECT_CONSTITUTION.md
sed -n '1,260p' AGENTS.md
sed -n '1,260p' .agent/skills/validating-physics/SKILL.md
sed -n '1,320p' .agent/sprints/2026-06-08-v014-static-metric-base-parity/sprint-contract.md
sed -n '1,320p' proofs/v014/wind_mass_divergence_probe.md
sed -n '1,320p' .agent/reviews/2026-06-08-v014-wind-mass-divergence-probe.md
rg --files src proofs scripts .agent | sed -n '1,260p'
rg -n "\b(ru|rv|rw|mu|mu_tend|mut|muu|muv|pgf|pressure|coriolis|coriol|advect|advection|diffus|spec|relax|boundary|large_step|rk|tendency|tend|source|rth|rqv)\b" src proofs scripts .agent -S
find src -maxdepth 5 -type f | sort | sed -n '1,260p'
git status --short
rg -n "^(def|class) " src/gpuwrf/dynamics src/gpuwrf/coupling src/gpuwrf/runtime src/gpuwrf/io src/gpuwrf/contracts -S
rg -n "horizontal_pressure_gradient|pressure_gradient|pgf|coriolis|coriol|advection|diffusion|diffus|rk_tendency|rk_addtend|ru_tend|rv_tend|mu_tend|rth|rqv|RUBLTEN|RVBLTEN|RTH|RQ|spec_zone|relax|apply_lateral|boundary|source" src/gpuwrf/dynamics src/gpuwrf/coupling src/gpuwrf/runtime src/gpuwrf/io -S
rg -n "savepoint|Savepoint|sp_|wrf|tendency|parity|oracle" tests scripts proofs src/gpuwrf -S
find tests -maxdepth 5 -type f | sort | sed -n '1,260p'
nl -ba src/gpuwrf/contracts/state.py | sed -n '300,430p'
nl -ba src/gpuwrf/dynamics/core/rk_addtend_dry.py | sed -n '1,560p'
nl -ba src/gpuwrf/dynamics/core/acoustic.py | sed -n '1,980p'
nl -ba src/gpuwrf/runtime/operational_mode.py | sed -n '1680,2365p'
nl -ba src/gpuwrf/dynamics/core/acoustic.py | sed -n '418,584p'
nl -ba src/gpuwrf/runtime/operational_mode.py | sed -n '1040,1695p'
nl -ba src/gpuwrf/runtime/operational_mode.py | sed -n '2340,3160p'
nl -ba src/gpuwrf/dynamics/core/dycore.py | sed -n '1,260p'
nl -ba src/gpuwrf/runtime/operational_mode.py | sed -n '2680,3060p'
nl -ba src/gpuwrf/dynamics/core/coupled.py | sed -n '1,360p'
nl -ba src/gpuwrf/dynamics/core/small_step_prep.py | sed -n '1,360p'
nl -ba src/gpuwrf/dynamics/core/small_step_finish.py | sed -n '1,280p'
sed -n '1,260p' tests/savepoint/README.md
nl -ba tests/savepoint/test_m6b5_dycore_step_parity.py | sed -n '1,360p'
nl -ba tests/savepoint/test_m6b6_coupled_step_parity.py | sed -n '1,420p'
nl -ba scripts/m6b6_coupled_step_compare.py | sed -n '1,420p'
nl -ba tests/savepoint/test_dycore_100_steps.py | sed -n '1,360p'
nl -ba tests/savepoint/test_diagnostic_harness.py | sed -n '1,360p'
nl -ba src/gpuwrf/diagnostics/comprehensive_harness.py | sed -n '1,420p'
nl -ba scripts/diagnostic_operator_term_budget_tracer.py | sed -n '1,360p'
nl -ba src/gpuwrf/runtime/operational_mode.py | sed -n '1,220p'
nl -ba src/gpuwrf/dynamics/flux_advection.py | sed -n '1,260p'
nl -ba src/gpuwrf/dynamics/flux_advection.py | sed -n '749,1185p'
nl -ba src/gpuwrf/dynamics/explicit_diffusion.py | sed -n '198,360p'
nl -ba src/gpuwrf/dynamics/explicit_diffusion.py | sed -n '1114,1335p'
nl -ba src/gpuwrf/dynamics/advection.py | sed -n '1,310p'
nl -ba src/gpuwrf/coupling/boundary_apply.py | sed -n '1,230p'
nl -ba src/gpuwrf/coupling/boundary_apply.py | sed -n '390,620p'
nl -ba src/gpuwrf/coupling/boundary_apply.py | sed -n '760,1048p'
nl -ba src/gpuwrf/dynamics/mu_t_advance.py | sed -n '1,360p'
nl -ba src/gpuwrf/dynamics/core/advance_w.py | sed -n '1,380p'
nl -ba src/gpuwrf/dynamics/core/rhs_ph.py | sed -n '1,320p'
nl -ba src/gpuwrf/dynamics/core/calc_p_rho.py | sed -n '1,260p'
find proofs/v014 -maxdepth 1 -type f -printf '%f\n' | sort
find .agent/reviews -maxdepth 1 -type f -printf '%f\n' | sort | sed -n '1,220p'
rg -n "class OperationalNamelist|use_flux_advection|diff_opt|km_opt|run_boundary|run_physics|dt_s|acoustic_substeps|rad_rk_tendf" src/gpuwrf/runtime/operational_mode.py -S
rg -n "def build_replay_case|OperationalNamelist.from_grid|run_forecast_operational|force_fp64|use_flux_advection|run_boundary|run_physics" src/gpuwrf/integration src/gpuwrf/runtime tests proofs scripts -S
```

The two `nl` commands for removed/nonexistent `tests/savepoint/test_m6b5_dycore_step_parity.py` and `tests/savepoint/test_m6b6_coupled_step_parity.py` failed with "No such file or directory"; that confirmed the current savepoint harness has moved to the files listed above.
