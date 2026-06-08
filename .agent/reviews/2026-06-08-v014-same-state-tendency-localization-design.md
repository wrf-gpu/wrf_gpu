# V014 Same-State Tendency Localization Design Review

## Objective

Design the next same-state tendency localization step after static/base parity, without editing `src` and without consuming GPU. The design targets h8-h14 ocean/low-terrain interior cells and compares same-state JAX CPU terms against CPU-WRF/source-derived truth for large-step momentum/mass tendencies, PGF, Coriolis, advection, diffusion, boundary/spec-relax, physics/source-tendency folding, and `ru`/`rv`/`mu` update assembly.

## Files Changed

- `proofs/v014/same_state_tendency_localization_plan.md`
- `proofs/v014/same_state_tendency_inventory.json`
- `.agent/reviews/2026-06-08-v014-same-state-tendency-localization-design.md`

No `src` files were edited. No GPU commands were run.

## Design Verdict

The smallest falsifiable next step is not a full forecast rerun and not another hourly-history compare. It is a CPU-only same-state term oracle:

1. Select 16-32 h8-h14 ocean/low-terrain interior columns from the existing Case 3 divergence, excluding at least an 8-cell boundary frame.
2. Generate compact CPU WRF source-derived savepoints for one lead first, preferably h10: all three RK stages and first plus last acoustic substep for selected patches or full-domain arrays.
3. Reconstruct the exact same state and namelist switches in a proof-local JAX CPU harness.
4. Compare named terms and one-stage updates in this order: input parity, mass coupling, advection, diffusion, large-step PGF, Coriolis, source-tendency folding, final large-step tendency, small-step prep, acoustic U/V, MU/theta, W/PH/pressure refresh, and boundary/spec-relax deltas.
5. Use the first failing named term as the only target for a later fix sprint.

This satisfies the local validating-physics rule: dynamic claims need WRF/source truth, not a JAX-vs-JAX self-compare.

## Exact Candidate Files And Functions To Instrument Or Wrap

JAX CPU wrappers should target these current paths:

- `src/gpuwrf/runtime/operational_mode.py`
  - `_rk_scan_step`
  - `_augment_large_step_tendencies`
  - `_acoustic_core_state_from_prep`
  - `_acoustic_scan`
  - `_carry_from_finished_stage`
  - `_dry_physics_tendencies_from_state_delta`
  - `_physics_step_forcing`
  - `_apply_physics_non_dry_updates`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
  - `large_step_horizontal_pgf`
  - `large_step_coriolis`
  - `rk_addtend_dry`
- `src/gpuwrf/dynamics/core/acoustic.py`
  - `advance_uv_wrf`
  - `acoustic_substep_core`
- `src/gpuwrf/dynamics/mu_t_advance.py`
  - `advance_mu_t_wrf`
  - `_advance_mu_t_periodic`
  - `_advance_mu_t_specified_or_nested`
- `src/gpuwrf/dynamics/core/advance_w.py`
  - `pg_buoy_w_dry`
  - `advance_w_wrf`
- `src/gpuwrf/dynamics/core/rhs_ph.py`
  - `rhs_ph_wrf`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
  - `calc_p_rho_wrf`
  - `calc_p_rho_step`
- `src/gpuwrf/dynamics/core/small_step_prep.py`
  - `small_step_prep_wrf`
- `src/gpuwrf/dynamics/core/small_step_finish.py`
  - `small_step_finish_wrf`
- `src/gpuwrf/dynamics/advection.py`
  - `compute_advection_tendencies`
- `src/gpuwrf/dynamics/flux_advection.py`
  - `couple_velocities_periodic`
  - `advect_u_flux`
  - `advect_v_flux`
  - `advect_w_flux`
  - `advect_scalar_flux`
- `src/gpuwrf/dynamics/explicit_diffusion.py`
  - `sixth_order_diffusion_tendency`
  - `constant_k_diffusion_tendency`
  - `conservative_constant_k_diffusion_tendency`
  - `smag2d_horizontal_km`
  - `horizontal_diffusion_coord_scalar_tendency`
  - `horizontal_diffusion_coord_momentum_tendency`
- `src/gpuwrf/coupling/boundary_apply.py`
  - `apply_lateral_boundaries`
  - `apply_normal_bdy_work`
  - `nested_ph_relax_tendency`
  - `nested_w_relax_tendency`
  - `spec_bdyupdate_ph_inloop`

CPU WRF truth hooks should target the corresponding WRF source routines:

- `dyn_em/solve_em.F` and `dyn_em/module_em.F` around RK stage orchestration, `rk_tendency`, and `rk_addtend_dry`
- `dyn_em/module_big_step_utilities_em.F` for horizontal PGF, Coriolis, `pg_buoy_w`, and `rhs_ph`
- `dyn_em/module_advect_em.F` for U/V/W/scalar advection
- `dyn_em/module_diffusion_em.F` and related utilities for active diffusion terms
- `dyn_em/module_small_step_em.F` for `small_step_prep`, `advance_uv`, `advance_mu_t`, `advance_w`, `calc_p_rho`, and `small_step_finish`
- `dyn_em/module_bc_em.F` and `share/module_bc.F` for specified/relax boundary and in-loop boundary work
- `dyn_em/module_first_rk_step_part2.F` plus active physics drivers for raw WRF source tendencies entering `rk_addtend_dry`

## Data Dependencies

Required before executing the localization harness:

- V014 static/base parity proof must be green, or remaining mismatches must be formally excluded as writer-only artifacts.
- Case 3 CPU WRF `wrfout` h8-h14, `wrfinput`, `wrfbdy`, namelist, and build metadata.
- Compact validation-instrumented CPU WRF term savepoints for the selected lead/stage/substep.
- JAX retained wrfouts only for selecting high-error cells, not as same-state truth.
- Exact operational switches: `dt_s`, `acoustic_substeps`, `run_physics`, `run_boundary`, `diff_opt`, `km_opt`, `use_flux_advection`, `force_fp64`, radiation cadence, `rad_rk_tendf`, boundary widths/intervals, and top damping/lid switches.
- Prognostic/static fields: `U/V/W/T/QVAPOR/P/PB/PH/PHB/MU/MUB`, vertical coordinate arrays, base-state coefficients, map factors, Coriolis/rotation fields, `RDX/RDY`, `HGT`, `LANDMASK`, `LU_INDEX`, boundary leaves, and raw physics/source tendencies.

## Unresolved Risks

- Dynamic localization is premature if static/base parity is still red.
- Existing savepoint tests and comprehensive diagnostics are useful scaffolding but are not same-state CPU-WRF term oracles.
- Current `_dry_physics_tendencies_from_state_delta` returns empty dry tendencies except a separate radiation-held-rate path; WRF source-tendency folding may therefore be a cadence/source-path mismatch. This must be measured, not assumed.
- Cropped patches may miss stencil or boundary dependencies; full-domain CPU execution with cropped statistics is safer for the first proof if feasible.
- Interior cells reduce boundary risk but do not eliminate accumulated boundary/spec-relax influence by h8-h14.

## Proof Objects Produced

- `proofs/v014/same_state_tendency_localization_plan.md`
- `proofs/v014/same_state_tendency_inventory.json`
- `.agent/reviews/2026-06-08-v014-same-state-tendency-localization-design.md`

## Commands Run

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
python -m json.tool proofs/v014/same_state_tendency_inventory.json >/tmp/same_state_tendency_inventory.validated.json
git diff -- proofs/v014/same_state_tendency_localization_plan.md proofs/v014/same_state_tendency_inventory.json .agent/reviews/2026-06-08-v014-same-state-tendency-localization-design.md
git status --short -- proofs/v014/same_state_tendency_localization_plan.md proofs/v014/same_state_tendency_inventory.json .agent/reviews/2026-06-08-v014-same-state-tendency-localization-design.md
wc -l proofs/v014/same_state_tendency_localization_plan.md proofs/v014/same_state_tendency_inventory.json .agent/reviews/2026-06-08-v014-same-state-tendency-localization-design.md
git status --short
grep -nP '[^\x00-\x7F]' proofs/v014/same_state_tendency_localization_plan.md proofs/v014/same_state_tendency_inventory.json .agent/reviews/2026-06-08-v014-same-state-tendency-localization-design.md
python - <<'PY'
from pathlib import Path
for p in [Path('proofs/v014/same_state_tendency_localization_plan.md'), Path('proofs/v014/same_state_tendency_inventory.json'), Path('.agent/reviews/2026-06-08-v014-same-state-tendency-localization-design.md')]:
    print(f'{p}: {p.stat().st_size} bytes')
PY
```

The two `nl` commands for `tests/savepoint/test_m6b5_dycore_step_parity.py` and `tests/savepoint/test_m6b6_coupled_step_parity.py` failed because those files do not exist in this checkout.
The `grep -nP '[^\x00-\x7F]' ...` command exited with status 1 and no output, meaning no non-ASCII bytes were found in the new files.

## Recommended Next Sprint

If V014 static/base parity becomes green but U/V/P/PH/MU divergence remains, open:

`V015 same-state CPU tendency localization`

The sprint should build the WRF term savepoint oracle and proof-local JAX CPU comparison harness, identify the first failing named term, and stop before model fixes. The first later fix sprint should be scoped only to that failing term or cadence path and should rerun the same-state proof before full forecast validation.
