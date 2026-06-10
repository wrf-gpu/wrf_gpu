# Manager Closeout

## Outcome

The sprint is closed as ACCEPTED WITH NARROWER BLOCKER.

Fable/Mythos disproved the previous broad blocker
`NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_NOAHMP_LAND_TILE_ENERGY`. The JAX
NoahMP land-tile energy solve matches WRF exact NMPIN inputs, and the local
production defect was a missing moist-theta to dry-theta decoupling before
Exner conversion in `assemble_noahmp_forcing`.

New verdicts:

- `proofs/v014/noahmp_land_tile_energy_closure.json`:
  `NOAHMP_LAND_TILE_ENERGY_CLOSED_NARROWED_TO_RRTMG_RADIATION_FORCING`
- `proofs/v014/noahmp_step1_closure.json`:
  `NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_RADIATION_FORCING_INTO_NOAHMP`

## Proof Objects

- `.agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md`
- `proofs/v014/noahmp_land_tile_energy_closure.{py,json,md}`
- `proofs/v014/noahmp_step1_closure.{py,json,md}`
- `tests/test_noahmp_coupler.py::test_forcing_decouples_moist_theta_to_dry_air_temperature`

## Merge Decision:

Merge the NoahMP forcing fix, focused test, proof objects, review, sprint
closeout, and roadmap/checklist updates.

Do not merge unrelated dirty files such as `proofs/v060/sfclayrev1_savepoint_parity_report.json`.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/noahmp_step1_closure.py proofs/v014/noahmp_land_tile_energy_closure.py`
- `python -m json.tool` on both proof JSON files
- `git diff --check` on the touched sprint files
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-7 python proofs/v014/noahmp_land_tile_energy_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-7 python proofs/v014/noahmp_step1_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-7 pytest -q tests/test_noahmp_coupler.py tests/test_v014_mynn_surface_layer_regressions.py tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py tests/test_v014_mynn_coldstart_init.py`

The pytest subset reported `17 passed, 1 skipped`.

## Key Numbers

- WRF exact-input NoahMP FSH RMSE: `0.0007711640412822121 W/m2`.
- NoahMP `sfctmp` bias before fix: about `+4.060948005384932 K`.
- NoahMP `sfctmp` RMSE after fix: `0.0033152073451598748 K`.
- Land HFX residual after fix with JAX radiation: `7.596194039500723 W/m2`.
- Land HFX residual after WRF exact SWDOWN/GLW swap: `0.09694345956039467 W/m2`.
- Strict Step-1 after fix: max_abs `1489.5135568470864`, RMSE
  `12.146876720723487`; worst cell `(i=66, j=37, k=3)` is water.

## Scope Changes

The acceptable fallback path was used. Strict Step-1 did not become green, but
the sprint returned a narrower WRF-anchored blocker and a production fix for the
local NoahMP forcing bug.

## Lessons

Moist theta is a state/dycore/LBC representation detail. Physics adapters must
receive dry thermodynamic views where WRF does, then recouple on writeback. The
same class of bug is now suspected in `surface_layer.py`, and the broader audit
already identifies other consumers that need dry/moist boundary discipline.

## Next Sprint

Close the surface-layer/sfclay-MYNN water-path `theta_m -> theta_dry` boundary
with WRF-anchored proof and MYNN regression gates. In parallel or immediately
after, close RRTMG Step-1 forcing parity for GLW/SWDOWN/RTHRATEN using a
temporary WRF forcing hook or record a formal scoped demotion.
