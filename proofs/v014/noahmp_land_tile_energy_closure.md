# V0.14 Noah-MP Land-Tile Energy Closure

Verdict: `NOAHMP_LAND_TILE_ENERGY_CLOSED_NARROWED_TO_RRTMG_RADIATION_FORCING`.

## Bottom line

- The JAX Noah-MP energy **algorithm is exact**: fed WRF's exact per-column NMPIN, FSH rmse `0.0007711640412822121` W/m2, SSOIL rmse `0.00021926861325835808`, TRAD rmse `1.9806433154212107e-05` K. The prior 'NoahMP land-tile energy' narrowing is **refuted**.
- **Root cause (fixed, production):** `state.theta` is the WRF MOIST potential temperature; `assemble_noahmp_forcing` converted it to air temperature with a naive Exner, leaving the lowest-level air temperature `+4.061` K too warm (= the 1+R_v/R_d*q_v factor). After the decouple fix, sfctmp rmse vs WRF T_ML = `0.0033152073451598748` K.
- After the fix, land-tile HFX rmse = `7.596194039500723` W/m2; swapping WRF's exact SWDOWN/GLW in collapses it to `0.09694345956039467` W/m2 -> the remaining lane is the RRTMG radiation forcing (GLW bias `+14.72`, SWDOWN bias `+3.58` W/m2).

## Fix

- `src/gpuwrf/physics/noahmp_coupler.py` :: `assemble_noahmp_forcing` -- decouple theta_m -> theta_dry (divide by 1 + R_v/R_d*q_v) before the Exner conversion of the lowest-level air temperature.
- test: `tests/test_noahmp_coupler.py::test_forcing_decouples_moist_theta_to_dry_air_temperature`.

## Energy solve vs WRF (WRF-exact inputs, land cells)

- FSH `{'count': 768, 'max_abs': 0.004352692140855652, 'rmse': 0.0007711640412822121, 'bias': -7.871259174547917e-05, 'ref_max_abs': 370.61135864}`
- SSOIL `{'count': 768, 'max_abs': 0.0014285211937590248, 'rmse': 0.00021926861325835808, 'bias': -6.257409976953008e-05, 'ref_max_abs': 273.67233276}`
- TRAD `{'count': 768, 'max_abs': 9.640974212743458e-05, 'rmse': 1.9806433154212107e-05, 'bias': -7.046281868354025e-06, 'ref_max_abs': 305.0635376}`

## Flux closure (real overlay, land cells)

- post-fix JAX radiation: HFX `{'count': 768, 'max_abs': 21.47836176406662, 'rmse': 7.596194039500723, 'bias': 6.5071596980481035, 'ref_max_abs': 370.61135864}`
- post-fix + WRF radiation: HFX `{'count': 768, 'max_abs': 0.4191707355015808, 'rmse': 0.09694345956039467, 'bias': 0.05956669988592684, 'ref_max_abs': 370.61135864}`

## Ranked remaining lanes (both out of this sprint's scope)

1. RRTMG step-1 surface radiation forcing into Noah-MP: GLW bias +14.72 W/m2, SWDOWN bias +3.58 W/m2. With WRF's exact SWDOWN/GLW the land-tile HFX residual collapses 7.60 -> 0.097 W/m2 rmse.
   - next: RRTMG longwave/shortwave forcing-parity hook on the pinned tree (GLW uniform clear-sky bias); RRTMG production is frozen this sprint.
2. Surface-layer (sfclay/MYNN) air temperature uses the SAME moist-theta -> naive-Exner conversion (surface_layer._potential_to_temperature) -> ~+4 K warm t1d over ALL cells; affects the water tiles incl. the strict worst cell (i=66, j=37, water). Out of this sprint's file ownership.
   - next: Apply the identical theta_m -> theta_dry decoupling in surface_layer.py and RE-VALIDATE the MYNN d02 path (it may have been tuned with the moist value; needs its own sprint).

## Fastest next command

`Emit an RRTMG step-1 longwave/shortwave forcing hook on the pinned tree (GLW/SWDOWN at the surface, both clear-sky) and column-diff vs the JAX rrtmg seed to close the +14.7 W/m2 GLW / +3.6 W/m2 SWDOWN bias; in parallel apply the same theta_m->theta_dry decoupling in surface_layer.py and RE-VALIDATE the MYNN d02 path. Then rerun JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py.`
