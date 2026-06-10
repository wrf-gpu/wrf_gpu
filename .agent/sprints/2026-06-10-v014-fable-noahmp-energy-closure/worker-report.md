# Worker Report

Summary: Fable/Mythos closed the claimed NoahMP land-tile energy/HFX blocker as
a production fix plus WRF-anchored proof. The JAX NoahMP energy solve is exact
when driven by WRF NMPIN, and the actual bug was that
`assemble_noahmp_forcing` treated WRF moist potential temperature `theta_m` as
dry potential temperature when converting to lowest-level air temperature.

Files changed:

- `src/gpuwrf/physics/noahmp_coupler.py`
- `tests/test_noahmp_coupler.py`
- `proofs/v014/noahmp_land_tile_energy_closure.{py,json,md}`
- `proofs/v014/noahmp_step1_closure.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md`

Key proof results:

- NoahMP energy solve with WRF exact inputs: FSH RMSE `0.0007711640412822121`
  W/m2.
- `sfctmp` bug: about `+4.06 K` before fix, `0.0033152073451598748 K` RMSE
  after decoupling `theta_m -> theta_dry`.
- Land HFX residual after fix: `7.596194039500723 W/m2`; with WRF exact
  SWDOWN/GLW it collapses to `0.09694345956039467 W/m2`.
- Strict Step-1 remains red at max_abs `1489.5135568470864`, RMSE
  `12.146876720723487`; worst cell is water, so NoahMP is no longer the
  limiter.

Commands run are listed in
`.agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md`.
