# Worker Report: V0.14 Fable NoahMP Step-1 Closure

Summary: Fable/Mythos closed the contracted NoahMP-disabled Step-1 blocker and
proved the remaining strict Step-1 red path is narrower:
`NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_NOAHMP_LAND_TILE_ENERGY`.

Main results:

- Step-1 proof builder now enables NoahMP with WRF-derived land/static state:
  `use_noahmp=True`, `sf_surface_physics=4`, and
  `inputs_have_noahmp_land=True`.
- Step-1 held radiation is seeded from the WRF-faithful `xtime + radt/2`
  convention; lead-0 was falsified by hook truth.
- Production bug fixed: `first_timestep` now reaches the sfclay run inside the
  NoahMP blend path. Without this, the WRF MYNN surface first-call semantics did
  not engage when NoahMP was active.
- Strict Step-1 remains red against the pinned one-run WRF truth:
  max_abs `1489.5135568470864`, RMSE `13.2001844004901`.
- The remaining leading blocker is the NoahMP land-tile surface energy solve,
  especially land theta/HFX at the worst cells; RRTMG GLW/RTHRATEN parity is
  secondary and parallelizable.

Proof objects:

- `proofs/v014/noahmp_step1_closure.{py,json,md}`
- rerun `proofs/v014/step1_mynn_source_coupling.{py,json,md}`
- rerun `proofs/v014/step1_surface_land_flux_handoff.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-fable-noahmp-step1-closure.md`
