# Memory Patch

Reviewer Status: ACCEPTED FOR LOCAL MEMORY / FUTURE-MANAGER UPDATE.

Recommended durable lesson:

Physics-adapter thermodynamics need an explicit representation boundary:

- State, LBC, and dycore transport use WRF moist potential temperature
  `theta_m`.
- Physics schemes that WRF feeds dry sensible temperature or dry potential
  temperature must receive a dry view created by dividing by
  `1 + R_v/R_d * qv` before Exner or tendency conversion.
- Any adapter writeback that returns dry tendencies into a moist-theta state
  must recouple explicitly.

This sprint closed one instance in `noahmp_coupler.assemble_noahmp_forcing`.
The same rule now gates the surface-layer/sfclay-MYNN water path and the broader
set of consumers listed in `proofs/v014/moist_theta_physics_consumer_audit.*`.

Operational caution:

Do not claim v0.14 grid parity from station TOST or land-only NoahMP evidence.
The strict Step-1 worst cell is water and remains red until the surface-layer
and RRTMG lanes are fixed or honestly scoped.
