# V0.14 Pre-RK Input Boundary

Verdict: `PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`.

## Target

- Domain/step: `d02`, step `6000`, h10 valid time `2026-05-02T04:00:00Z`.
- WRF surface: after `grid%itimestep` increment in `solve_em.F`, before current-step physics/RK.
- JAX source: produced step-5999 full carry checkpoint.

## Comparison

- First mismatch: `T` max_abs `6.218735851548047` RMSE `4.638818160588427`.
- `T`: status `DIFF`, max_abs `6.218735851548047`, RMSE `4.638818160588427`.
- `P`: status `DIFF`, max_abs `589.6789731315657`, RMSE `526.4973831519894`.
- `PB`: status `DIFF`, max_abs `1047.015625`, RMSE `223.43483550580925`.
- `MU`: status `DIFF`, max_abs `267.01919069732367`, RMSE `195.8714431231374`.
- `MUB`: status `DIFF`, max_abs `1050.3046875`, RMSE `224.13660680282618`.

## Next Decision

Trace the JAX checkpoint/prestep carry producer and previous-step WRF/JAX update path.
