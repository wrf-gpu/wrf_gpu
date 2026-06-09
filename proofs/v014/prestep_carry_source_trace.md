# V0.14 Prestep Carry Source Trace

Verdict: `PRODUCER_WRITES_BAD_FINAL_CARRY`.

## Summary

- Classification: `PRODUCER_WRITES_BAD_FINAL_CARRY`.
- Serialization: `CHECKPOINT_READ_WRITE_PRESERVES_TARGET_LEAVES`.
- Producer source: live nested replay from native domain load; not retained wrfout/restart.
- Current-step RK/acoustic was not run by this trace.

## Field Results

| field | checkpoint source | max abs | RMSE | closest inspected same-checkpoint candidate |
| --- | --- | ---: | ---: | --- |
| `T` | `carry.state.theta - 300 K, k=1` | 6.218735851548047 | 4.638818160588427 | `checkpoint_prestep_carry.t_2ave_minus_300` max_abs 3.677698918664362 |
| `P` | `carry.state.p_perturbation, k=1` | 589.6789731315657 | 526.4973831519894 | `checkpoint_prestep_carry.P` max_abs 589.6789731315657 |
| `PB` | `carry.state.p_total - carry.state.p_perturbation, k=1` | 1047.015625 | 223.43483550580925 | `checkpoint_prestep_carry.PB` max_abs 1047.015625 |
| `MU` | `carry.state.mu_perturbation` | 267.01919069732367 | 195.8714431231374 | `checkpoint_prestep_carry.mu_save` max_abs 266.9440572350254 |
| `MUB` | `carry.state.mu_total - carry.state.mu_perturbation` | 1050.3046875 | 224.13660680282618 | `checkpoint_prestep_carry.mu_total_minus_mu_save` max_abs 1050.2069387110678 |

## Decision

Open a narrow previous-step handoff bisection sprint: capture CPU-only JAX d02 carries at steps 5997, 5998, and 5999 immediately before/after _operational_force, _advance_chunk, and _carry_from_finished_stage, then compare State theta/p/mu target leaves plus t_2ave/t_save/mu_save/muts against the existing CPU-WRF pre-RK truth. Do not edit current-step RK/acoustic first.
