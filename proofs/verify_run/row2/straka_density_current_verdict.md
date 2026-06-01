# Straka density current verdict

Verdict: PASS
Status: RAN_TO_COMPLETION

## Checks

| Check | Value | Threshold | Passed |
| --- | ---: | --- | --- |
| all_snapshots_finite | 1.0 | all snapshot arrays finite | True |
| theta_prime_min_900s | -9.970995032353244 | -25 <= min(theta prime) <= -5 K | True |
| max_abs_w_900s | 14.574919073013664 | 1 <= max(|w|) <= 50 m/s | True |
| front_position_900s | 14150.0 | |front position - 15000 m| <= 2000 m | True |
| rotor_count_proxy_900s | 4.0 | 2 <= rotor proxy count <= 4 | True |
| relative_mass_drift | 2.2509471651212643e-09 | max relative dry-column mass drift <= 1e-8 | True |

## Evidence

- Device: cuda:0
- CPU affinity: [0, 1, 2, 3]
- Timesteps: 9000
- Snapshot seconds: [900.0]
- Plots: proofs/verify_run/row2/plots/density_current_theta_prime_900s.ppm

## References

- https://www2.mmm.ucar.edu/projects/srnwp_tests/density/density.html
- https://journals.ametsoc.org/view/journals/mwre/141/4/mwr-d-12-00144.1.xml

The density current matched the declared front-position, rotor-proxy, bounded-theta, active-motion, and mass checks.
