# Straka density current verdict

Verdict: FAIL
Status: RAN_TO_COMPLETION

## Checks

| Check | Value | Threshold | Passed |
| --- | ---: | --- | --- |
| all_snapshots_finite | 1.0 | all snapshot arrays finite | True |
| theta_prime_min_900s | -8.081613050630438 | -25 <= min(theta prime) <= -5 K | True |
| max_abs_w_900s | 15.30414907967646 | 1 <= max(|w|) <= 50 m/s | True |
| front_position_900s | 14050.0 | |front position - 15000 m| <= 2000 m | True |
| rotor_count_proxy_900s | 3.0 | 2 <= rotor proxy count <= 4 | True |
| relative_mass_drift | 3.3792625872974946e-08 | max relative dry-column mass drift <= 1e-8 | False |

## Evidence

- Device: cuda:0
- CPU affinity: [0, 1, 2, 3]
- Timesteps: 9000
- Snapshot seconds: [900.0]
- Plots: proofs/f7n/plots/density_current_theta_prime_900s.ppm

## References

- https://www2.mmm.ucar.edu/projects/srnwp_tests/density/density.html
- https://journals.ametsoc.org/view/journals/mwre/141/4/mwr-d-12-00144.1.xml

The density-current run failed at least one declared check; this is an honest dycore-correctness failure, not a reference pass.
