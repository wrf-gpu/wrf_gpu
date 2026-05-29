# Straka density current verdict

Verdict: FAIL
Status: RAN_TO_COMPLETION

## Checks

| Check | Value | Threshold | Passed |
| --- | ---: | --- | --- |
| all_snapshots_finite | 0.0 | all snapshot arrays finite | False |
| theta_prime_min_900s | None | -25 <= min(theta prime) <= -5 K | False |
| max_abs_w_900s | None | 1 <= max(|w|) <= 50 m/s | False |
| front_position_900s | None | |front position - 15000 m| <= 2000 m | False |
| rotor_count_proxy_900s | 0.0 | 2 <= rotor proxy count <= 4 | False |
| relative_mass_drift | None | max relative dry-column mass drift <= 1e-8 | False |

## Evidence

- Device: cuda:0
- CPU affinity: [0, 1, 2, 3]
- Timesteps: 9000
- Snapshot seconds: [900.0]
- Plots: proofs/f7h/plots/density_current_theta_prime_900s.ppm

## References

- https://www2.mmm.ucar.edu/projects/srnwp_tests/density/density.html
- https://journals.ametsoc.org/view/journals/mwre/141/4/mwr-d-12-00144.1.xml

The density-current run failed at least one declared check; this is an honest dycore-correctness failure, not a reference pass.
