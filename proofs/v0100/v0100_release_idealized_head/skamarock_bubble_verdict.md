# Skamarock warm bubble verdict

Verdict: PASS
Status: RAN_TO_COMPLETION

## Checks

| Check | Value | Threshold | Passed |
| --- | ---: | --- | --- |
| all_snapshots_finite | 1.0 | all snapshot arrays finite | True |
| theta_prime_max_500s | 1.9200603177468452 | 0.5 <= max(theta prime) <= 2.5 K | True |
| max_abs_w_500s | 11.680188113242943 | 1 <= max(|w|) <= 30 m/s | True |
| thermal_rise_500s | 1924.347589651457 | positive-theta centroid rises by at least 500 m | True |
| horizontal_drift_500s | 0.0 | positive-theta centroid drift <= 250 m | True |
| relative_mass_drift | 0.0 | max relative dry-column mass drift <= 1e-8 | True |

## Evidence

- Device: cuda:0
- CPU affinity: [0, 1, 2, 3]
- Timesteps: 5000
- Snapshot seconds: [100.0, 250.0, 500.0]
- Plots: proofs/v0100/v0100_release_idealized_head/plots/warm_bubble_theta_prime_100s.ppm, proofs/v0100/v0100_release_idealized_head/plots/warm_bubble_theta_prime_250s.ppm, proofs/v0100/v0100_release_idealized_head/plots/warm_bubble_theta_prime_500s.ppm

## References

- https://www2.mmm.ucar.edu/people/skamarock/Papers/cv_20.pdf

The warm bubble rose coherently with bounded theta prime, active vertical motion, symmetry, and conserved dry mass under the declared checks.
