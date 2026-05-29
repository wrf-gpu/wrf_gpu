# Skamarock warm bubble verdict

Verdict: FAIL
Status: RAN_TO_COMPLETION

## Checks

| Check | Value | Threshold | Passed |
| --- | ---: | --- | --- |
| all_snapshots_finite | 1.0 | all snapshot arrays finite | True |
| theta_prime_max_500s | 2.016418436651236 | 0.5 <= max(theta prime) <= 2.5 K | True |
| max_abs_w_500s | 14.809335863020543 | 1 <= max(|w|) <= 30 m/s | True |
| thermal_rise_500s | 212.9650002401927 | positive-theta centroid rises by at least 500 m | False |
| horizontal_drift_500s | 1.8189894035458565e-12 | positive-theta centroid drift <= 250 m | True |
| relative_mass_drift | 0.0 | max relative dry-column mass drift <= 1e-8 | True |

## Evidence

- Device: cuda:0
- CPU affinity: [0, 1, 2, 3]
- Timesteps: 5000
- Snapshot seconds: [100.0, 250.0, 500.0]
- Plots: proofs/f7j/plots/warm_bubble_theta_prime_100s.ppm, proofs/f7j/plots/warm_bubble_theta_prime_250s.ppm, proofs/f7j/plots/warm_bubble_theta_prime_500s.ppm

## References

- https://www2.mmm.ucar.edu/people/skamarock/Papers/cv_20.pdf

The warm-bubble run failed at least one declared check; this is an honest dycore-correctness failure, not a reference pass.
