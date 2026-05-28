# Skamarock warm bubble verdict

Verdict: BLOCKED
Status: BLOCKED_GPU_UNAVAILABLE

## Checks

| Check | Value | Threshold | Passed |
| --- | ---: | --- | --- |
| gpu_available | 0.0 | JAX GPU backend visible | False |
| initial_condition_finite | 1.0 | analytic IC arrays finite | True |

## Evidence

- Device: None
- CPU affinity: [0, 1, 2, 3]
- Timesteps: 0
- Snapshot seconds: [100.0, 250.0, 500.0]
- Plots: proofs/f2/plots/warm_bubble_initial_theta_prime.ppm

## References

- https://www2.mmm.ucar.edu/people/skamarock/Papers/cv_20.pdf

The analytic initial condition was generated, but the dycore integration was not run because no JAX GPU backend was visible.
