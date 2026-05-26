# Hypothesis Notes

## Matched

3. Operational skips a clamp/limiter. The step-11 bad cell was reproduced with `theta=2.44e12 K`, `qc=3.2233444e7 kg/kg`, and invalid pressure entering Thompson. A one-cell Thompson reproducer showed the large `qc` came from ice creation under invalid thermodynamics, then instant melt to cloud water.

## Partially matched

1. Thompson saturation adjustment overshoots. Not the direct source for the reproduced bad cell. The huge condensate appeared before the saturation-adjustment stage in the local one-cell trace.

5. Initial qc/qv inputs corrupted. The bad cell's previous-step `qv=1.193467e-4` and `qc=0` were not themselves explosive. Later localization found boundary/RK moisture NaNs elsewhere in the domain, so the operational coupling boundary needed a finite/admissible guard.

## Not matched

2. qv to qc conversion sign error. No sign inversion was found in the failing cell trace.

4. Latent heat coupling formula error. The enormous theta followed the enormous condensate; bounded condensation in the regression test gives `dtheta < 5 K`.

6. Time-step too large for Thompson. The run used `dt_s=10.0`; no evidence pointed to timestep size as the primary cause.
