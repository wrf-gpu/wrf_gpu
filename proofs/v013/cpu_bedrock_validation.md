# v0.13 BEDROCK CPU-only Validation Slice

**Positioning:** wrf_gpu is a fast GPU-native WRF-COMPATIBLE reimplementation, not a bit-true Fortran port. This slice proves the model RUNS STABLY, CONSERVES, and RESTARTS bit-identically (RUNS-stability), not Fortran EQUIVALENCE.

**Base:** branch `validation-cpu-bedrock` off d2314546 (v0.13 trunk tip). **Platform:** CPU-only, 8 threads cores 8-15, no GPU. **Result: 5 PASS / 0 FAIL / 1 SKIP.**

| Test | Result | Key number |
|------|--------|------------|
| Idealized warm bubble (Skamarock, fp64, 5000 steps) | PASS | rise 1924 m, max\|w\| 11.7 m/s, mass drift **0.0** |
| Idealized density current (Straka, fp64, 9000 steps) | PASS | front 14.15 km, 4 rotors, mass drift 2.25e-9 |
| Conservation budget (controlled, closed + open-LBC) | PASS | dry-mass/water/MSE residuals ~0 (tol 1e-12) |
| Restart bit-identity (full carry: state+land+rad) | PASS | bit-identical; corruption fails closed |
| Checkpoint + wrfrst-NetCDF roundtrip (2nd path) | PASS | bit-identical, WRF schema conformant |
| qke finiteness / NaN suppression | PASS | NaN suppressed, in-range unchanged |
| Live-dycore mass invariant (small grid) | SKIP | State.zeros is GPU-only by design (not a failure) |

**Most reassuring:** mass conservation is essentially exact — warm-bubble dry-column mass drift = 0.0, controlled budget residuals ~machine-eps, density-current 2.25e-9 over 9000 fp64 steps. **Biggest gap:** everything here is RUNS-stability only; physics EQUIVALENCE (TOST, forecast skill) and the live-coupled multi-step mass invariant remain GPU-campaign work.
