# V014 Wind/Mass Divergence Probe

Generated UTC: `2026-06-08T22:25:36.307569+00:00`

CPU-only wrfout anatomy probe for Case 3. This is not an equivalence pass and it does not run the model.

## Inputs

- GPU retained wrfouts: `/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- CPU-WRF truth: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- common leads: `1` to `24` h (`24` files)
- JAX_PLATFORMS during run: `cpu`

## Overall Native-Shape Differences

| Field | RMSE | Bias | MAE | Max abs | n |
|---|---:|---:|---:|---:|---:|
| `U` | 4.612114 | 2.719197 | 3.435568 | 22.841482 | 11151360 |
| `V` | 5.830290 | 3.703455 | 4.445072 | 18.132606 | 11249568 |
| `W` | 0.128292 | -0.000226 | 0.030668 | 12.175193 | 11333520 |
| `T` | 2.268375 | 0.758029 | 1.458843 | 10.471375 | 11081664 |
| `QVAPOR` | 0.000950 | -0.000000 | 0.000485 | 0.006696 | 11081664 |
| `P` | 228.121789 | -147.639395 | 147.866907 | 1286.162903 | 11081664 |
| `PH` | 336.207958 | -244.366599 | 251.462388 | 926.572510 | 11333520 |
| `MU` | 273.821210 | -242.799263 | 249.634186 | 1500.875427 | 251856 |
| `MUB` | 58.768664 | 3.189832 | 7.621791 | 1115.210938 | 251856 |
| `PB` | 28.642487 | 0.960325 | 2.294614 | 1111.718750 | 11081664 |
| `PHB` | 45.352539 | -0.640949 | 3.594300 | 2237.942383 | 11333520 |
| `U10` | 2.068389 | -0.944459 | 1.647574 | 11.969265 | 251856 |
| `V10` | 2.523741 | 1.036291 | 1.858821 | 16.034206 | 251856 |
| `T2` | 0.994190 | -0.050914 | 0.488782 | 10.372650 | 251856 |
| `PSFC` | 525.287772 | -504.513309 | 505.904082 | 1892.890625 | 251856 |

## Lead Window h10-h14

| Field | RMSE | Bias | MAE | Max abs | n |
|---|---:|---:|---:|---:|---:|
| `U` | 5.944339 | 4.160025 | 4.511536 | 22.841482 | 2323200 |
| `V` | 7.150577 | 5.816985 | 5.982461 | 18.132606 | 2343660 |
| `W` | 0.130955 | -0.003166 | 0.035927 | 12.175193 | 2361150 |
| `T` | 2.281452 | 0.785795 | 1.619931 | 9.671448 | 2308680 |
| `QVAPOR` | 0.000932 | 0.000016 | 0.000490 | 0.005667 | 2308680 |
| `P` | 219.211129 | -148.928077 | 148.952003 | 1250.266357 | 2308680 |
| `PH` | 312.627535 | -219.221807 | 234.964087 | 801.530762 | 2361150 |
| `MU` | 238.528460 | -222.672868 | 223.030113 | 1467.483398 | 52470 |
| `MUB` | 58.768665 | 3.189828 | 7.621791 | 1115.210938 | 52470 |
| `PB` | 28.642488 | 0.960325 | 2.294615 | 1111.718750 | 2308680 |
| `PHB` | 45.352539 | -0.640948 | 3.594300 | 2237.942383 | 2361150 |
| `U10` | 2.030984 | -0.337551 | 1.628286 | 11.969265 | 52470 |
| `V10` | 3.720731 | 2.785053 | 3.017545 | 14.536606 | 52470 |
| `T2` | 1.319676 | 0.200315 | 0.548830 | 10.372650 | 52470 |
| `PSFC` | 503.750545 | -489.743522 | 491.309374 | 1809.437500 | 52470 |

## h10-h14 Lead Anatomy

| Lead h | V10 RMSE | V10 Bias | U10 RMSE | PSFC RMSE | U RMSE | V RMSE | P RMSE | PH RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3.335 | 2.184 | 2.107 | 616.081 | 5.432 | 5.757 | 281.885 | 339.735 |
| 11 | 4.287 | 3.511 | 2.006 | 500.825 | 5.961 | 7.142 | 220.169 | 290.988 |
| 12 | 3.923 | 3.082 | 1.767 | 419.274 | 6.196 | 7.390 | 174.270 | 285.358 |
| 13 | 3.563 | 2.627 | 2.015 | 456.380 | 6.146 | 7.670 | 191.108 | 310.341 |
| 14 | 3.410 | 2.522 | 2.231 | 504.350 | 5.956 | 7.619 | 213.170 | 332.934 |

## Splits

### V10 Spatial Splits

| Split | Bin | RMSE | Bias | n |
|---|---|---:|---:|---:|
| `elevation_ocean` | `ocean` | 2.545 | 1.086 | 233424 |
| `elevation_ocean` | `land_0_300m` | 2.414 | 0.471 | 8448 |
| `elevation_ocean` | `land_300_1000m` | 2.340 | 0.481 | 6672 |
| `elevation_ocean` | `land_gt_1000m` | 1.443 | 0.123 | 3312 |
| `quadrant` | `NW` | 3.036 | 1.757 | 62568 |
| `quadrant` | `NE` | 1.586 | 0.294 | 63360 |
| `quadrant` | `SW` | 3.197 | 1.474 | 63360 |
| `quadrant` | `SE` | 1.877 | 0.624 | 62568 |
| `boundary` | `frame_5cells` | 2.541 | 1.332 | 51600 |
| `boundary` | `interior_excluding_5cell_frame` | 2.519 | 0.960 | 200256 |

## Correlations

Low-level correlations use mass-grid diagnostics: U/V are horizontally destaggered, PH is vertically destaggered.

| Pair | Pooled r | h10-h14 r |
|---|---:|---:|
| `dV10__dV_k0` | 0.998 | 0.996 |
| `dU10__dU_k0` | 0.998 | 0.997 |
| `dV10__dPSFC` | -0.290 | -0.209 |
| `dV10__dP_k0` | -0.312 | -0.244 |
| `dV10__dPH_k0` | 0.219 | 0.068 |
| `dPSFC__dP_k0` | 0.916 | 0.868 |
| `dPSFC__dPH_k0` | -0.076 | -0.038 |

## Vertical Coupling

| k | corr(dV10,dV) | corr(dU10,dU) | corr(dPSFC,dP) | corr(dPSFC,dPH) |
|---:|---:|---:|---:|---:|
| 0 | 0.998 | 0.998 | 0.916 | -0.076 |
| 1 | 0.992 | 0.993 | 0.916 | -0.068 |
| 2 | 0.988 | 0.985 | 0.916 | -0.052 |
| 3 | 0.975 | 0.965 | 0.916 | -0.024 |
| 4 | 0.935 | 0.913 | 0.916 | -0.004 |
| 5 | 0.852 | 0.796 | 0.915 | -0.016 |
| 6 | 0.727 | 0.609 | 0.915 | -0.047 |
| 7 | 0.607 | 0.482 | 0.914 | -0.075 |
| 8 | 0.563 | 0.487 | 0.913 | -0.096 |
| 9 | 0.455 | 0.510 | 0.911 | -0.110 |
| 10 | 0.229 | 0.432 | 0.909 | -0.114 |
| 11 | 0.188 | 0.237 | 0.905 | -0.100 |

## Ranked Root-Cause Hypotheses

### 1. Prognostic wind-column divergence with a near-surface projection, coupled to mass/geopotential error and peaking around h10-h14.

- verdict: `favored_by_this_probe`
- evidence for:
  - V10 pooled RMSE 2.524 m/s and h10-h14 RMSE 3.721 m/s are above the 1.5 m/s equivalence envelope.
  - Native 3D wind errors are larger than surface diagnostic errors: U RMSE 4.612 m/s, V RMSE 5.830 m/s.
  - Surface wind is coupled to low-level prognostic wind: corr(dV10,dV_k0)=0.998, corr(dU10,dU_k0)=0.998; by-level dV coupling starts at 0.998.
  - Mass fields are not quiet: PSFC RMSE 525.288 Pa, P RMSE 228.122 Pa, PH RMSE 336.208 m2/s2.
  - h10-h14 mass correlations include corr(dV10,dP_k0)=-0.244 and corr(dV10,dPH_k0)=0.068.
- evidence against / limits:
  - This wrfout-only probe cannot localize the first bad tendency component or prove whether wind drives mass or mass drives wind.

### 2. Static base-state or wrfout/grid-base reconstruction mismatch contributes to the mass/geopotential residual.

- verdict: `plausible_contributor_not_full_explanation`
- evidence for:
  - Compatible base-state fields are not identical: MUB RMSE 58.769 Pa, PB RMSE 28.642 Pa, PHB RMSE 45.353 m2/s2.
  - MUB/PB/PHB statistics are essentially lead-invariant, so this is reproducible static structure rather than random forecast noise.
  - PSFC and low-level P are strongly coupled: corr(dPSFC,dP_k0)=0.916.
- evidence against / limits:
  - Dynamic fields are much larger and lead-window dependent: V RMSE 5.830 m/s and h10-h14 V10 RMSE 3.721 m/s.
  - The static base-state signal cannot by itself explain the h10-h14 V10 peak or the old case-to-case V10 bias sign changes.

### 3. Surface/PBL or source-tendency cadence feedback amplifies a real low-level wind error after the early leads.

- verdict: `plausible_but_not_proven`
- evidence for:
  - V10 bias/RMSE worsens in the h10-h14 window: pooled bias 1.036, h10-h14 bias 2.785.
  - Ocean V10 RMSE 2.545 m/s and low-land V10 RMSE 2.414 m/s show the failure is not only steep-terrain noise.
  - The surface wind error tracks low-level prognostic wind, so a near-surface feedback/cadence issue remains a viable owner.
- evidence against / limits:
  - T2 RMSE 0.994 K and QVAPOR RMSE 0.000950 kg/kg are much smaller relative to their known envelopes, arguing against broad thermodynamic/moisture blow-up.
  - No same-state component tendency comparison was run here, so this is not yet an implementation-localized diagnosis.

### 4. Boundary-frame forcing defect/regression dominates the V10 failure.

- verdict: `disfavored_by_this_probe`
- evidence for:
  - The 5-cell frame still has V10 RMSE 2.541 m/s.
- evidence against / limits:
  - Interior V10 RMSE remains 2.519 m/s; frame/interior RMSE ratio is 1.009.
  - Excluding the boundary frame does not collapse the error or flip this into a harmless edge-only artifact.

### 5. Pure 10 m diagnostic sign/formula bug.

- verdict: `disfavored_by_this_probe`
- evidence for:
  - The largest user-visible symptom is still U10/V10 grid error.
- evidence against / limits:
  - 3D U/V native RMSEs (4.612/5.830 m/s) exceed U10/V10 RMSEs (2.068/2.524 m/s).
  - Low-level coupling is substantial: corr(dV10,dV_k0)=0.998 and corr(dU10,dU_k0)=0.998.
  - The V10 bias changes with lead window in existing V014 evidence, which is not the shape of a single static sign bug.

### 6. Old absent Coriolis or post-step-only normal-boundary bug reappeared unchanged.

- verdict: `low_priority_unless_a_tendency_probe_contradicts`
- evidence for:
  - Wind/mass coupling is real, so momentum assembly remains the broad search space.
- evidence against / limits:
  - The prior-attribution sidecar established both old bugs are fixed ancestors of current HEAD.
  - This probe does not show a boundary-frame-dominated signature, and a wrfout-only anatomy cannot implicate a missing Coriolis term without component tendencies.

## Next Fix Probe

Run a CPU-only same-state tendency localization on the h8-h14 window, sampled over the ocean/low-terrain interior cells where V10 and PSFC both fail. The first target should split large-step momentum and mass terms into PGF, Coriolis, advection, diffusion, boundary/spec-relax, physics/source-tendency folding, and resulting `ru`/`rv`/`mu` updates.
