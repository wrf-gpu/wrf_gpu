# M7 Honest Speedup + Skill Verdict

## Honest Speedup

The apples-to-apples d02-only timing number is **50.20x**:

- GPU d02 24h end-to-end wall: **324.7756 s**
- CPU d02-only cumulative WRF timing: **16,305.3113 s**
- Ratio: **50.2048x**

The earlier **156.82x** claim should not be published. It came from a timing denominator path that double-counted mirrored WRF timing records and used an incomplete/ambiguous source run. The complete 20260521 Gen2 CPU run is `20260521_18z_l3_24h_20260522T133443Z`; its `namelist.output` contains zero `Timing for main` records, so the proof object uses de-duplicated sibling `rsl.error.0000` / `rsl.out.0000` records and states that caveat explicitly.

Additional denominators:

- CPU full-nest d01-d05 aggregate: **44,897.1417 s**, **138.24x**. Conservative aggregate, not apples-to-apples.
- CPU d01+d02 minimum physical subset: **33,329.1154 s**, **102.62x**. Includes CPU d01 work the GPU run did not execute.
- CPU d01-only context row: **17,023.8041 s**, **52.42x**. Context only, not a d02 comparison.

## Skill

The GPU forecast is **not within +/-20% of CPU skill** on T2/U10/V10. It is materially worse on every aggregate BIAS/RMSE/MAE comparison:

| Variable | Metric | GPU | CPU | Relative delta |
|---|---:|---:|---:|---:|
| T2 | RMSE | 7.8588 | 2.1487 | +265.75% |
| T2 | MAE | 5.7574 | 1.6807 | +242.56% |
| U10 | RMSE | 11.3111 | 2.3065 | +390.41% |
| U10 | MAE | 9.1696 | 1.7122 | +435.54% |
| V10 | RMSE | 9.4353 | 2.7523 | +242.81% |
| V10 | MAE | 7.4336 | 1.9725 | +276.86% |

The side-by-side scoring used 24 common valid times from **2026-05-21T19:00:00Z** through **2026-05-22T18:00:00Z**, 73 AEMET stations, and the same `gpuwrf.validation.forecast_vs_obs` interpolation/scoring scaffold for both GPU and CPU wrfouts.

## Publication-Ready?

**NO.**

The timing-only number to publish, if a timing caveat is explicitly allowed, is **50.20x GPU d02 24h end-to-end vs CPU d02-only cumulative WRF timing**. The combined M7 speedup-and-skill claim is **not publication-ready** because GPU skill is materially worse than CPU on T2/U10/V10 and fails the pre-declared +/-20% tolerance. The M7 closeout should be amended before any public claim.
