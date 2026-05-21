# M6-S7 Tier-4 Probtest Tolerance Freeze

Status: BLOCKED
Prototype label: M6 prototype; full ensemble at M7
Freeze time UTC: 2026-05-21T16:24:07+00:00

## Choices Frozen Before Held-Out Candidate

- Sample: 3 deterministic historical Gen2 wrf_l3 day-members (20260430_18z, 20260502_18z, 20260520_18z); this is not a perturbed ensemble.
- Held-out exclusion: 20260519_18z was excluded before tolerance derivation because it is the M6-S2 pinned GPU candidate day.
- Variables: U10, V10, T2, qv2, precip
- Leads: 6h, 12h, 24h
- Strata: canary, land, sea, elevation_band_0, elevation_band_1, elevation_band_2, elevation_band_3, elevation_band_4, elevation_band_5
- Tolerance factor: k = 1.96 (1.96 approximates a two-sided 95% normal interval for this M6 prototype.)
- Variance method: per-grid-cell sample variance across members with ddof=1; RMS of per-grid-cell member standard deviation; tolerance = k * sigma_rms_member_std.
- Precipitation: (RAINC + RAINNC + optional RAINSH at lead) - same components at lead 0.

## Prototype Limits

This is an M6 prototype only; full ensemble at M7. Ten deterministic day-members give a useful operational spread estimate, but precipitation tails and humidity regime coverage remain weak. The cost model in `artifacts/m6/tier4/cost_model.json` gates M7 ensemble dispatch and currently treats the M6-S5 lifted-cap runtime as provisional unless the S5 verdict artifact is present.

## Candidate Separation

No held-out candidate field is needed to compute this file. Candidate validation is written separately after this report and `probtest_tolerances.json` exist, preserving the no-after-failure tolerance rule.

## Blockers

- expected 10 historical members, found 4 under /mnt/data/canairy_meteo/runs/wrf_l3
- excluded complete members with non-pinned d02 grid shape: 20260509_18z_l3_24h_20260511T190519Z has d02 shape (66, 120), expected pinned (66, 159)
