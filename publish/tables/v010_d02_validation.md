# v0.1.0 d02 (3 km) Validation — GPU vs nightly CPU-WRF

Source proof: `proofs/v010_validation/v010_d02_result.json`
(verdict `D02_VALIDATED`, `all_pass=true`, `no_blowup=true`, wall 4470.3 s,
base model = Coriolis-corrected dycore @ HEAD `5319b8d`, generated
2026-05-31T09:46:45Z). Reproduce the canonical tables with
`taskset -c 0-3 python proofs/v010_validation/render_table.py --result proofs/v010_validation/v010_d02_result.json`.

**Setup.** GPU initialised from each run's t=0 `wrfout_d02` snapshot; scored
gridded against the nightly CPU-WRF (Gen2 NVHPC build) `wrfout_d02` truth.
REUSE-ONLY — launches NO new WRF runs. Grid 159×66×44, dx = 3 km, 44 levels;
dt = 10 s, 10 acoustic substeps, RK3, RRTMG cadence 180, fp64.

- **3 distinct corpus days:** case1 = 2026-05-29 18z (freshest, strongest-flow);
  case2 = 2026-05-09 18z (independent weak/reversed-zonal regime, 20 d earlier);
  case3 = 2026-05-21 18z (moderate flow, the validated +1h/+3h continuity pin).
- **L2 / L3** are both the **d02 (3 km)** domain: L2 = the d02 output of the 72 h
  L2-WRF run (leads 6/12/24/48/72 h); L3 = the d02 output of the 24 h L3-WRF run
  (leads 6/12/24 h). case3 is L3-only (its L2 sibling is a partial history).
- **RMSE ceilings (pass gate):** T2 ≤ 4.0+1.0·(lead/24) K, U10 ≤ 5.0+1.0·(lead/24),
  V10 ≤ 6.0+1.0·(lead/24) m/s. All cases/levels: T2/U10/V10 PASS, no fail reasons.

## FULL DOMAIN — GPU-vs-nightly-WRF RMSE (bias), n=10494 pts

| case | lvl | lead h | T2 rmse (bias) K | U10 rmse (bias) m/s | V10 rmse (bias) m/s | PRECIP rmse mm | finite |
|---|---|---:|---:|---:|---:|---:|:--:|
| case1 | L2 | 6 | 1.88 (+1.42) | 1.51 (+1.03) | 1.70 (+1.20) | 0.006 | Y |
| case1 | L2 | 12 | 2.10 (+1.58) | 1.55 (+1.03) | 1.76 (+1.22) | 0.101 | Y |
| case1 | L2 | 24 | 1.34 (+1.18) | 1.54 (+1.02) | 2.07 (+1.61) | 0.345 | Y |
| case1 | L2 | 48 | 1.09 (+0.88) | 1.79 (+0.90) | 2.34 (+1.65) | 1.248 | Y |
| case1 | L2 | 72 | 1.06 (+0.69) | 1.80 (+1.15) | 2.38 (+1.75) | 1.562 | Y |
| case1 | L3 | 6 | 1.88 (+1.42) | 1.51 (+1.03) | 1.70 (+1.20) | 0.006 | Y |
| case1 | L3 | 12 | 2.10 (+1.58) | 1.55 (+1.03) | 1.76 (+1.22) | 0.101 | Y |
| case1 | L3 | 24 | 1.34 (+1.18) | 1.54 (+1.02) | 2.07 (+1.60) | 0.345 | Y |
| case2 | L2 | 6 | 1.71 (+1.41) | 1.14 (−0.45) | 1.39 (+0.99) | 0.002 | Y |
| case2 | L2 | 12 | 1.74 (+1.40) | 1.21 (−0.38) | 1.41 (+0.54) | 0.068 | Y |
| case2 | L2 | 24 | 1.10 (+0.93) | 1.71 (−0.89) | 1.54 (+0.71) | 1.143 | Y |
| case2 | L2 | 48 | 1.08 (+0.89) | 1.63 (−0.49) | 1.88 (+1.27) | 1.392 | Y |
| case2 | L2 | 72 | 1.18 (+1.03) | 1.24 (+0.55) | 1.57 (+1.14) | 1.589 | Y |
| case2 | L3 | 6 | 1.71 (+1.41) | 1.14 (−0.45) | 1.39 (+0.99) | 0.002 | Y |
| case2 | L3 | 12 | 1.74 (+1.40) | 1.21 (−0.38) | 1.41 (+0.54) | 0.067 | Y |
| case2 | L3 | 24 | 1.10 (+0.93) | 1.70 (−0.89) | 1.54 (+0.72) | 1.061 | Y |
| case3 | L3 | 6 | 1.96 (+1.41) | 0.92 (+0.15) | 1.95 (+1.47) | 0.001 | Y |
| case3 | L3 | 12 | 2.22 (+1.61) | 1.09 (+0.23) | 1.84 (+1.35) | 0.052 | Y |
| case3 | L3 | 24 | 1.17 (+0.90) | 1.52 (+0.69) | 2.47 (+1.81) | 0.196 | Y |

## TENERIFE BOX — RMSE (bias), n=955 pts (lat 27.9–28.7, lon −17.0 to −16.0)

| case | lvl | lead h | T2 rmse (bias) K | U10 rmse (bias) m/s | V10 rmse (bias) m/s | PRECIP rmse mm | finite |
|---|---|---:|---:|---:|---:|---:|:--:|
| case1 | L2 | 6 | 2.75 (+2.10) | 1.40 (+0.77) | 1.47 (+0.87) | 0.000 | Y |
| case1 | L2 | 12 | 3.12 (+2.35) | 1.82 (+1.07) | 1.76 (+0.87) | 0.103 | Y |
| case1 | L2 | 24 | 1.37 (+1.22) | 1.87 (+1.30) | 2.09 (+1.50) | 0.398 | Y |
| case1 | L2 | 48 | 0.96 (+0.73) | 2.21 (+1.21) | 2.43 (+1.73) | 1.436 | Y |
| case1 | L2 | 72 | 0.86 (+0.44) | 1.84 (+1.13) | 2.08 (+1.19) | 1.568 | Y |
| case1 | L3 | 6 | 2.75 (+2.10) | 1.40 (+0.77) | 1.47 (+0.87) | 0.000 | Y |
| case1 | L3 | 12 | 3.12 (+2.35) | 1.82 (+1.07) | 1.76 (+0.87) | 0.103 | Y |
| case1 | L3 | 24 | 1.37 (+1.22) | 1.87 (+1.30) | 2.09 (+1.50) | 0.398 | Y |
| case2 | L2 | 6 | 2.40 (+1.97) | 1.23 (−0.14) | 1.34 (+0.74) | 0.000 | Y |
| case2 | L2 | 12 | 2.37 (+1.89) | 1.22 (+0.13) | 1.62 (+0.31) | 0.052 | Y |
| case2 | L2 | 24 | 1.37 (+1.08) | 1.73 (−0.41) | 1.91 (+0.98) | 3.153 | Y |
| case2 | L2 | 48 | 1.03 (+0.73) | 1.98 (+0.41) | 2.30 (+1.70) | 3.619 | Y |
| case2 | L2 | 72 | 1.09 (+0.82) | 1.65 (+1.04) | 1.87 (+1.17) | 3.761 | Y |
| case2 | L3 | 6 | 2.40 (+1.97) | 1.24 (−0.14) | 1.34 (+0.74) | 0.000 | Y |
| case2 | L3 | 12 | 2.37 (+1.88) | 1.22 (+0.14) | 1.61 (+0.31) | 0.055 | Y |
| case2 | L3 | 24 | 1.37 (+1.08) | 1.73 (−0.37) | 1.93 (+0.96) | 2.912 | Y |
| case3 | L3 | 6 | 2.73 (+1.99) | 0.88 (−0.12) | 1.63 (+0.90) | 0.000 | Y |
| case3 | L3 | 12 | 2.87 (+1.96) | 1.00 (+0.34) | 1.54 (+0.87) | 0.004 | Y |
| case3 | L3 | 24 | 0.94 (+0.72) | 1.73 (+0.94) | 2.57 (+1.27) | 0.045 | Y |

## Skill vs persistence (skill = 1 − GPU_RMSE/pers_RMSE; >0 = GPU beats persistence)

Mean skill over scored leads, with win/tie/loss counts.

| case / lvl | region | T2 | U10 | V10 |
|---|---|---|---|---|
| case1/L2 | full | −0.14 (W1/T0/L4) | **+0.22 (W5/T0/L0)** | **+0.14 (W5/T0/L0)** |
| case1/L2 | tenerife | **+0.26 (W4/T1/L0)** | **+0.30 (W5/T0/L0)** | **+0.10 (W4/T1/L0)** |
| case1/L3 | full | −0.19 (W0/T0/L3) | **+0.25 (W3/T0/L0)** | **+0.21 (W3/T0/L0)** |
| case1/L3 | tenerife | +0.09 (W2/T1/L0) | **+0.33 (W3/T0/L0)** | **+0.14 (W3/T0/L0)** |
| case2/L2 | full | +0.01 (W1/T2/L2) | **+0.43 (W5/T0/L0)** | **+0.13 (W3/T1/L1)** |
| case2/L2 | tenerife | **+0.32 (W5/T0/L0)** | **+0.46 (W5/T0/L0)** | **+0.18 (W3/T1/L1)** |
| case2/L3 | full | +0.02 (W1/T1/L1) | **+0.41 (W3/T0/L0)** | **+0.22 (W2/T1/L0)** |
| case2/L3 | tenerife | **+0.23 (W3/T0/L0)** | **+0.46 (W3/T0/L0)** | **+0.23 (W2/T1/L0)** |
| case3/L3 | full | −0.30 (W0/T0/L3) | **+0.45 (W3/T0/L0)** | **+0.23 (W3/T0/L0)** |
| case3/L3 | tenerife | +0.11 (W1/T0/L2) | **+0.52 (W3/T0/L0)** | **+0.51 (W3/T0/L0)** |

**Reading.** U10 and V10 **beat persistence in every case/region** (post-Coriolis
win). T2 is more mixed — full-domain T2 often loses to persistence (a low-error,
hard-to-beat baseline for surface temperature at these leads) but is skillful in
the Tenerife box at most cases. PRECIP loses to persistence at every lead (PRECIP
RMSE is small in absolute terms — ≤ ~1.6 mm full / ≤ 3.8 mm Tenerife — but
persistence's "zero precip" baseline is very strong on these mostly-dry days);
PRECIP skill is reported for completeness, not claimed as a v0.1.0 win.

## No-blow-up (gpu_mean / gpu_std at final lead, all finite)

| case / lvl | lead | T2 mean/std K | U10 mean/std m/s | V10 mean/std m/s |
|---|---|---|---|---|
| case1/L2 | +72h | 294.8 / 1.1 | −2.0 / 2.8 | −5.4 / 3.0 |
| case1/L3 | +24h | 294.2 / 0.9 | −2.4 / 2.6 | −6.2 / 3.0 |
| case2/L2 | +72h | 293.4 / 1.4 | −0.6 / 1.2 | −2.8 / 1.7 |
| case2/L3 | +24h | 293.4 / 1.5 | +0.7 / 1.3 | −0.9 / 1.1 |
| case3/L3 | +24h | 294.6 / 1.2 | +0.2 / 2.4 | −4.6 / 2.8 |

All fields finite at every scored lead; physical-plausibility sanity bounds
(T2 ∈ [250,330] K, |U10|,|V10| ≤ 40 m/s) hold throughout the 72 h runs.
