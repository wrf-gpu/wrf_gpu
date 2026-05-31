# Wind Skill — before/after Coriolis + persistence skill

The single largest forecast-quality fix of the release: the GPU dycore momentum
tendency was **missing the Coriolis force**. Adding the WRF-faithful Coriolis term
flipped the wrong-sign lower-column zonal wind and pushed winds past persistence.

Root-cause + fix proofs: `proofs/wind/case3_v10_momentum_budget_findings.md`,
`proofs/wind/coriolis_fix_verdict.md` (verdict `CORIOLIS_ADDED`, HEAD `5319b8d`).
Pre-Coriolis snapshot: `proofs/wind/revalidate_wind.json`.

## Before → after Coriolis — case3 (2026-05-21, L3 d02, +24 h)

Persistence skill (skill = 1 − GPU_RMSE/pers_RMSE; >0 = beats persistence).
Source: `proofs/wind/coriolis_case3_v10_budget.json`, `coriolis_fix_verdict.md`.

| metric | BEFORE (no Coriolis) | AFTER (Coriolis) |
|---|---:|---:|
| **V10 skill** | −0.132 | **+0.169** (beats persistence) |
| **U10 skill** | −0.003 | **+0.361** |

Lower-column zonal wind sign corrected (the Coriolis signature; truth = CPU-WRF):

| level k | gpu_u BEFORE | gpu_u AFTER | wrf_u |
|---:|---:|---:|---:|
| 0 | +1.12 (wrong sign) | +0.14 | −0.69 |
| 1 | +1.38 (wrong sign) | −0.13 (now negative) | −0.79 |
| 2 | +1.40 (wrong sign) | −0.33 (now negative) | −0.88 |
| 3 | +1.28 (wrong sign) | −0.79 | −1.01 |

## Before → after Coriolis — case2 (2026-05-09, L2 d02, 24 h) RMSE

Source: `proofs/wind/coriolis_case2_localize.json` vs baseline
`proofs/wind/gpu_wind_localize.json` (via `coriolis_fix_verdict.md`).

| field / region | BASE RMSE | CORIOLIS RMSE | Δ |
|---|---:|---:|---:|
| V10 water | 3.322 | **1.557** | −1.764 |
| V10 all | 3.275 | **1.540** | −1.735 |
| U10 water | 2.501 | **1.719** | −0.782 |
| U10 all | 2.539 | **1.705** | −0.834 |
| T2 land | 2.330 | **1.874** | −0.456 |
| T2 water | 0.872 | 1.022 | +0.150 |
| T2 all | 1.040 | 1.102 | +0.062 |

Both wind components improved ~30–53 % everywhere; T2 has a small water-side shift
(+0.15 K, second-order from the corrected wind advecting temperature), within
operational tolerance and improving on land.

## Post-Coriolis persistence skill in the full d02 validation

From the validated d02 result (`proofs/v010_validation/v010_d02_result.json`; see
`publish/tables/v010_d02_validation.md`), mean skill over scored leads:

| field | case1 full | case2 full | case3 full | case1 ten. | case2 ten. | case3 ten. |
|---|---:|---:|---:|---:|---:|---:|
| **U10** | +0.22/+0.25 | +0.43/+0.41 | +0.45 | +0.30/+0.33 | +0.46/+0.46 | +0.52 |
| **V10** | +0.14/+0.21 | +0.13/+0.22 | +0.23 | +0.10/+0.14 | +0.18/+0.23 | +0.51 |
| T2 | −0.14/−0.19 | +0.01/+0.02 | −0.30 | +0.26/+0.09 | +0.32/+0.23 | +0.11 |

(L2/L3 split shown where both exist.) **U10 and V10 beat persistence in every
case/region post-Coriolis.** T2 is mixed (full-domain often loses to a strong
low-error persistence baseline; skillful in the Tenerife box at most cases).

## Validation that the fix did not break the core

- **Idealized bit-identity (f=0):** warm bubble + Straka close-gate — **2 passed**
  (`max|ru_cor| = max|rv_cor| = 0.0` at f=0, so adding Coriolis is the identity on
  idealized cases). `tests/idealized/test_dycore_close_gate.py`.
- **24 h coupled stability + conservation:** `all_finite` / `physically_plausible`
  PASS, guards-off (`proofs/perf/coriolis_segscan_24h.json`). Coriolis does **zero
  work** by construction (force ⊥ velocity: u·(fv) + v·(−fu) = 0).
- **GPT-5.5 xhigh sidecar cross-check:** SIGN/STAGGER/COUPLED/CADENCE all PASS;
  "no bugs in sign/stagger/map factor; interior stencil is faithful"
  (`proofs/wind/gpt_sidecar_verdict.md`).

## Honest residual

The pre-Coriolis revalidation (`proofs/wind/revalidate_wind.json`, verdict
`WIND_PARTIAL`) recorded case3 V10 *losing* to persistence (skill −0.099); the
Coriolis fix is what turned case3 V10 positive (+0.169). Remaining wind work
(P0-6 real-terrain/map-factor/boundary closure, P1-4 MYNN completeness) is on the
post-0.1.0 roadmap. T2 trades a small water-side cost (+0.15 K) for the large wind
gain — net positive on case2, neutral-to-slightly-negative on case3.
