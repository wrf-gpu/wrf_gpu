# v0.1.0 d03 (1 km Tenerife) Status — boundary-pump fix + honest residual

Source proofs:
`proofs/v010_validation/d03_summary_run24h_v5fix.json`,
`proofs/v010_validation/d03_validation_run24h_v5fix.json`
(prior-iteration histories `..._v4.json` retained as self-correction evidence).

**Setup.** GPU d03 = 1 km Tenerife nest, gridded RMSE vs corpus CPU-WRF
`wrfout_d03` (1 km) truth. Run id `20260521_18z_l3_24h`, 24 forecast hours,
dt = 3 s, device cuda:0, all fields finite. Persistence baseline holds the t=0
corpus d03 surface state at every lead.

**Verdict (honest):** `D03_1KM_BOUNDED_FAIL` / `validation_status = FAIL`.
final-lead RMSE exceeds the d02-inherited 3.0 K T2 threshold (just barely: 3.01 K).
This is reported as a **bounded fail**, NOT a pass — v0.1.0 ships d03 as
"overnight d02-quality with a known daytime warm bias," not as validated.

## Boundary-pump self-correction (v4 → v5fix)

The pre-fix d03 (run24h_v4) had a lateral-boundary "pump" injecting energy at the
1 km nest edge, driving a ~6.8 K final-lead T2 error and gross wind blow-up. The
v5fix corrects the boundary handling. Same run, same truth, same scorer:

| Field (final lead +24h) | v4 (pre-fix) RMSE | v5fix RMSE | Δ |
|---|---:|---:|---:|
| **T2** (K) | 6.756 | **3.009** | **−3.75** |
| U10 (m/s) | 7.414 | **3.488** | −3.93 |
| V10 (m/s) | 8.387 | **4.403** | −3.98 |
| T2 bias (K) | +5.571 | +2.096 | −3.48 |

v4 verdict was the same `D03_1KM_BOUNDED_FAIL`, but with errors ~2× larger and a
runaway warm/wind signature; v5fix brings T2 to the d02 quality band and winds to
near-persistence.

## v5fix final-lead (+24h) scores vs persistence

| Field | RMSE | bias | persistence RMSE | skill | beats persistence? | within threshold? |
|---|---:|---:|---:|---:|:--:|:--:|
| **T2** (K) | 3.009 | +2.096 | 2.298 | −0.310 | No | No (thr 3.0) |
| **U10** (m/s) | 3.488 | +1.318 | 2.981 | −0.170 | No | Yes (thr 7.5) |
| **V10** (m/s) | 4.403 | −0.208 | 4.679 | +0.059 | **Yes** | Yes (thr 7.5) |
| RAINNC (mm) | 0.133 | +0.020 | ~0 | (n/a — near-dry) | No | Yes |

T2 mean RMSE over all 24 leads = **2.43 K** (min 1.39 K overnight, max 4.39 K
midday); final-lead T2 = 3.01 K.

## Persistence wins/losses across all 24 leads

| Field | wins | losses |
|---|---:|---:|
| T2 | 12 | 12 |
| U10 | 4 | 20 |
| V10 | **16** | 8 |

V10 beats persistence at most leads; U10 mostly loses at 1 km (persistence is a
strong baseline for the island's steady trade-wind zonal component); T2 is split
overnight-good / daytime-poor.

## Honest residual — daytime surface-flux over-flux (NOT fixed in v0.1.0)

The remaining error is a **daytime surface-flux over-flux**, shared with d02 (not a
d03-specific defect): sensible heat flux HFX runs ~3.7× too high at midday, driving
a **+2.9 K midday T2 warm bias**. Peak T2 mean-bias = **+2.97 K at 13z**
(2026-05-22 13:00 UTC); overnight T2 RMSE is ~1.4 K (good). This is a surface-layer
/ land-surface / radiation fidelity gap (P1-3 RRTMG slope/topo-shading + P0-3
prognostic Noah-MP + P1-4 MYNN), deferred to v0.1.x/v0.2.0; it is **not** a
boundary or dycore-stability problem.

**Pipeline:** 24/24 wrfout files written, all readable, 41 minimum variables
present (`proofs/v010_validation/wrfout_inventory.json`, status PASS); wall
1794 s total (74.8 s/forecast-hour).
