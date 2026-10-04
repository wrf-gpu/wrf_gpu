# equiv-T2 EDMF-ON full-run remeasure — VERDICT

Branch: `worker/opus/v020-tost-daytimefix` (consolidated daytime-T2 fix chain).
Case: d03 1km `20260521_18z_l3_24h_20260522T133443Z`, use_noahmp ON, fp64,
radt=30min (RADCAD=600, L1-fix-required), daytime leads 09z/12z/15z (+15/18/21h).
Run: `equiv_t2_diag.py` (edmf=True wired live in `physics_couplers.mynn_adapter`).

## VERDICT: HOLD — stable, but no operational improvement.

edmf=True is (a) **STABLE** in the full coupled run but (b) does **NOT** raise
QFX/LH toward WRF or reduce the daytime T2 bias. It is a near-no-op at the surface
and slightly dries near-surface qair (the wrong direction). Do NOT include in the
equivalence-TOST default; keep `edmf=False` operationally.

## (a) STABILITY — PASS

| | value |
|---|---|
| verdict | STABLE |
| worst max\|W\| over d03 | **6.44 m/s** @ early spinup (step 1200); per-lead max\|W\| 4.25 / 4.46 / 4.68 |
| NaN/Inf (w, theta, qv, ph, u, v) | none |
| physical (limit 50 m/s; surface-w failure mode was ~300 m/s) | True at every chunk |

No surface-w blow-up. The fix passes the mandatory full-run stability gate.

## (b) EFFECT — edmf=ON vs edmf=OFF baseline (daytime land-mean)

| lead | metric | edmf=OFF | edmf=ON | delta (on-off) | WRF |
|---|---|---|---|---|---|
| 09z | T2_bias K | +0.952 | +0.965 | +0.013 | — |
| 09z | LH W/m2 | 5.84 | 5.81 | -0.026 | 11.67 |
| 09z | QFX kg/m2/s | 2.325e-6 | 2.315e-6 | -1.0e-8 | 4.651e-6 |
| 09z | qair kg/kg | 0.008773 | 0.008769 | -4e-6 | 0.008813 |
| 12z | T2_bias K | +0.502 | +0.492 | -0.010 | — |
| 12z | LH W/m2 | 13.56 | 13.52 | -0.035 | 22.25 |
| 12z | QFX kg/m2/s | 5.401e-6 | 5.387e-6 | -1.4e-8 | 8.862e-6 |
| 12z | qair kg/kg | 0.008741 | 0.008675 | -6.6e-5 | 0.010915 (Q2) |
| 15z | T2_bias K | +0.665 | +0.684 | +0.018 | — |
| 15z | LH W/m2 | 14.62 | 14.59 | -0.030 | 23.00 |
| 15z | QFX kg/m2/s | 5.826e-6 | 5.814e-6 | -1.2e-8 | 9.162e-6 |
| 15z | qair kg/kg | 0.008828 | 0.008701 | -1.3e-4 | 0.011043 (Q2) |

The expected +7-11% QFX/LH rise did NOT appear. Deltas are ~0.2-0.4% and the wrong
sign on LH/QFX; near-surface qair is slightly DRIER with EDMF on.

## Why (consistent with the MYNN-EDMF FINDINGS honest caveat)

The +7-11% in the lane's `integration_mf_vs_ed.json` was measured under a
**responsive-QFX single-column closure** (surface flux = ch*wspd*(qsfc-qv0)): the MF
lowered near-surface RH -> larger surface-air vapor gradient -> more QFX. In the
**operational coupler** the surface QFX/LH/HFX are computed UPSTREAM by the WRF
revised surface layer + Noah-MP from the surface state; the MYNN-EDMF only
redistributes existing PBL moisture (it dries the lower PBL, moistens the
entrainment zone, conserves column water — `s_awqv(kts)=0` always). It therefore
cannot, by construction, raise the surface QFX the coupler already fixed. The
residual deficit is on the LAND-vapor / surface-exchange lane (EAH canopy vapor,
CHS), exactly as the FINDINGS said ("NOT the sole or dominant cause").

## Note on the wiring

The EDMF kernel `mynn_edmf.dmp_mf_columns` is a vmap over a SINGLE batch axis
`(B,nz)`; the operational coupler feeds `(ny,nx,nz)`. mynn_adapter now flattens
`(ny,nx)->ny*nx` before the call and reshapes back (commit 84edce1). With edmf back
to the default `edmf=False` this flatten/unflatten round-trips exactly (lossless),
so reverting the activation is a one-line change with no residue.
