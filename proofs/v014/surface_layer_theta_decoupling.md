# V0.14 Surface-Layer Water-Path Moist-Theta Decoupling

Verdict: `WATER_PATH_MOIST_THETA_BUG_CONFIRMED_DRY_TAIR_DECOUPLING_CLOSES_SFCLAY_FLUX`.

## Air-temperature bias (moist theta_m Exner - dry t_air)
- water: bias `4.643305843927778` K, max_abs `5.520558019111945` K (n=`9726`).
- land:  bias `4.062949285727465` K, max_abs `4.990203697869617` K.

## sfclay HFX vs WRF PRE_NOAHMP (= WRF SFCLAY1D)
- WATER buggy(moist):       rmse `11.869274410392945`, bias `-10.323335817670003`, max_abs `36.68002302713165` W/m2.
- WATER fixed(t_air only):  rmse `1.3746585734389127` W/m2.
- WATER fixed(full phy_prep): rmse `0.011760911267731817`, max_abs `0.20705355688308913` W/m2.
- LAND  buggy(moist):       rmse `65.93799778716168` W/m2.
- LAND  fixed(full phy_prep): rmse `0.15981735137452419` W/m2.

## Strict worst cell (water, Fortran i=66 j=37 k=3)
- is_water `True` (xland `2.0`); t1d buggy `297.77413296020154` K -> fixed `293.13849680536316` K (bias `4.63563615483838` K); tsk `294.51077271` K.
- HFX buggy `-0.5190157386365953` -> tair_only `1.3356650742601874` -> full `1.0932045884737` vs WRF `1.0946348906` W/m2.
- UST buggy `0.010693643716870779` -> full `0.016466889808833846` vs WRF `0.016466887668`.

## kinematic theta_flux vs WRF (MYNN bottom BC)
- WATER buggy rmse `0.009814857942815237` -> full phy_prep `8.231550268241452e-06` K m/s.

## Fix
- `src/gpuwrf/coupling/noahmp_surface_hook.py`: _build_column_view supplies dry t_air = theta_dry*(p/p0)^kappa with theta_dry = state.theta/(1 + Rv/Rd*qv); surface layer + noahmp forcing consume it instead of re-deriving from moist theta_m.
