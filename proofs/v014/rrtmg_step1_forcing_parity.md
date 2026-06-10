# V0.14 RRTMG Step-1 Forcing Parity

Verdict: `RRTMG_STEP1_RESIDUAL_LOCALIZED_TO_CLEAR_SKY_DERIVED_RRTMG_BOUNDARY`.

## Boundary

- Exact residual boundary: Clear-sky RRTMG derived optical/gas/top-buffer profile or kernel boundary. Current WRF hooks prove GLW and RTHRATEN outputs differ, and exonerate gross clock, surface handoff, thermodynamic state, cloud occupancy, layer ordering, theta conversion, and mass coupling; they do not dump the exact WRF RRTMG derived columns needed to name the first divergent LW/SW quantity.
- Production fix obvious: `False`.
- Next action: temporary WRF RRTMG forcing hook for derived LW/SW optical/gas/top-buffer profiles, fluxes, and heating arrays.

## Key numbers

- GLW/LWDN vs WRF PRE_NOAHMP: bias `17.44070059852181` W/m2, RMSE `17.520282676800505`, max_abs `22.521139408469537`.
- RTHRATEN mass-coupled vs WRF part2: max_abs `19.425283200182427`, RMSE `2.4884141898276413`, bias `-0.27883526570760864`.
- SWDOWN midpoint remains close: RMSE `2.758969795939516` W/m2; accepted lead-0 contrast RMSE `56.43169584231224` W/m2.

## Exonerated Boundaries

- Clock/geometry: `NOT_PRIMARY`. Midpoint SWDOWN/SWNORM is close to WRF, while accepted lead-0 SWDOWN RMSE is much worse; LW GLW is independent of solar geometry.
- Surface/land handoff: `NOT_PRIMARY_FOR_GLW_RTHRATEN`. WRF GLW is dflx(1,1) and NoahMP receives LWDN=GLW; SWDOWN is already within a few W/m2. No WRF emissivity hook exists in this fixture, but surface emissivity is not the downward GLW handoff.
- Gross thermodynamics/cloud: `GROSS_STATE_EXONERATED_DERIVED_OPTICS_UNHOOKED`. Theta max_abs `5.788684885033035e-05`, p_total max_abs `0.0390625`, qv max_abs `5.969281098756885e-08`.
- Layer ordering: `NOT_PRIMARY`; vertical flip max_abs `37.926177717778316`.
- Theta conversion: `NOT_PRIMARY`; no-Exner max_abs `19.496191827004914`.
- Mass coupling: `NOT_PRIMARY`; JAX-vs-WRF mass_h max_abs `0.01818978786468506`, and using WRF mass leaves RTHRATEN max_abs `19.42528244567689`.

## Release Impact

- Block next NoahMP strict attempt: `False`.
- Block final v0.14 strict release if unresolved: `True`.
