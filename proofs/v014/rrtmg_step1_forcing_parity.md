# V0.14 RRTMG Step-1 Forcing Parity

Verdict: `RRTMG_STEP1_FORCING_PARITY_MATERIALLY_REDUCED_BY_DRY_THETA_INPUT_FIX`.

## Boundary

- Exact residual boundary: Dominant pre-fix boundary was WRF radiation_driver -> RRTMG_LWRAD input T3D=t: JAX built T from stored theta_m while WRF phy_prep passes dry theta temperature. The production owner is gpuwrf.coupling.physics_couplers._rrtmg_column_inputs. Remaining split LW/SW residual is bounded by proofs/v014/rrtmg_rthraten_closure.*.
- Production fix applied: `True`.
- Next action: Rerun the split WRF-oracle closure proof: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/rrtmg_rthraten_closure.py.

## Key numbers

- GLW/LWDN vs WRF PRE_NOAHMP: bias `0.3182764792020434` W/m2, RMSE `0.35152062180598787`, max_abs `1.2638192801302353`.
- RTHRATEN mass-coupled vs WRF part2: max_abs `2.798351397503893`, RMSE `0.3645729657536835`, bias `0.22941737095565604`.
- SWDOWN midpoint remains close: RMSE `2.736296679206838` W/m2; accepted lead-0 contrast RMSE `56.43169584231224` W/m2.

## Exonerated Boundaries

- Clock/geometry: `NOT_PRIMARY`. Midpoint SWDOWN/SWNORM is close to WRF, while accepted lead-0 SWDOWN RMSE is much worse; LW GLW is independent of solar geometry.
- Surface/land handoff: `NOT_PRIMARY_FOR_GLW_RTHRATEN`. WRF GLW is dflx(1,1) and NoahMP receives LWDN=GLW; SWDOWN is already within a few W/m2. No WRF emissivity hook exists in this fixture, but surface emissivity is not the downward GLW handoff.
- Gross thermodynamics/cloud: `GROSS_STATE_EXONERATED_DERIVED_OPTICS_UNHOOKED`. Theta max_abs `5.788684885033035e-05`, p_total max_abs `0.0390625`, qv max_abs `5.969281098756885e-08`.
- Layer ordering: `NOT_PRIMARY`; vertical flip max_abs `37.97668063305632`.
- Theta conversion: `NOT_PRIMARY`; no-Exner max_abs `3.0863258306336405`.
- Mass coupling: `NOT_PRIMARY`; JAX-vs-WRF mass_h max_abs `0.01818978786468506`, and using WRF mass leaves RTHRATEN max_abs `2.7983511916417285`.

## Release Impact

- Block next NoahMP strict attempt: `False`.
- Block final v0.14 strict release if unresolved: `True`.
