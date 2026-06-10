# V0.14 GPT RRTMG Step-1 Forcing Parity Review

## Objective

Localize the secondary Step-1 RRTMG forcing residual without production source,
test, GPU, TOST, Switzerland, Grid-Delta, FP32, memory, or NoahMP proof edits.

## Files Changed

- `proofs/v014/rrtmg_step1_forcing_parity.py`
- `proofs/v014/rrtmg_step1_forcing_parity.json`
- `proofs/v014/rrtmg_step1_forcing_parity.md`
- `.agent/reviews/2026-06-10-v014-gpt-rrtmg-step1-forcing-parity.md`

## Commands Run

- `python -m py_compile proofs/v014/rrtmg_step1_forcing_parity.py` -> pass
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/rrtmg_step1_forcing_parity.py` -> pass
- `python -m json.tool proofs/v014/rrtmg_step1_forcing_parity.json >/tmp/rrtmg_step1_forcing_parity.validated.json` -> pass
- `git diff --check` -> pass

## Proof Objects

- `proofs/v014/rrtmg_step1_forcing_parity.json`
- `proofs/v014/rrtmg_step1_forcing_parity.md`

## Exact Residual Boundary

Verdict:
`RRTMG_STEP1_RESIDUAL_LOCALIZED_TO_CLEAR_SKY_DERIVED_RRTMG_BOUNDARY`.

Measured residuals:

- GLW/LWDN vs WRF PRE_NOAHMP: bias `17.44070059852181` W/m2, RMSE
  `17.520282676800505`, max_abs `22.521139408469537`.
- Mass-coupled RTHRATEN vs WRF part2: max_abs `19.425283200182427`,
  RMSE `2.4884141898276413`, bias `-0.27883526570760864`.
- SWDOWN midpoint convention remains close: RMSE `2.758969795939516` W/m2;
  accepted lead-0 contrast RMSE is `56.43169584231224` W/m2.

Exonerated:

- Clock/solar geometry: not primary.
- NoahMP/surface handoff: not primary for GLW/RTHRATEN. WRF LW sets
  `glw(i,j) = dflx(1,1)` and NoahMP receives `LWDN = GLW(I,J)`.
- Gross thermodynamic state: theta perturbation max_abs `5.788684885033035e-05`
  K, total pressure max_abs `0.0390625` Pa, qv max_abs
  `5.969281098756885e-08` kg/kg against WRF part2.
- Cloud occupancy: clear-sky, hydrometeor cloud sum zero.
- Layer ordering: vertical flip worsens RTHRATEN to max_abs
  `37.926177717778316`.
- Flux-to-theta conversion: skipping Exner does not improve max_abs
  (`19.496191827004914`).
- Mass coupling: JAX mass_h vs WRF mass_h max_abs `0.01818978786468506`;
  using WRF mass leaves RTHRATEN max_abs `19.42528244567689`.

Named remaining boundary:
WRF's derived RRTMG clear-sky optical/gas/top-buffer profile or downstream
kernel boundary. Current hooks prove the output residual and rule out the
gross boundaries above, but do not dump the exact derived WRF RRTMG columns.

## Recommended Fix

No production fix is obvious from the current hooks.

Next proof/fix step: add a temporary WRF RRTMG forcing hook, then compare before
editing production. For LW, dump `play/plev/tlay/tlev`, `h2ovmr/o3vmr` and trace
gases, `emis`, `cldfrac`, cloud paths/effective radii, `tauaer`, `dflx/uflx/hr`,
and clear-sky counterparts around `module_ra_rrtmg_lw.F:RRTMG_LWRAD`. For SW,
dump matching column optics, topographic correction, fluxes, and heating before
the `RTHRATENLW + RTHRATENSW` sum.

## Unresolved Risks

- The exact first divergent RRTMG derived quantity is not yet named because the
  WRF fixture lacks a radiation-column forcing hook.
- Prior v0.13 clear-sky LW oracle was close on a different real WRF snapshot, so
  this should not be generalized as a blanket LW-kernel failure without the
  Step-1 forcing hook.
- RRTMG remains secondary to the active NoahMP HFX blocker but must be closed or
  formally demoted before final v0.14 strict release.

## Blocking Recommendation

Do not block the next NoahMP-focused strict Step-1 attempt on this lane.
Do block final v0.14 strict release if this RRTMG residual remains unresolved or
undemoted by manager decision.
