# V0.14 Step-1 SFCLAY Boundary Fix

Verdict: `STEP1_SFCLAY_FIRST_CALL_FIXED_NEXT_BLOCKER_TSK_ZNT_SURFACE_INPUTS`.

## Production Fix

- Added WRF MYNN surface first-call semantics to `surface_layer(..., first_timestep=True)`:
  UST first guess, MOL=0, QSFC=qv/(1+qv), and Li_etal_2010 z/L seed.
- Threaded Step-1 flags through d02 replay and operational `_physics_step_forcing`; updated the Step-1 proof helpers.

## WRF-Anchored Evidence

- First-call land QSFC gate: max_abs `0.0`.
- All-domain QSFC diagnostic (water recomputes inside WRF): max_abs `0.011736815970716026`.
- UST rmse warm -> first: `0.08667703917523994` -> `0.02954126268295198`.
- theta-flux rmse warm -> first: `0.0213378637896416` -> `0.02315528543965535`.
- qv-flux rmse warm -> first: `1.9833425562981398e-05` -> `1.442591864492997e-05`.
- MYNN RTHBLTEN rmse warm -> first: `0.00018383323104506096` -> `0.00018372241771801064`.

## Strict Step-1

- after-conv `T_TENDF` max_abs `1497.6112467075195`, rmse `13.296448784742802`.

## Remaining Blocker

Narrow next blocker: WRF-anchored TSK/ZNT surface input sourcing before sfclay_mynn. Fastest next command is a tiny surface-driver hook around module_surface_driver/module_sf_mynn for incoming TSK/ZNT/UST/QSFC/MOL and outgoing UST/HFX/QFX/ZNT on the current d02 Step-1 case, then compare those exact arrays against JAX _surface_column_view inputs and diagnostics.

Proof objects: `proofs/v014/step1_sfclay_boundary_fix.json`.
