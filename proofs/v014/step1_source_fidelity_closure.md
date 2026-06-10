# V0.14 Step-1 Source-Fidelity Closure

Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`.

## Evidence

- Strict WRF after `conv_t_tendf_to_moist` vs current JAX dry `T_TENDF`: max_abs `2457.578397008898`, rmse `21.364579991779515`.
- JAX mass-coupled MYNN `RTHBLTEN` remains too weak: max_abs `260.83156991819124` vs WRF `2522.90576171875`.
- JAX mass-coupled MYNN qv source is also too weak: max_abs `0.045505018412171354` vs WRF `QV_TEND` `0.4930315017700195`.
- Available same-boundary scalar inputs are not the order-10 error: T max_abs `5.788684885033035e-05`, QV max_abs `5.969281098756885e-08`, P max_abs `0.0390625`.
- Forcing radiation only moves max_abs to `2454.113955669592`; held `RTHRATEN` is secondary.
- WRF qv/moist conversion is represented now and remains secondary: removing the WRF `QV_TEND` term would leave max_abs `231.55175407007107`.
- WRF oracle active sources close the accepted formula: max_abs `0.00016236981809925055`, rmse `8.089162788029723e-07`.

## Single Blocker

JAX MYNN driver/kernel source outputs are too weak before dry-source coupling: both mass-coupled RTHBLTEN and qv source are about an order of magnitude below WRF at Step 1.

## Fastest Next Route

Emit one WRF MYNN driver hook at Step 1 around module_bl_mynnedmf_driver: input columns/fluxes/turbulence state immediately before mynnedmf, raw dth1/dqv1 immediately after mynnedmf_post_run, and the module_em mass-scaled RTHBLTEN/RQVBLTEN. Then compare that exact boundary to JAX _mynn_column_from_state/step_mynn_pbl_column outputs.

Proof objects: `proofs/v014/step1_source_fidelity_closure.json`.
