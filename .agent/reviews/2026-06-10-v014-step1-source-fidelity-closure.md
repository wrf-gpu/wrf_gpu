# Review: V0.14 Step-1 Source-Fidelity Closure

Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`.

Production change is narrow: `rad_rk_tendf=1` source-leaf mode now carries MYNN `rqvblten` and applies WRF `conv_t_tendf_to_moist`; the default `rad_rk_tendf=0` branch is preserved.

The strict Step-1 proof does not close: after-conv residual max_abs `2457.578397008898`, rmse `21.364579991779515`.

Accepted remaining blocker: JAX MYNN driver/kernel source outputs are too weak before dry-source coupling: both mass-coupled RTHBLTEN and qv source are about an order of magnitude below WRF at Step 1.

Next proof/fix route: Emit one WRF MYNN driver hook at Step 1 around module_bl_mynnedmf_driver: input columns/fluxes/turbulence state immediately before mynnedmf, raw dth1/dqv1 immediately after mynnedmf_post_run, and the module_em mass-scaled RTHBLTEN/RQVBLTEN. Then compare that exact boundary to JAX _mynn_column_from_state/step_mynn_pbl_column outputs.
