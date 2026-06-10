# Reviewer Report

Decision: ACCEPT_AS_LOCAL_CORRECTNESS_FIX_AND_NARROWING.

The patch is narrow and WRF-sourced. It changes only the MYNN surface-column
algebra and grid-backed surface-column density view, keeps analytic/no-grid
fallback behavior, and does not introduce host/device transfers, dynamic shapes,
or broad dycore edits.

## Review Notes

- The `QVSH=QV1D/(1+QV1D)` change matches WRF virtual-theta semantics and is
  covered by a focused regression test.
- The first-timestep `BR` clamp uses WRF's narrower `[-2,2]` limit while warm
  calls retain `[-4,4]`.
- `_wrf_phy_prep_rho_from_state` mirrors WRF `rho=(1+qv)/alt` for the
  hypsometric nested path using float32-style arithmetic; the proof and focused
  test bound the reconstruction at the surface hook.
- The proof's surface residuals are below the sprint thresholds, but the strict
  Step-1 residual remains large, so the sprint correctly narrows rather than
  declares parity.

## Required Follow-Up

Open a GPT-5.5 xhigh MYNN source-coupling sprint. The next proof must add or
rerun a WRF `module_pbl_driver` / `module_bl_mynnedmf` raw-source hook after the
fixed surface outputs and compare exact MYNNEDMF input fluxes plus raw
post-driver `dth1/dqv1` against `mynn_adapter_with_source_leaves`.
