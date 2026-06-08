# V0.14 Same-State Savepoint Request

## Verdict

Generated a CPU-WRF instrumentation request for h10 same-state savepoints: domain `d02`, run `20260501_18z_l2_72h_20260519T173026Z`, valid time `2026-05-02T04:00:00+00:00`, 24 selected mass-grid cells.

This is only a request manifest. It makes no equivalence or root-cause claim.

## Source

- Authoritative selected-cell proof: `proofs/v014/dynamic_field_attribution.json` sha256 `a8979ebd844d6aa6a05319bbfdaf548e119f2a8f0165c0ceed56c0854a84c020`
- Selection source: `proofs/v014/dynamic_field_attribution.json localization_manifest.selected_cells`; retained JAX output is used only for choosing cells, not as same-state truth.

## Request

- Instrument one real CPU-WRF large timestep at the h10 target and record the exact WRF model-step number, `Times`, `dt`, RK weights, and namelist switches.
- Save all three RK stages and the first plus last acoustic substep for each stage.
- Save native-staggered arrays, not destaggered diagnostics.
- Write full native vertical columns for each selected mass cell and adjacent U/V/W/PH faces. The first-probe reporting levels are `[0, 1, 2, 16, 17, 18, 24, 25, 26, 28, 29, 30, 31, 32]`, but they are not sufficient for stencil or vertical-coupling terms.
- Use the per-cell patch bounds in JSON. Bounds are zero-based stop-exclusive, with one-based inclusive WRF/Fortran translations included.

## Cells

1:(y=9,x=13), 2:(y=25,x=39), 3:(y=41,x=14), 4:(y=49,x=17), 5:(y=32,x=53), 6:(y=27,x=143), 7:(y=44,x=15), 8:(y=38,x=14), 9:(y=39,x=11), 10:(y=36,x=73), 11:(y=22,x=37), 12:(y=30,x=50), 13:(y=38,x=76), 14:(y=13,x=89), 15:(y=16,x=35), 16:(y=36,x=11), 17:(y=22,x=45), 18:(y=19,x=36), 19:(y=41,x=17), 20:(y=38,x=145), 21:(y=23,x=49), 22:(y=47,x=14), 23:(y=50,x=26), 24:(y=27,x=146)

Full lat/lon, patch bounds, native faces, and diagnostic context are in the JSON artifact.

## Term Groups

stage_input, mass_coupling, momentum_advection, scalar_theta_mu_advection, diffusion, horizontal_pgf, coriolis, source_tendency_folding, small_step_prep, acoustic_uv, mu_theta, w_ph, pressure_rho_refresh, boundary_spec_relax, final_stage_state

The JSON expands each group into timing, native grid, and requested term arrays.

## Expected Artifact

Preferred output is one compact NetCDF4/Zarr/HDF5 artifact named like `wrf_same_state_savepoints_20260501_18z_l2_72h_20260519T173026Z_d02_h10.<nc|zarr|h5>`.
It must carry global WRF build/namelist/instrumentation identifiers, the selected-cell manifest echo, native patch bounds, per-term dataset metadata, and companion build/run logs.

## Dependency

Sartre's WRF source/build feasibility artifact is present and should be used for the instrumentation tree/build choice.

- Feasibility JSON: `proofs/v014/same_state_wrf_savepoint_feasibility.json`
- Recommended path summary: Use a disposable instrumented copy of /home/enric/src/wrf_pristine/WRF, because it has a current WRF v4.7.1 source tree and built CPU executables. Run the Case 3 d01/d02 forecast from 2026-05-01_18:00:00 to h10 with a small env-gated Fortran emitter that dumps selected d02 cell patches around existing dycore routine boundaries. Do not use the old external/wrf_savepoint_patch as-is; its hooks are empty and its build paths are stale.
