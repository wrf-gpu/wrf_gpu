# V0.14 Same-State WRF Savepoint Feasibility

Generated: 2026-06-09
Mode: CPU-only inspection. No GPU, no WRF source edits, no repo `src/` edits, no Hermes.

## Verdict

The fastest reliable path is to use a disposable instrumented copy of:

`/home/user/src/wrf_pristine/WRF`

This tree has WRF v4.7.1 source plus built CPU executables. The implementation sprint should not patch it in place. It should copy/worktree it under a scratch output root, add an env-gated Fortran emitter, run Case 3 from `2026-05-01_18:00:00` to h10, and dump selected d02 cell patches around existing dycore routine boundaries.

Do not use `external/wrf_savepoint_patch` as-is. Its wrapper hook bodies are empty and its build scripts reference stale missing paths.

No restart shortcut was found for Case 3, so exact h10 source state requires an instrumented forward run from the initial condition unless the next sprint first creates a restart.

## Case

- Case id: `20260501_18z_l2_72h_20260519T173026Z`
- CPU truth history: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- Launch/input directory: `/mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z`
- Target h10 wrfout: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-02_04:00:00`
- GPU output for cell selection only: `/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- `namelist.input` SHA256: `1a4711a1df97dfd28ff72f512024c07b3463de7a942f04f4c5b679f2ef690e38`
- `wrfinput_d02` SHA256: `150108f807fbc441783b2b2f309d32c555380b4bd8d909bbca9fe94057ed3c34`

d02 geometry is `west_east=159`, `south_north=66`, `bottom_top=44`, with staggered dimensions `160/67/45` and `DT=6.0`.

The selected first target is h10, valid `2026-05-02T04:00:00Z`. The 24 selected mass-grid cells and native stagger context should be read from `proofs/v014/dynamic_field_attribution.json`; the compact cell list is also embedded in `proofs/v014/same_state_wrf_savepoint_feasibility.json`.

## WRF Source And Build Availability

| Path | Role | Build status | Notes |
| --- | --- | --- | --- |
| `/home/user/src/wrf_pristine/WRF` | Fastest buildable candidate | Built | Git head `f52c197ed39d12e087d02c50f412d90d418f6186`, `v4.7.1-dirty`; `main/wrf.exe` and `main/real.exe` present. GNU/gfortran serial build linked against `/home/user/miniconda3/envs/wrfbuild`. Use only through a disposable copy. |
| `/home/user/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF` | Historical source path from Case 3 `rsl.error.0000` | Not built | Exists and matches original run provenance path, but no `configure.wrf` or `main/wrf.exe` was found. Use for provenance/source comparison. |
| `/home/user/src/wrf_ideal_f7i/WRF` | Source reference only | Not built | No executable. Not useful for fastest h10 savepoint generation. |
| `/home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF` | Old patch path | Missing | Stale. |
| `/home/user/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` | Old executable path | Missing | Stale. |

Important caveat: `/home/user/src/wrf_pristine/WRF` is dirty and already contains unrelated oracle/probe edits, including old hardcoded idealized dumps in `dyn_em/solve_em.F` and `phys/module_wrfgpu2_oracle.F`. The next sprint should copy it, record the base diff, and apply a new V0.14 patch in that copy only.

## Existing Scripts And Tests

Reusable patterns:

- `src/gpuwrf/validation/savepoint_io.py`: HDF5 savepoint metadata and payload hashing pattern.
- `src/gpuwrf/validation/savepoint_schema.py`: schema-validation pattern, but current `VALID_OPERATORS` is too narrow for V0.14 term groups.
- `scripts/m6b0r_jax_vs_wrf_compare.py`: comparison/report pattern only.
- `/home/user/src/wrf_pristine/WRF/phys/module_wrfgpu2_oracle.F`: useful raw-binary plus sidecar Fortran emission pattern.

Not sufficient as source truth:

- `scripts/m6b0r_wrf_savepoint_extract.py` extracts from wrfout and computes `calc_coef_w` in Python; this is not WRF source-derived dycore term truth.
- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` has empty hook bodies.
- `external/wrf_savepoint_patch/build.sh` and `build_relinked.sh` point at stale missing trees/executables.
- Current savepoint tests are useful CPU scaffolding but do not emit per-RK/acoustic source-derived WRF term savepoints.

## Candidate Instrumentation Points

Start at routine-boundary snapshots in `solve_em.F`; only split into inner routines after the first failing block is localized.

Primary orchestration hooks:

- `/home/user/src/wrf_pristine/WRF/dyn_em/solve_em.F::solve_em`, line 3.
- `rk_tendency` call, line 882.
- `rk_addtend_dry` call, line 968.
- `small_step_prep` call, line 1090.
- `calc_p_rho` calls, lines 1110 and 1624.
- `advance_uv` call, line 1280.
- `advance_mu_t` call, line 1393.
- `advance_w` call, line 1500.
- `small_step_finish` call, line 1825.
- `spec_bdyupdate` calls for U/V/T/MU/MUTS/W around lines 1346, 1356, 1462, 1472, 1482, and 1609.

Inner split candidates:

- Large-step: `dyn_em/module_em.F::rk_tendency` line 190, `rk_addtend_dry` line 959.
- Pressure/Coriolis/geopotential: `dyn_em/module_big_step_utilities_em.F::{rhs_ph,horizontal_pressure_gradient,pg_buoy_w,coriolis,phy_prep,set_tend}` at lines 1365, 2183, 2419, 3640, 4730, 5792.
- Small-step: `dyn_em/module_small_step_em.F::{small_step_prep,small_step_finish,calc_p_rho,calc_coef_w,advance_uv,advance_mu_t,advance_w}` at lines 16, 295, 438, 570, 654, 969, 1178.
- Advection: `dyn_em/module_advect_em.F::{advect_u,advect_v,advect_scalar,advect_w,advect_scalar_pd,advect_scalar_mono}` at lines 126, 1530, 3029, 4364, 6069, 9495.
- Diffusion: `dyn_em/module_diffusion_em.F::{cal_deform_and_div,horizontal_diffusion_u_2,horizontal_diffusion_v_2,horizontal_diffusion_w_2,horizontal_diffusion_s,vertical_diffusion_s,phy_bc,compute_diff_metrics}` at lines 17, 3118, 3323, 3519, 3711, 4789, 5901, 6882.
- Boundary/spec-relax: `share/module_bc.F::{set_physical_bc2d,set_physical_bc3d,spec_bdyupdate,flow_dep_bdy}` at lines 202, 651, 1955, 2335.
- Physics/source tendency folding: `dyn_em/module_first_rk_step_part1.F`, `dyn_em/module_first_rk_step_part2.F`, and focused `phys/module_physics_addtendc.F` `add_a2a` sites if source-tendency folding is implicated.

## Minimal Patch Strategy

1. Create a disposable WRF copy under a scratch path such as `/mnt/data/wrf_gpu2/v014_same_state_wrf/WRF`.
2. Copy the Case 3 run directory to a scratch run directory and reduce the copied namelist to stop at h10 first.
3. Add a small env-gated emitter module in the WRF copy, using raw little-endian f64 payloads plus JSON sidecars. This avoids HDF5 linking changes inside WRF.
4. Add first hooks in `solve_em.F` only. Dump pre/post snapshots around the large-step and small-step calls listed above.
5. Run a marker validation before full emission. It must prove d02 h10 step mapping, domain id, native index/stagger handling, and one field patch against h10 wrfout.
6. Emit h10 selected-cell patches with halo 8, all relevant native stagger indices, and the planned vertical levels. Include all RK stages and the acoustic substeps needed to separate `advance_uv`, `advance_mu_t`, `advance_w`, and pressure/density updates.
7. Postprocess raw payloads into HDF5 or manifest-backed arrays using a V0.14 proof-local schema.
8. Only if the first hook layer localizes a broad block, patch the inner module for that block.

Recommended env contract:

- `WRFGPU2_SAMESTATE=1`
- `WRFGPU2_SAMESTATE_ROOT=/mnt/data/wrf_gpu2/v014_same_state_wrf_savepoints/...`
- `WRFGPU2_SAMESTATE_GRID=2`
- `WRFGPU2_SAMESTATE_START_STEP` / `WRFGPU2_SAMESTATE_END_STEP`
- `WRFGPU2_SAMESTATE_RK_STAGES=1,2,3`
- `WRFGPU2_SAMESTATE_ACOUSTIC_SUBSTEPS=first,last` or explicit ids after marker validation
- `WRFGPU2_SAMESTATE_CELLS=proofs/v014/dynamic_field_attribution.json`
- `WRFGPU2_SAMESTATE_HALO=8`

Expected h10 d02 own-step mapping is near `10h * 3600 / 6 = 6000`, but this must be verified from WRF `grid%itimestep` and domain time before accepting savepoints.

## Expected Artifact Schema

Use schema id `v014-same-state-wrf-term-savepoint-v1`.

Runtime output can be raw binary plus sidecars; the proof artifact should postprocess to HDF5 or a manifest-backed array directory. Required metadata:

- Case/source/build: case id, domain, valid time, lead, WRF source path, git head/describe/status, executable path and SHA256, build mode, compiler, compile flags, patch SHA256.
- Inputs: namelist SHA256, `wrfinput`/`wrfbdy` checksums, run directory, source `rsl` path.
- Timing: `grid%itimestep`, lead seconds, d02 dt, RK stage, RK step weight, acoustic substep, routine name, event name, pre/post marker.
- Selection: selected cells, mass/U/V/W/PH native stagger indices, patch bounds, halo radius, vertical levels, tile/rank owner.
- Variables: name, role, term group, routine provenance, units, stagger, dtype, shape, memory order, index origin, checksum.

First-pass fields should include state inputs `U,V,W,T,QVAPOR,P,PB,PH,PHB,MU,MUB`; metrics/maps such as `ZNU,ZNW,DN,DNW,RDN,RDNW,C1*,C2*,MAPFAC_*,F,E,SINALPHA,COSALPHA,RDX,RDY`; and tendencies/work arrays such as `ru_tend,rv_tend,rw_tend,ph_tend,t_tend,mu_tend,ru_tendf,rv_tendf,rw_tendf,ph_tendf,t_tendf,mu_tendf,p,al,rho,ww,muave,muts,mudf,ru_m,rv_m,ww_m,t_2ave`.

Term groups:

- `stage_input`
- `mass_coupling`
- `advection`
- `diffusion`
- `pressure_gradient`
- `coriolis`
- `source_tendency_folding`
- `large_step_total`
- `small_step_prep`
- `acoustic_uv`
- `mu_theta`
- `w_ph_pressure`
- `boundary_spec_relax`
- `stage_output`

## Timing Estimate

From Case 3 `rsl.error.0000`:

- d02 timing lines: 43,200
- Warm median per d02 step: 0.13411 s
- Warm mean per d02 step: 0.151967 s
- Warm p90 per d02 step: 0.19338 s
- Median d02 compute per forecast hour: 80.466 s
- Estimated h10 d02 compute: 804.66 s

The original run used 28 MPI tasks. The currently built `/home/user/src/wrf_pristine/WRF/main/wrf.exe` appears serial, so a serial h10 run may be materially slower. A dmpar rebuild in the disposable copy is the better fidelity/performance option if the next sprint can spend build time.

## Risks

- No Case 3 restart files were found, so h10 requires forward integration from the initial time.
- The fastest built tree is dirty and serial, while the original truth run was 28-rank dmpar from the historical source path.
- Stale savepoint patch scaffolding can look useful but does not emit real dycore truth.
- Cropped patches can miss stencil/boundary context; use halo 8 and validate native stagger indexing before accepting data.
- MPI/tile ownership can duplicate or omit cells unless files carry rank/tile owner metadata or are post-merged.
- Nested-domain step mapping can be off by one; marker validation is mandatory.
- Physics source-tendency folding is broad and cadence-dependent; instrument it only after the first hook layer implicates it.
- Current repo validation schema operator names do not cover V0.14 term groups without a separate schema-extension contract.

## Next Implementation Sprint Contract Outline

Title: V0.14 same-state WRF source term savepoint generator.

Objective: generate CPU-WRF source-derived d02 h10 term savepoints for the selected 24 dynamic divergence cells, then compare them to the existing JAX CPU same-state term path without using JAX self-truth.

Constraints:

- CPU only.
- No GPU performance claims.
- No edits to repo `src` unless a separate schema-extension contract is approved.
- Do not patch `/home/user/src/wrf_pristine/WRF` in place.
- All truth must come from instrumented WRF source execution, not wrfout interpolation or Python recomputation.

Acceptance gates:

- Record exact WRF source path, executable hash, patch hash, namelist/input checksums, and step/time mapping.
- Marker field patch matches h10 wrfout at native indices within serialization tolerance.
- Savepoints include pre/post data for the requested high-value term groups or explicitly document postponed groups.
- Comparison report identifies the first failing h10 term group or states that no tested group reproduces the divergence.
- No source-derived claim relies on JAX-vs-JAX self-comparison.
