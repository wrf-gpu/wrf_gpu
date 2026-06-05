# v0.12.0 complete-port plan review

Reviewer role: GPT-5.5 xhigh plan validation, no code execution beyond local file reads, no GPU.

Primary document reviewed: `.agent/decisions/PLAN-V0120-COMPLETE-PORT.md`.

Local references used: `PROJECT_CONSTITUTION.md`, `AGENTS.md`, project-local review/physics validation skills, `publish/GPU_PORT_GAPS_TODO.md`, `README.md`, `docs/namelist-compatibility.md`, `src/gpuwrf/io/wrf_scheme_catalog.py`, `src/gpuwrf/contracts/physics_registry.py`, `proofs/v0110/wrfout_completeness.md`, and local pristine WRF v4 `run/README.namelist` / `Registry` excerpts.

## 1. Coverage verdict

**GAPS.**

The plan is a good direction-setting document, and Phase 0 correctly calls for a definitive Registry-based audit. But as written it does **not** yet fully cover its stated goal: a complete, faithful WRF v4 ARW port with no credible "not a valid port" flank. Several v0.12.0 phases use broad placeholders such as "full namelist", "full physics matrix", "auxhist/auxinput", "moving/vortex nests if claimed", and "implement or justified-scope" without enumerating the WRF v4 options, package state, proof gates, or explicit exclusions.

The core problem is scope mismatch:

- The stated goal says every remaining WRF v4 feature/option must be implemented or explicitly scoped out.
- The concrete phase list names only a common operational subset plus a few known debts.
- A WRF v4 developer could credibly flag missing physics scheme IDs, shallow/stochastic physics, surface/ocean/lake/urban options, full dynamics controls, ndown/offline nesting, WRFDA/WRFPLUS boundary, full aux streams, full Registry-state restart, and lat-lon/global/polar support.

Phase 0 can repair this only if it becomes a committed machine-readable coverage ledger: every WRF v4 Registry package, namelist rconfig, I/O stream, and scheme ID must end with one of `implemented`, `validated`, `recognized_fail_closed`, or `explicitly_out_of_scope_with_reason`.

## 2. Specific missing or under-scoped WRF v4 features

### Physics scheme matrix

**Microphysics is incomplete.** Phase 1 lists a common subset but misses many recognized WRF v4 `mp_physics` options from the local WRF v4 catalog:

- Missing or only ambiguously covered: `5` Ferrier new Eta/HRW, `9` Milbrandt-Yau 2-moment, `11` CAM 5.1 microphysics, `13` SBU-YLin, `14` WDM5, `17/19/21/22` legacy NSSL variants, `18` NSSL 2-moment 4-ice with predicted CCN, `24` WSM7, `26` WDM7, `27` UDM 7-class, `29` RCON Thompson-aerosol variant, `30/32` HUJI spectral-bin fast/full, `38` Thompson 2-moment graupel/hail, `40` Morrison aerosol, `50/51/52/53` P3 family, `55` Jensen-ISHMAEL, `56` NTU multi-moment, `95` Ferrier old Eta, `96` MAD-WRF, `97` Goddard GCE.
- "NSSL" is too vague. It needs exact coverage for `17/18/19/21/22` or an explicit legacy/new-NSSL split.
- Required scheme controls are not scoped: `mp_zero_out`, `mp_zero_out_thresh`, `mp_zero_out_all`, `hail_opt`, `morr_rimed_ice`, `progn`, `ccn_conc`, `no_mp_heating`, `use_mp_re`, Thompson table controls, aerosol IC/BC controls for `28/29/40`, Goddard GCE hail/ice controls, radar reflectivity, HAILCAST, and MAD-WRF controls.

**Cumulus is incomplete.** Phase 1 lists `cu_physics` `1/2/3/5/6` and "SAS family(4/14/84)"; WRF v4 has:

- Missing: `7` Zhang-McFarlane CAM5, `10` modified KF/PDF trigger, `11` multi-scale KF, `16` New Tiedtke, `93` Grell-Devenyi, `94/95/96` HWRF/GFS SAS variants, `99` previous KF.
- `84` appears to be a wrong code; the WRF v4 catalog has `94`, not `84`.
- Shallow convection is a separate flank: `shcu_physics=1..5`, `ishallow`, Deng shallow cu, Park-Bretherton, GRIMS, NSAS shallow cu, and related aerosol/radiation controls are not mentioned.
- Cumulus diagnostic/radiation controls are absent: `cu_diag`, `cu_rad_feedback`, `bmj_rad_feedback`, `kfeta_trigger`, `kf_edrates`, `cugd_avedx`, `nsas_dx_factor`, `shallowcu_forced_ra`, `cudt`, convective transport averaging, and MSKF aerosol controls.

**PBL is incomplete.** Phase 1 covers `1/2/5/7/8/11`, but omits:

- `3` Hybrid EDMF GFS/ACM-GFS, `4` QNSE-EDMF, `9` UW/CAM5, `10` TEMF, `12` Grenier-Bretherton-McCaa, `16` TKE + epsilon, `17` TKE + epsilon + TPE, `99` MRF.
- MYNN option fidelity is under-scoped: `bl_mynn_closure`, `bl_mynn_tkeadvect`, `icloud_bl`, `bl_mynn_cloudmix`, `bl_mynn_mixlength`, `bl_mynn_cloudpdf`, `bl_mynn_edmf`, `bl_mynn_edmf_mom`, `bl_mynn_edmf_tke`, `bl_mynn_mixscalars`, `bl_mynn_mixqt`, `scalar_pblmix`, `tracer_pblmix`, `tke_budget`, and `grav_settling`.

**Surface layer is incomplete and one code is ambiguous.**

- WRF v4 `sf_sfclay_physics=1` is revised MM5; `91` is old MM5. The plan says "MM5(1)" and "revised-MM5(91)", which should be corrected.
- Missing: `3` NCEP GFS surface layer, `4` QNSE, `7` Pleim-Xiu, `10` TEMF.
- Surface-layer controls such as `isfflx`, `iz0tlnd`, `isftcflx`, topographic wind correction, shallow-water roughness coupling, and scheme/PBL mandatory pairings need explicit validation or rejection.

**Land/surface physics is incomplete.**

- Missing `sf_surface_physics`: `1` thermal diffusion/slab, `5` CLM4, `6` CTSM, `7` Pleim-Xiu LSM, `8` SSiB. The plan lists only Noah, RUC, Noah-MP.
- Urban scope is incomplete: the plan excludes BEP/BEM but not clearly `sf_urban_physics=1` single-layer UCM. If all urban is out, state `sf_urban_physics=1/2/3` fail-closed with reasons.
- Lake and ocean are not consistently treated. The plan says lake out-of-scope, but should also cover `sf_lake_physics`, `sf_ocean_physics=1/2`, SST update, fractional sea ice, sea-ice albedo/snow/thickness options, shallow-water roughness, and ocean mixed-layer/PWP state.
- Noah-MP completeness requires the full `&noah_mp` option matrix or explicit unsupported values: `dveg`, `opt_crs`, `opt_sfc`, `opt_btr`, `opt_run`, `opt_frz`, `opt_inf`, `opt_rad`, `opt_alb`, `opt_snf`, `opt_tbot`, `opt_stc`, `opt_gla`, `opt_rsf`, `opt_soil`, `opt_crop`, `opt_irr`, `opt_irrm`, `opt_tdrn`, `soiltstep`, `noahmp_output`, `noahmp_acc_dt`.
- Irrigation/mosaic/PXLSM controls are unmentioned: `sf_surf_irr_scheme`, `irr_*`, `sf_surface_mosaic`, `mosaic_cat`, `mosaic_lu`, `mosaic_soil`, `pxlsm_*`.

**Radiation is incomplete.**

- Longwave missing: `ra_lw_physics=7` FLG/UCLA, `14` RRTMG-K/KIAPS, `24` fast RRTMG, `31` Held-Suarez forcing, `99` GFDL/Eta semi-supported.
- Shortwave missing: `ra_sw_physics=2` Goddard old, `7` FLG/UCLA, `14` RRTMG-K/KIAPS, `24` fast RRTMG, `99` GFDL/Eta. The plan's "Goddard(5)" is incomplete because SW has both `2` and `5`, while LW uses `5`.
- Radiation option controls are not scoped: `radt`, `cldovrlp`, `idcor`, `ra_sw_eclipse`, `ghg_input`, `swint_opt`, `couple_farms`, CAM ozone/aerosol dimensions, `o3input`, `aer_opt`, AOD/Angstrom/SSA/asymmetry controls, `icloud`, `insert_init_cloud`, `slope_rad`, `topo_shading`, `shadlen`, `use_mp_re`, and cloud fraction/radiative effective radius interactions.

### Dynamics and numerics

Phase 2 names many important dynamics items, but it is not yet concrete enough for a no-flanks port.

Missing or under-scoped:

- Advection controls beyond "orders 2-6": `h_mom_adv_order`, `v_mom_adv_order`, `h_sca_adv_order`, `v_sca_adv_order`, `momentum_adv_opt=1/3`, `moist_adv_opt`, `scalar_adv_opt`, `chem_adv_opt`, `tracer_adv_opt`, `tke_adv_opt`, `moist_adv_dfi_opt`, and `phi_adv_z=1/2`.
- Positive-definite/monotonic transport must be WRF-equivalent, not an ad hoc clamp. Include `pos_def`, WRF scalar/moisture limiters, `mp_zero_out`, and per-field limiter engagement budgets.
- Hybrid-coordinate and vertical numerics: `hybrid_opt=0/2`, `etac`, `zadvect_implicit`, `w_crit_cfl`, `time_step_sound`, `use_theta_m`, `use_q_diabatic`, `non_hydrostatic`, `top_lid`, and base-state parameters.
- Diffusion detail: `diff_opt=0/1/2`, `km_opt=1/2/3/4/5`, `diff_6th_opt=0/1/2`, `diff_6th_factor`, `diff_6th_slopeopt`, `diff_6th_thresh`, `mix_full_fields`, `mix_isotropic`, `mix_upper_bound`, `c_s`, `c_k`, `khdif`, `kvdif`, `tke_drag_coefficient`, `tke_heat_flux`, and `sfs_opt` nonlinear backscatter/aniso options.
- Damping/filtering detail: `damp_opt=0/1/2/3`, `dampcoef`, `zdamp`, `w_damping`, `smdiv`, `emdiv`, `epssm`, polar/global filters (`fft_filter_lat`, `coupled_filtering`), and polar/grid-average options.
- Gravity-wave drag cannot remain "implement if completeness demands"; WRF v4 has `gwd_opt=1` and `gwd_opt=3` plus `gwd_diags`. Implement or explicitly reject both.
- Adaptive timestep needs the full control group: `use_adaptive_time_step`, `step_to_output_time`, `target_cfl`, `target_hcfl`, `max_step_increase_pct`, `starting_time_step`, `max_time_step`, `min_time_step`, `adaptation_domain`, and restart state.
- DFI must be exact or rejected: `dfi_opt=1/2/3`, filter families `dfi_nfilter=0..8`, DFI stop times, DFI hydrometeor handling, and DFI-specific moist/scalar allocations.

### Boundaries and LBCs

The plan mentions LBC/specified boundaries but under-scopes WRF's boundary-control matrix.

Required coverage or explicit rejection:

- `specified`, `nested`, `spec_bdy_width`, `spec_zone`, `relax_zone`, `spec_exp`, `constant_bc`, `multi_bdy_files`.
- Open, periodic, symmetric, and polar boundary modes: `periodic_x/y`, `open_xs/xe/ys/ye`, `symmetric_*`, `polar`.
- `have_bcs_moist` and `have_bcs_scalar` for ndown/offline runs.
- WRF boundary tendencies for all relevant Registry state: dynamics, moisture, scalar number concentrations, chemistry/tracers if recognized, aerosols, TKE/QKE, DFI fields, GWD fields, and stochastic fields where active.
- WRF boundary-order degradation for high-order advection near specified/nested strips, not only interior stencils.

### Nesting

Phase 4 is materially under-scoped.

- One-way/two-way needs concrete acceptance for `max_dom`, `grid_id`, `parent_id`, `i_parent_start`, `j_parent_start`, `parent_grid_ratio`, `parent_time_step_ratio`, `feedback`, and `smooth_option`.
- Multiple simultaneous child domains and child subcycling must be tested, not only one child.
- Feedback must cover all feedback-eligible WRF fields, accumulators, hydrometeors/scalars, and physics carry. It also needs parent smoothing/desmoothing parity.
- `interp_method_type` for coarse-to-fine interpolation and nested LBC construction is missing.
- `ndown`/offline nesting is not named in the plan. It needs coverage for `have_bcs_moist`, `have_bcs_scalar`, vertical refinement (`vert_refine_fact`, `vert_refine_method`, `rebalance`), and wrfbdy generation/consumption.
- Moving nests are not optional under this goal. "If claimed" is not enough; either implement specified moving nests (`num_moves`, `move_id`, `move_interval`, `move_cd_x/y`) and vortex-following nests (`vortex_interval`, `max_vortex_speed`, `corral_dist`, `track_level`, `time_to_move`) or explicitly scope them out.
- `input_from_hires`, moving-nest terrain/landuse handling, and moving-nest restart state are unmentioned.

### Data assimilation and nudging

Phase 4 says FDDA "implement or justified-scope", but it should be concrete:

- Grid analysis nudging: `grid_fdda=1`, `gfdda_inname`, `gfdda_interval_m`, `gfdda_end_h`, `fgdt`, `guv`, `gt`, `gq`, PBL and vertical-factor controls.
- Spectral nudging: `grid_fdda=2`, `fgdtzero`, `xwavenum`, `ywavenum`, `gph`, `ktrop`, `if_zfac_*`, `dk_zfac_*`, FFT state.
- Surface FDDA/FASDAS: `grid_sfdda=1/2`, `sgfdda_*`, `guv_sfc`, `gt_sfc`, `gq_sfc`, `rinblw`, `pxlsm_soil_nudge`.
- Observation nudging: `obs_nudge_opt`, `auxinput11_interval`, `auxinput11_end_h`, `max_obs`, `fdda_start/end`, `obs_nudge_wind/temp/mois/pstr`, coefficients, radii, time window, PBL restrictions, surface spreading, and negative-QV innovation guard.
- WRFDA/WRFPLUS/4DVAR/TL/AD are not in the plan. If "DA" means only forecast-model FDDA, explicitly scope WRFDA/WRFPLUS out.

### I/O, wrfout, wrfrst, and streams

The I/O phase is directionally right but not complete enough.

- The 375-variable reference is one CPU-WRF file, not the whole WRF Registry I/O universe. The plan needs a scheme-dependent Registry I/O matrix, including dimensions, staggering, units, metadata, accumulators, optional package fields, diagnostics, and inactive-field behavior.
- `wrfrst` full restart must include domain-tree state, physics carry for every supported scheme, radiation carry, land/ocean/lake/urban state, cumulus accumulators, adaptive timestep state, DFI state, moving-nest state, FDDA state, stochastic seeds/patterns, alarms/timers, and all optional Registry packages selected by namelist.
- Restart acceptance should distinguish GPU self-restart continuity from CPU-WRF interoperability. If CPU-WRF `wrfrst` read/write interop is not required, say so.
- Aux streams need all 1-24 history/input controls, not just "auxhist/auxinput": `auxhistN_*`, `auxinputN_*`, intervals, begin/end windows, frames, names, and `io_form_aux*`. Specific streams matter for SST (`auxinput4`), obs nudging (`auxinput11`), aerosols (`auxinput15/17`), and FDDA.
- WRF I/O controls missing: `history_interval`, `frames_per_outfile`, `restart_interval`, `write_hist_at_0h_rst`, `write_restart_at_0h`, `output_ready_flag`, `nocolons`, `inputout_*`, `write_input`, `iofields_filename`, `ignore_iofields_warning`, and time-series output (`tslist`).
- `io_form_*` support must be explicit. WRF Registry includes intio, netCDF, HDF/PHD5, GRIB1/2, pnetCDF, PIO, netCDF-parallel, ADIOS2; README.namelist highlights at least netCDF, PHD5, GRIB1, GRIB2, pnetCDF. Implement only NetCDF if that is the product boundary, but reject the others loudly.

### Namelist and Registry coverage

"Full WRF namelist parse + validate" needs to be a first-class deliverable, not a single checker.

Required acceptance:

- Parse all standard groups relevant to ARW forecast use: `time_control`, `domains`, `physics`, `dynamics`, `bdy_control`, `dfi_control`, `fdda`, `noah_mp`, `stoch`, and recognized optional groups.
- Handle Fortran namelist syntax, repeat counts, per-domain arrays, defaults, `max_dom` broadcasting, and WRF suite overrides (`-1` values where used).
- Validate cross-option dependencies: PBL/surface-layer pairings, LSM/num_soil_layers, MYNN options, radiation/cloud/aerosol dependencies, ndown boundary scalar/moist flags, moving-nest compile gates, urban/LSM compatibility, diffusion/PBL compatibility, DFI allocations, and I/O stream requirements.
- Emit a machine-readable ledger of every recognized unsupported option, not only selected option failures.

### Map projections, global/polar, and static geography

The plan says "map projections beyond Lambert/Mercator/Polar as needed", but the no-flanks goal requires a concrete projection boundary.

- WRF Registry uses `map_proj=0` cylindrical/latitude-longitude, `1` Lambert, `2` polar stereographic, `3` Mercator. `map_proj=0`/lat-lon is missing.
- Rotated-pole/global support, `rotated_pole`, polar boundary, polar filters, map factors, Coriolis/curvature, `actual_distance_average`, and `swap_pole_with_next_j` need implementation or loud rejection.
- WPS/real static-geography options are under-scoped: landuse/soil categories, `mminlu`, `iswater/islake/isice/isurban`, geog checksums, `surface_input_source`, `num_land_cat`, `num_soil_cat`, `use_wudapt_lcz`, bathymetry, lake depth, sea ice, SST, LAI/albedo, and monthly fields.

### Native init, WPS/real, and vertical interpolation

The v0.12.0 plan assumes v0.4 native init exists, but a complete WRF v4 port also needs explicit coverage for `real.exe`-equivalent choices:

- WPS/metgrid input mapping, `input_from_file`, `fine_input_stream`, `all_ic_times`, `interval_seconds`, and external LBC cadence.
- Vertical interpolation/extrapolation options: `interp_type`, `extrap_type`, `t_extrap_type`, `use_levels_below_ground`, `use_surface`, `lagrange_order`, `sfcp_to_sfcp`, `force_sfc_in_vinterp`, `use_sh_qv`, `interp_theta`, `hypsometric_opt`, custom `eta_levels`, `auto_levels_opt`, `dzbot`, `max_dz`, and p-top settings.
- Aerosol/WIF input and time-varying aerosol streams for Thompson-aerosol/RCON are unscoped.

### Other ARW subsystems a reviewer could flag

These do not necessarily need implementation, but they need named scope decisions:

- Stochastic physics: `rand_perturb`, `sppt`, `skebs`, `spp`, `spp_conv`, `spp_pbl`, `spp_lsm`, `multi_perturb`, perturbation seeds/patterns, and stochastic boundary perturbations.
- Lightning, HAILCAST, radar reflectivity diagnostics, wind farms, trajectories, SCM forcing, WRF-Solar/FARMS, and WRF-Fire.
- Chemistry, WRF-Chem, WRF-Hydro, and BEP/BEM are mentioned partly, but the exclusion list should be complete and tied to fail-closed namelist behavior.

## 3. Risks and sequencing issues

- **Phase 0 is doing too much implicit work.** If Phase 0 is the binding scope, it needs to produce the canonical coverage ledger before Phase 1 begins. Otherwise implementers can close Phase 1 against the short list while leaving unlisted WRF features open.
- **The physics matrix is too broad for one milestone unless split into common-supported vs full-recognized.** Exact WRF parity for every MP/PBL/CU/RA/LSM option is a very large project. A credible v0.12.0 plan should separate "implemented common matrix" from "recognized and deliberately unsupported WRF catalog" rather than implying all schemes will be ported.
- **Acceptance criteria are not consistently falsifiable.** "Implemented + oracle-validated" should specify savepoint boundary, field list, tolerances, cases, precision, scheme controls, scan wiring, integrated-run gates, and fail-closed behavior for unsupported dependencies.
- **I/O/restart can block everything late.** Full `wrfrst` for all physics/dynamics/DA/nesting/stochastic combinations should be designed before broad scheme implementation, or the state ABI will churn repeatedly.
- **Nesting and boundary semantics should precede Gotthard if the demo uses a nested or specified-boundary real case.** Otherwise the suite may validate only a single-domain subset while the release claims complete WRF v4 ARW.
- **Completeness vs project constitution tension.** The constitution says this is not a line-by-line WRF port and prioritizes useful WRF compatibility. The v0.12.0 plan goal says complete WRF v4 with no open flanks. The plan needs an explicit product-scope decision resolving that tension.

## 4. Gotthard / Central-Switzerland test soundness and framing

The Gotthard suite is a good idea: non-Canary, steep Alpine terrain, 1 km, 24 h, CPU-WRF v4 reference, GPU port comparison, all-grid-point/per-output-time statistics, default published CPU reference, and explicit non-bitwise framing.

The honest equivalence framing is correct:

- It should claim **numerical/operational equivalence within predeclared tolerance**, not bitwise equality to Fortran.
- "Equal at all points/times" must mean every compared value at every sampled output time satisfies the declared tolerance rule.
- GPU self-determinism can be bitwise, but that is a separate claim from CPU-WRF equivalence.

Soundness gaps to fix before shipping:

- **Fix the domain, do not auto-size the default.** "Largest that fits with headroom in 32 GB" makes the default test hardware-dependent and incompatible with a single published CPU reference. Predeclare exact `e_we/e_sn/e_vert`, `dx/dy`, `time_step`, `p_top`, vertical levels, projection, output interval, and physics suite. Optional larger demos can be separate.
- **Pin the CPU-WRF reference.** Record WRF version, compiler, precision, namelist, WPS/geog versions, AIFS source/date/lead, `wrfinput`/`wrfbdy` checksums, and wrfout checksums. Google Drive is acceptable only with checksums and a non-link-rot fallback plan.
- **Define the field inventory.** The Phase 6 text lists T2/U10/V10/precip and core 3D fields with an ellipsis. For a v0.12.0 complete-port gate, compare every common `wrfout` variable in the declared reference inventory, including staggered fields, hydrometeors, number concentrations, radiation, PBL, surface/land, accumulators, and diagnostics. If the demo intentionally compares a smaller public set, call it a demo, not complete-port proof.
- **Predeclare tolerances before the local confirmation run.** Tolerances should be per variable and account for units, staggering, masks, hydrometeor sparsity, accumulators, and absolute/relative floors. Do not tune after seeing the default day.
- **Clarify "all timesteps".** Ordinary `wrfout` compares output times, not internal RK/acoustic timesteps. If the claim is all internal timesteps, the suite needs savepoints. If the claim is all output times, say that.
- **Pointwise 24 h at 1 km may be brittle.** A max-abs/everywhere gate can be valid if the port is genuinely near-identical, but Alpine convection and steep terrain can amplify tiny numerical differences. Keep max-abs reporting, but define the pass rule carefully and include RMSE/bias/quantiles plus inner-domain masks separate from full-domain LBC strips.
- **A single 24 h date is a regression/demo, not broad generalization proof.** It demonstrates one non-Canary Alpine case. For "generalizes" language, add more dates/regimes or phrase the suite as an external-region equivalence smoke/regression.
- **AIFS licensing and redistribution need an explicit note.** The default CPU solution can be redistributed only if the data/license path permits it; otherwise publish manifests and require user-side generation.

Bottom line: the Gotthard suite is sound as a shipped, self-serve equivalence demo if pinned and tolerance-predeclared. It is not by itself proof of a complete WRF v4 port, and the plan should avoid saying it "proves generalization" from one default day.

