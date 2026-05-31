# GPU Port Gaps TODO

Executive summary: the current code is a real single-domain GPU forecast path for Canary d02 replay, not yet a complete WRF v4 operational replacement. It runs a d02 state built from Gen2/CPU-WRF artifacts with RK3 split-explicit dynamics, Thompson, WRF revised surface layer, MYNN, RRTMG-style SW/LW radiation, and WRF-style lateral strip replay. The blocking gaps are live multi-domain nesting, removal of WPS/real.exe/CPU-WRF artifact dependency, prognostic Noah-MP land state, WRF-compatible restart/output completeness, d01 parent-domain physics, and closure of real-terrain/map-factor/boundary dynamics under the Canary hierarchy. The Canary corpus namelist is a five-domain 9/3/1 km nest (`max_dom=5`) with Thompson, MYNN, revised surface layer, Noah-MP, RRTMG, Kain-Fritsch only on d01, topographic/slope radiation enabled, and WRF specified/nested boundaries; the port currently owns only the d02 replay path.

## Audit Context

- Repository inspected at `5319b8d` on local branch `manager-2026-05-23`; the requested branch name was `worker/opus/final-verdict`, but the requested HEAD matched.
- Corpus namelist found: `../canairy_meteo/Gen2/surface_geo_v2_1/physics_sweep/_baseline/namelist.input`. No `wrf_l2` or `wrf_l3` namelist files were present under this workspace search.
- Canary namelist anchors: `max_dom=5`, 9/3/1 km grid hierarchy, and parent ratios in `namelist.input:31-50`; physics suite in `namelist.input:56-74`; dynamics and advection options in `namelist.input:77-93`; specified/nested boundary flags in `namelist.input:96-102`.
- Radiation actually present: RRTMG-style SW and LW column kernels (`src/gpuwrf/physics/rrtmg_sw.py:1`, `src/gpuwrf/physics/rrtmg_lw.py:1`) called through a WRF-style held `RTHRATEN` cadence (`src/gpuwrf/runtime/operational_mode.py:1563-1598`). This is not a configurable WRF radiation suite.
- Land surface actually present: a prescribed Noah-MP state subset, explicitly not prognostic Noah-MP (`src/gpuwrf/physics/noah_mp.py:1-6`), plus WRF revised surface layer / sfclayrev (`src/gpuwrf/physics/surface_layer.py:1-22`). Daily mode refreshes land fields hourly from the Gen2 corpus (`src/gpuwrf/integration/daily_pipeline.py:292-315`, `src/gpuwrf/integration/daily_pipeline.py:357-359`).

## P0 - Blocks True Port / Complete Replacement

### P0-1: Live multi-domain nesting and grid hierarchy

- WRF v4 has: `max_dom` domain trees, parent/child placement, parent and child timestep ratios, one-way nested boundary interpolation, optional two-way feedback, and multiple simultaneous nests. The Canary namelist uses five domains with d01 at 9 km, d02 at 3 km, and three 1 km island nests (`namelist.input:35-50`). It sets `feedback=0`, so two-way feedback is not needed for the current baseline, but live one-way parent-to-child nesting is still required (`namelist.input:50`, `namelist.input:100-101`).
- Port has/lacks: `GridSpec` represents one grid, not a hierarchy (`src/gpuwrf/contracts/grid.py:341-350`). Daily mode takes one `domain` string, defaulting to `d02` (`src/gpuwrf/integration/daily_pipeline.py:64-79`). `build_replay_case` loads one `Gen2Run` domain and returns one `ReplayCase` (`src/gpuwrf/integration/d02_replay.py:734-855`). There is a helper that can build child-sized boundary leaves by interpolating recorded parent-domain history (`src/gpuwrf/integration/d02_replay.py:541-731`), but that is offline replay, not a live parent/child scheduler. `apply_lateral_boundaries` applies strips to one `State` only (`src/gpuwrf/coupling/boundary_apply.py:80-119`).
- Canary impact: this blocks replacement of the nightly CPU-WRF job because the operational product includes the 1 km nests. Even with `feedback=0`, d02/d03/d04/d05 need live parent states for nested boundaries. The current path can replay a d02-like domain but cannot produce the whole live hierarchy.
- Effort: XL.
- Priority: P0.
- TODO: implement a domain-tree runtime with parent and child state carries, parent-to-child interpolation, child timestep subcycling, per-domain physics/radiation cadence, multi-domain output, and optional feedback kept behind a separate gate.

### P0-2: Native initialization, WPS/real.exe replacement, and external lateral forcing

- WRF v4 has: WPS plus `real.exe` to create `wrfinput_*` and `wrfbdy_*` from external analyses/forecasts, including horizontal/vertical interpolation, base-state construction, soil/land fields, static geography, and boundary tendencies. The corpus uses `input_from_file=.true.` and 6-hour forcing intervals (`namelist.input:18-29`).
- Port has/lacks: daily mode builds the real case from `gpuwrf.integration.d02_replay.build_replay_case` (`src/gpuwrf/integration/daily_pipeline.py:156-243`). `build_replay_case` loads the initial state directly from Gen2 WRF history variables (`U`, `V`, `W`, `T`, `QVAPOR`, `P`, `PB`, `PH`, `PHB`, `MU`, `MUB`) and uses a WRF history file as the metric source (`src/gpuwrf/integration/d02_replay.py:742-820`). Boundary leaves are packed from hourly `wrfout` side history (`src/gpuwrf/integration/d02_replay.py:300-430`) or from recorded parent-domain history (`src/gpuwrf/integration/d02_replay.py:541-731`). A `wrfbdy` decoder exists (`src/gpuwrf/io/boundary_replay.py:181-232`), but the operational path still consumes pre-existing WRF/Gen2 artifacts instead of generating native external forcing.
- Canary impact: the port cannot run a nightly forecast from AIFS/GFS/ERA5/metgrid inputs by itself. It still needs CPU-WRF-produced or WRF-compatible corpus files for initial conditions, boundary forcing, metrics, and land fields, so it is not a complete replacement.
- Effort: XL.
- Priority: P0.
- TODO: own the operational ingest path: external model data -> grid/static fields -> vertical coordinate/base-state -> `State` -> time-varying lateral forcing, with parity tests against `real.exe` and WRF `wrfbdy`.

### P0-3: Prognostic Noah-MP land-surface model

- WRF v4 has: Noah-MP for `sf_surface_physics=4`, including prognostic soil temperature, soil moisture/liquid, skin temperature, snow/canopy/hydrology and surface energy/water budget coupling. The Canary namelist uses Noah-MP on every domain (`namelist.input:61`).
- Port has/lacks: `noah_mp.py` is explicitly a "Prescribed Noah-MP subset" and "does not implement prognostic Noah-MP" (`src/gpuwrf/physics/noah_mp.py:1-6`). `State` carries only a thin surface subset such as `t_skin`, top `soil_moisture`, `xland`, `lakemask`, `mavail`, `roughness_m`, and `lu_index`, not the full Noah-MP prognostic state (`src/gpuwrf/contracts/state.py:414-435`). The daily pipeline refreshes `TSK`, `SST`, `SMOIS`, `SH2O`, and `TSLB` from Gen2 files at hourly output boundaries (`src/gpuwrf/integration/daily_pipeline.py:292-315`, `src/gpuwrf/integration/daily_pipeline.py:357-359`). The surface layer consumes prescribed surface fields (`src/gpuwrf/physics/surface_layer.py:302-336`).
- Canary impact: this is a hard standalone-forecast blocker. Island T2, fluxes, sea-breeze/downslope winds, cloud base, and nighttime cooling depend on land-surface memory and terrain/coast coupling. The current hourly land refresh also depends on CPU-WRF/Gen2 output.
- Effort: XL.
- Priority: P0.
- TODO: implement prognostic Noah-MP or a WRF-faithful operational subset with all state variables needed for the Canary namelist, then remove hourly corpus land refresh from the production path.

### P0-4: d01 parent-domain cumulus physics

- WRF v4 has: cumulus parameterizations, including Kain-Fritsch for `cu_physics=1`. The Canary namelist uses Kain-Fritsch on d01 and disables cumulus on d02-d05 (`namelist.input:64-65`).
- Port has/lacks: runtime physics is hardwired to Thompson -> surface layer -> MYNN -> RRTMG (`src/gpuwrf/runtime/operational_mode.py:1551-1598`). `OperationalNamelist` has no WRF `cu_physics` selector or alternate scheme selectors (`src/gpuwrf/runtime/operational_mode.py:83-136`). The older coupled-core constant explicitly records `cu_physics=0` and `sf_surface_physics=0` for the coupled path (`src/gpuwrf/dynamics/core/coupled.py:50-58`).
- Canary impact: this does not directly affect the current single d02 replay because d02 has `cu_physics=0`. It blocks a real live d01 parent forecast. Without d01 cumulus, parent heating/precipitation and therefore nested d02 boundary forcing will diverge from the nightly CPU-WRF configuration.
- Effort: L.
- Priority: P0.
- TODO: implement Kain-Fritsch or change the operational parent-domain physics contract with explicit validation showing the replacement parent forecast is equivalent enough for d02/d03-d05 boundaries.

### P0-5: WRF-compatible wrfout, wrfrst, and operational diagnostics completeness

- WRF v4 has: rich `wrfout` history files, restart files (`wrfrst`) that can resume full model state, and metadata/diagnostics expected by downstream tools. The corpus writes hourly history, NetCDF input/boundary/restart formats, and has restart controls even though `restart=.false.` in this baseline (`namelist.input:20-28`).
- Port has/lacks: the writer is explicitly "Minimal WRF-compatible" (`src/gpuwrf/io/wrfout_writer.py:1`). It writes a minimum variable set (`src/gpuwrf/io/wrfout_writer.py:25-71`) and loops only over `MINIMUM_WRFOUT_VARIABLES` (`src/gpuwrf/io/wrfout_writer.py:347-382`). Several fields are fallback/recomputed rather than full WRF history, including CLDFRA defaults and surface flux fallbacks (`src/gpuwrf/io/wrfout_writer.py:555-621`). Lat/lon falls back to a simple planar approximation if not present (`src/gpuwrf/io/wrfout_writer.py:755-784`). Checkpoints are project pickle files for restart-continuity probes, not WRF `wrfrst` files (`src/gpuwrf/runtime/checkpoint.py:1`, `src/gpuwrf/runtime/checkpoint.py:52-86`). The restart probe compares final GPU wrfouts, not WRF restart interoperability (`src/gpuwrf/integration/daily_pipeline.py:830-867`).
- Canary impact: downstream postprocessing, archive comparison, restart-after-failure, cycling, and operational handoff remain incomplete. A nightly replacement must produce enough WRF-compatible output for existing consumers and must resume after interruptions without rerunning from t0.
- Effort: L to XL, depending on required downstream variable set.
- Priority: P0.
- TODO: define the Canary required wrfout/wrfrst variable contract, write full metadata and diagnostics, add true restart files including runtime carry/radiation/land/physics state, and prove restart continuity over 24-72 h.

### P0-6: Real-terrain, map-factor, and specified/nested-boundary dynamics closure

- WRF v4 has: map-factor-aware C-grid dynamics, terrain-following pressure-gradient and diffusion terms, boundary-aware advection order degradation near specified/nested boundaries, damping/filter options, and terrain/upper-boundary treatments tuned for real orography. The Canary namelist uses steep-island domains with `diff_6th_opt=2`, Rayleigh damping, `w_damping=1`, and `gwd_opt=1` (`namelist.input:77-93`).
- Port has/lacks: metrics can store map factors, slopes, and Coriolis (`src/gpuwrf/contracts/grid.py:56-104`), but the F7 dycore status still lists "3D terrain slope diffusion cross-coordinate terms, map factors, lateral specified/nested boundaries" as Phase-B scope, not closed (`proofs/f7/DYCORE_STATUS.md:68-73`). Flux advection is frozen to h=5/v=3 and the periodic boundary path, with WRF boundary degradation out of scope (`src/gpuwrf/dynamics/flux_advection.py:9-15`). Explicit diffusion states periodic x/y and unit map factors (`src/gpuwrf/dynamics/explicit_diffusion.py:18-20`). `rhs_ph` documents idealized/periodic scope with map factors and higher-order horizontal branches deferred (`src/gpuwrf/dynamics/core/rhs_ph.py:38-45`). The acoustic surface-w boundary condition deliberately feeds decoupled winds as a stability tradeoff and says it is "NOT a WRF match" (`src/gpuwrf/dynamics/core/acoustic.py:618-629`).
- Canary impact: this is directly tied to the current wind gap. The README reports T2 skill but U10/V10 not yet operationally skillful, with V10 beaten by persistence (`README.md:31-36`). Canary forecasts are dominated by volcanic terrain, coastlines, and nested lateral boundaries, so these remaining real-grid details are not cosmetic.
- Effort: XL.
- Priority: P0.
- TODO: close real-grid dycore parity under d02 and 1 km nest fixtures: map factors in all large/small-step operators, boundary-order degradation, terrain-slope diffusion/PGF terms, stable WRF-matching surface-w handling, and wind-skill gates.

### P0-7: Coupled conservation budgets and non-masking safety policy

- WRF v4 has: mass/moist/scalar conservation expectations, WRF positive-definite and monotonic options, physics tendencies integrated at defined RK/physics cadences, and no hidden "fallback to previous value" masking of coupled forecast failures.
- Port has/lacks: runtime guards revert invalid moisture to the origin state (`src/gpuwrf/runtime/operational_mode.py:371-377`), apply a custom theta limiter (`src/gpuwrf/runtime/operational_mode.py:444-512`), and guard post-boundary fields (`src/gpuwrf/runtime/operational_mode.py:1604-1617`). Physics is applied as split state updates after the dycore (`src/gpuwrf/runtime/operational_mode.py:1551-1598`), while the `rk_addtend_dry` call currently receives an empty `DryPhysicsTendencies()` object (`src/gpuwrf/runtime/operational_mode.py:1342-1352`). Thompson falls back to the pre-physics state for thermodynamically invalid columns (`src/gpuwrf/physics/thompson_column.py:1166-1182`). The project has some guards-off dycore proof (`proofs/f7/DYCORE_STATUS.md:47-49`), but full coupled 24-72 h conservation budgets are not a completed WRF replacement proof.
- Canary impact: guards can make a forecast finite while hiding local physical defects in moisture, heat, or boundary coupling. For operational rainfall, island cloud, and wind diagnostics, the port needs explicit dry-mass, water, energy/enthalpy, precipitation, and limiter-engagement budgets.
- Effort: L.
- Priority: P0.
- TODO: add coupled budget diagnostics and acceptance thresholds, report limiter/guard engagement in every operational proof, and remove or formally justify any fallback that changes physical state without a WRF-equivalent mechanism.

## P1 - Important For Fidelity, Robustness, Or WRF Coverage

### P1-1: Data assimilation, FDDA, grid/obs nudging, and spectral nudging

- WRF v4 has: analysis nudging, observation nudging, grid FDDA, spectral nudging, and related controls for cycling and large-scale constraint.
- Port has/lacks: there is no FDDA/nudging/spectral-nudging implementation under `src/gpuwrf`; the only "nudging" path found is lateral boundary relaxation (`src/gpuwrf/coupling/boundary_apply.py:1-44`, `src/gpuwrf/coupling/boundary_apply.py:80-119`). `OperationalNamelist` has no FDDA/nudging controls (`src/gpuwrf/runtime/operational_mode.py:83-136`).
- Canary impact: the inspected baseline namelist does not enable FDDA, so this is not a blocker for matching that exact nightly configuration. It is still required for a broad WRF v4 replacement or any future cycled/nudged Canary setup.
- Effort: M to XL.
- Priority: P1.
- TODO: add a nudging design only if the operational namelist or cycling plan requires it; otherwise document it as unsupported WRF functionality.

### P1-2: Physics scheme option coverage beyond the pinned Canary suite

- WRF v4 has: many choices for microphysics, radiation, PBL, surface layer, land surface, cumulus, urban physics, shallow cumulus, aerosols, and chemistry interactions.
- Port has/lacks: the operational runtime exposes static runtime controls, not WRF scheme selectors (`src/gpuwrf/runtime/operational_mode.py:83-136`). The active physics call order is fixed (`src/gpuwrf/runtime/operational_mode.py:1551-1598`). The implemented operational set is Thompson, revised surface layer, MYNN, and RRTMG-style SW/LW; no alternate WRF physics families are implemented in `src/gpuwrf/physics`.
- Canary impact: low for the exact d02/d03-d05 baseline if the pinned schemes are accepted, except for Noah-MP and d01 cumulus already listed as P0. High for "full WRF v4 replacement" claims or changing the nightly namelist.
- Effort: XL.
- Priority: P1.
- TODO: state the supported physics matrix explicitly and reject unsupported namelist combinations at load time. Add schemes only when a real Canary experiment needs them.

### P1-3: RRTMG radiation fidelity gaps: terrain shading, slope radiation, cloud fraction, gases, and surface properties

- WRF v4 has: RRTMG SW/LW with exact grid lat/lon, terrain/slope radiation options, land-surface albedo/emissivity coupling, cloud fraction handling, trace gas/ozone/aerosol options, and pressure/interface inputs from the model state. The Canary namelist enables `topo_shading=1` and `slope_rad=1` (`namelist.input:68-69`).
- Port has/lacks: RRTMG-style SW/LW kernels exist (`src/gpuwrf/physics/rrtmg_sw.py:1`, `src/gpuwrf/physics/rrtmg_lw.py:1`) and are called at WRF-style held cadence (`src/gpuwrf/runtime/operational_mode.py:1563-1598`). However, lat/lon for radiation is a deterministic approximation from grid center and spacing (`src/gpuwrf/coupling/physics_couplers.py:463-483`), albedo/emissivity come from fixed MODIS lookup tables (`src/gpuwrf/coupling/physics_couplers.py:41-44`, `src/gpuwrf/coupling/physics_couplers.py:486-492`), cloud fraction is a simple hydrometeor occupancy diagnostic (`src/gpuwrf/coupling/physics_couplers.py:584-588`), trace gases are hardcoded constants (`src/gpuwrf/physics/rrtmg_constants.py:16-20`), and pressure interfaces are reconstructed from midpoint pressure (`src/gpuwrf/physics/rrtmg_sw.py:240-257`). I found no topographic shading or slope-radiation implementation in the operational radiation path.
- Canary impact: important for island diurnal heating, orographic cloud, valley/mountain circulations, and solar exposure on steep slopes. This can feed directly into T2, PBL, and wind skill.
- Effort: M to L.
- Priority: P1.
- TODO: use real XLAT/XLONG/terrain/slope/aspect fields, implement WRF topographic shading and slope radiation, couple surface radiative properties to the prognostic land model, and validate SWDOWN/GLW against WRF fixtures.

### P1-4: MYNN PBL completeness

- WRF v4 has: MYNN variants/options with moisture, cloud, mixing-length, optional mass-flux/EDMF behavior, and many namelist-dependent switches.
- Port has/lacks: the kernel is a JAX MYNN2.5 column kernel (`src/gpuwrf/physics/mynn_pbl.py:1`) and consumes real surface-layer fluxes on the operational path (`src/gpuwrf/physics/mynn_pbl.py:167-180`). It documents "WRF option-2 MYNN master length scale with EDMF/cloud terms disabled" (`src/gpuwrf/physics/mynn_pbl.py:264-266`) and "dry level-2.5 path" (`src/gpuwrf/physics/mynn_pbl.py:316-317`). It applies dry U/V/theta/qv implicit solves (`src/gpuwrf/physics/mynn_pbl.py:488-501`) and returns PBLH (`src/gpuwrf/physics/mynn_pbl.py:515-536`).
- Canary impact: marine PBL, trade-wind inversion, island wake, and near-surface wind skill are central to the nightly use case. The README currently identifies wind skill as the main remaining science gap (`README.md:31-36`).
- Effort: L.
- Priority: P1.
- TODO: compare all MYNN namelist/default options used by the CPU-WRF build, implement missing MYNN pieces that affect wind/T2/Q2/PBLH, and validate against WRF column and real-case fixtures.

### P1-5: Thompson microphysics parity debts

- WRF v4 has: full Thompson `mp_physics=8` with dynamic sedimentation substeps, detailed snow/graupel/riming behavior, cloud-water sedimentation terms, and WRF column guards/diagnostics.
- Port has/lacks: the operational adapter advances hydrometeors, number concentrations, latent heating, sedimentation, and precip accumulators (`src/gpuwrf/coupling/physics_couplers.py:591-620`, `src/gpuwrf/coupling/physics_couplers.py:691-710`). The column code still exposes known approximations: fixed-cap sedimentation substeps instead of WRF per-column `nstep` (`src/gpuwrf/physics/thompson_column.py:829-840`), a snow fall-speed approximation for inactive riming boost (`src/gpuwrf/physics/thompson_column.py:980-990`, `src/gpuwrf/physics/thompson_column.py:1015-1018`), neglected cloud-water sedimentation (`src/gpuwrf/physics/thompson_column.py:1087-1095`), and invalid-column fallback (`src/gpuwrf/physics/thompson_column.py:1166-1182`). The historical source/sink-only entry remains for old tests (`src/gpuwrf/physics/thompson_column.py:1190-1206`), while operational coupling uses the full precip entry (`src/gpuwrf/physics/thompson_column.py:1209-1224`).
- Canary impact: convective showers, orographic precipitation, cloud-radiation feedback, and hydrometeor fields can diverge. This is less obviously blocking than nesting/land, but it matters for a "true port" claim.
- Effort: M.
- Priority: P1.
- TODO: quantify each approximation against WRF column savepoints and real-case precip/cloud metrics; replace fixed or approximate pieces where residuals exceed operational tolerances.

### P1-6: Positive-definite/monotonic scalar advection and WRF boundary-order options

- WRF v4 has: configurable scalar/moisture advection orders, positive-definite and monotonic transport options, and boundary-specific degradation of high-order flux stencils.
- Port has/lacks: flux advection is fixed to h=5/v=3 and periodic boundary assumptions, with WRF boundary degradation explicitly out of scope (`src/gpuwrf/dynamics/flux_advection.py:9-15`). The operational path adds flux-form advection only when `use_flux_advection` is set (`src/gpuwrf/runtime/operational_mode.py:1185-1238`). Moisture safety currently uses validity guards (`src/gpuwrf/runtime/operational_mode.py:371-377`) rather than WRF's full family of positive-definite/monotonic scalar transport options. The Canary namelist uses `moist_adv_opt=1` and `scalar_adv_opt=1` (`namelist.input:91-92`), so this is more a general WRF option gap than a direct mismatch to that baseline.
- Canary impact: important near steep gradients, clouds, and lateral boundaries; can affect water conservation and small-scale precip. Not the top blocker if the current namelist keeps simple scalar advection options.
- Effort: M.
- Priority: P1.
- TODO: implement WRF boundary-order degradation and add explicit support/validation for the scalar and moisture advection options accepted by the runtime.

### P1-7: Gravity wave drag and broader dynamics namelist controls

- WRF v4 has: gravity wave drag and a broader set of damping/diffusion controls. The Canary namelist sets `gwd_opt=1` (`namelist.input:93`).
- Port has/lacks: I found no gravity-wave-drag implementation or `gwd_opt` control in `OperationalNamelist` (`src/gpuwrf/runtime/operational_mode.py:83-136`). The port does implement important damping/filter pieces such as `w_damping`, `damp_opt=3`, and `diff_6th_opt=2` in the daily path (`src/gpuwrf/integration/daily_pipeline.py:195-213`) and acoustic config (`src/gpuwrf/dynamics/core/acoustic.py:60-80`), but not the full WRF dynamics option matrix.
- Canary impact: likely most important on the 9 km parent and upper-level momentum budget; may affect nested wind forcing. The exact operational importance needs a CPU-WRF sensitivity check.
- Effort: M.
- Priority: P1.
- TODO: verify whether `gwd_opt=1` is active/load-bearing in the CPU-WRF Canary build and either implement it or document a justified namelist deviation.

### P1-8: Precision policy and performance shortcuts need operational proof gates

- WRF v4 has: a mostly single-precision operational default unless built otherwise, but its numerics are validated as a single model configuration. A replacement must have one declared precision mode and proof artifacts for it.
- Port has/lacks: the precision matrix gates many live fields to fp32 by default (`src/gpuwrf/contracts/precision.py:77-138`), while daily mode currently forces fp64 (`src/gpuwrf/integration/daily_pipeline.py:195-205`; `src/gpuwrf/runtime/operational_mode.py:327-344`). Thompson has an opt-in fp32 internal path but documents no speed win (`src/gpuwrf/physics/thompson_column.py:121-141`). README says fp32 downcast was validated numerically but gives about 0x additional speedup, while the fp64 acoustic core remains the hot path (`README.md:31-36`).
- Canary impact: force-fp64 is safer for fidelity but weakens the speed target; mixed precision could be acceptable only if validated on the same 24-72 h d02/L3 skill gates and conservation budgets.
- Effort: M.
- Priority: P1.
- TODO: make the operational precision mode explicit, rerun all skill/conservation/restart gates under that mode, and reject any per-kernel fp32/fusion optimization without WRF-oracle and budget proof.

## P2 - Nice-To-Have Or General WRF Breadth

### P2-1: Map projection and grid generality beyond the Canary case

- WRF v4 has: broader map-projection/grid handling, moving nests, global/regional variants, multiple vertical/grid configurations, and many geography combinations.
- Port has/lacks: `ProjectionKind` is limited to `lambert`, `mercator`, and `polar` (`src/gpuwrf/contracts/grid.py:13-15`), and `GridSpec` validates only hybrid-eta C-grid (`src/gpuwrf/contracts/grid.py:380-387`). WRF-compatible lat/lon may be approximated in output if not carried in state (`src/gpuwrf/io/wrfout_writer.py:755-784`).
- Canary impact: low if the operational Canary domains stay on the supported Lambert-like WRF grid and real XLAT/XLONG are carried. Higher if the system is marketed as a general WRF v4 replacement.
- Effort: M.
- Priority: P2.
- TODO: restrict the product claim to supported Canary grids, or add projection/grid loaders and tests for each WRF grid family claimed.

### P2-2: Full WRF namelist parsing and rejection of unsupported options

- WRF v4 has: namelist-driven configuration across dynamics, physics, domains, I/O, and boundary behavior.
- Port has/lacks: daily mode hardcodes the operational settings into `OperationalNamelist.from_grid` (`src/gpuwrf/integration/daily_pipeline.py:195-213`). `OperationalNamelist` is a Python dataclass of runtime controls, not a WRF namelist schema (`src/gpuwrf/runtime/operational_mode.py:83-136`).
- Canary impact: medium operational risk: unsupported options can be silently ignored if a future nightly namelist changes. Low if the supported config is frozen and checked.
- Effort: S to M.
- Priority: P2.
- TODO: add a namelist compatibility checker that accepts only the implemented Canary subset and fails loudly on unsupported WRF controls.

### P2-3: Additional wrfout diagnostics and budget products

- WRF v4 has: many optional diagnostic fields and auxhist streams. The port currently emits a minimum subset and M9 surface diagnostics only (`src/gpuwrf/runtime/operational_mode.py:1694-1746`, `src/gpuwrf/io/wrfout_writer.py:25-71`).
- Port has/lacks: surface diagnostics include SWDOWN, GLW, HFX, LH, PBLH, TSK, T2, U10, V10, and PSFC (`src/gpuwrf/runtime/operational_mode.py:1694-1746`). Writer coverage is intentionally limited (`src/gpuwrf/io/wrfout_writer.py:54-71`).
- Canary impact: depends on downstream dashboards, station verification, hydrology/aviation products, and public archive expectations.
- Effort: S to L.
- Priority: P2.
- TODO: inventory current downstream consumers and add only the variables they actually read, with WRF-compatible metadata and units.

## Bottom Line

The code should not be described as a full WRF v4 port or complete nightly CPU-WRF replacement yet. The accurate current claim is narrower: a single-domain d02 GPU forecast/replay path with several WRF-faithful core pieces and an improving operational physics stack. The shortest path to a truthful replacement is not adding random WRF option breadth; it is closing the P0 chain in this order: live multi-domain hierarchy, native initialization/boundaries, prognostic Noah-MP, full operational I/O/restart, d01 cumulus, real-terrain/boundary dynamics, and coupled conservation proof.
