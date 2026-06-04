# GPT v0.10.0 Phase 1 Optimization Analysis: wrf_gpu2 v0.9.0 Kernel

Scope: shipped v0.9.0 source at `/home/enric/src/wrf_gpu2/.claude/worktrees/gpt-v090-final`, commit `016d993c1b4c28926b8014fe569db71aa6ee712e`, tag `v0.9.0`.

Mode: analysis and planning only. I made no repository code changes and ran no GPU workloads. All timings below use existing artifacts unless explicitly marked `estimate`.

Important local-rule note: I used the project-local repo instructions and did not use the old global `wrf-gpu-port` skill, per `AGENTS.md`.

## Executive Conclusion

The v0.9.0 operational forecast is not arithmetic-throughput-bound. It is dominated by:

- launch/serialization and memory traffic in the dycore/acoustic loop: ~7,236 kernels + ~3,922 memory ops per step, including ~6,890 tiny `loop_*` elementwise fusions per step, with 43-68% GPU idle in the existing nsys analysis (`proofs/perf/compute_cycle_analysis.md`);
- Thompson microphysics: ~20-21 ms isolated, roughly half of a ~43.6-46.3 ms coupled step (`proofs/perf/phase_breakdown.json`, `proofs/perf/warmed_timing.json`, `proofs/perf/fusion_confirm_results.md`);
- wrapper/output/diagnostic host materialization outside the timestep loop, especially daily-pipeline full-state finite checks and per-field wrfout D2H pulls (`src/gpuwrf/integration/daily_pipeline.py`, `src/gpuwrf/io/wrfout_writer.py`; output CPU/D2H share partly estimated from `proofs/perf/v020_wallclock_wins.md`).

The top actionable v0.10 levers are therefore structural: reduce scan/loop launch count, eliminate needless D2H materialization in the daily wrapper, reduce Thompson masked sedimentation work if a no-clipping histogram proves it safe, and fuse or hoist the repeated stage/substep setup work. Precision should be delayed until those changes make the kernel more bandwidth/compute exposed; current fp32 tests are ~1.00x.

## Evidence Base

Primary performance artifacts read:

- `proofs/perf/compute_cycle_analysis.md`: dycore roofline, launch-count and nsys summary, fp32 explanation, acoustic unroll evidence.
- `proofs/perf/roofline_costonly.json`: RTX 5090 peaks and dycore cost analysis.
- `proofs/perf/phase_breakdown.json`: isolated per-phase timings, bytes, FLOPs, call counts.
- `proofs/perf/warmed_timing.json`: warmed coupled timing and cold compile timing.
- `proofs/perf/segscan_24h.json`: segmented 24 h warmed timing and memory.
- `proofs/perf/fusion_confirm_results.md`: coupled command-buffer flag regression.
- `proofs/perf/fusion_results.md`: dynamics-only command-buffer win and acoustic-unroll prior result.
- `publish/runtime_optimization_analysis.md`: synthesized current runtime analysis and refuted levers.
- `publish/GPU_PORT_GAPS_TODO.md`: broader port gaps, used only as context.
- `proofs/perf/v020_wallclock_wins.md`: compile cache, async output, `time_utc` cache-key and continuous-carry records.
- `proofs/thompson_perf/kernel_lever_summary.json`: Thompson lever timings and verdicts.
- `proofs/thompson_perf/THOMPSON_PERF_ANALYSIS.md`: Thompson decomposition.
- `proofs/thompson_perf/PRECIP_ORACLE_AND_IMPLICIT_SED.md`: implicit-sedimentation fidelity rejection.

Source files inspected:

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/runtime/operational_state.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/coupling/boundary_apply.py`
- `src/gpuwrf/physics/thompson_column.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- `src/gpuwrf/io/wrfout_writer.py`

## Where Time Goes

Current warmed numerator:

| Measurement | Value | Source |
|---|---:|---|
| Coupled warmed step | 43.556 ms/step | `proofs/perf/warmed_timing.json` |
| Coupled warmed forecast hour | 15.680 s/fc-hour | `proofs/perf/warmed_timing.json` |
| 24 h segmented warmed step | 45.537 ms/step | `proofs/perf/segscan_24h.json` |
| 24 h segmented measured wall | 393.44 s = 6.56 min | `proofs/perf/segscan_24h.json` |
| Coupled no-flag confirm | 43.95-44.09 ms/step; 24 h = 16.66 s/fc-hour in wind-fix confirm | `proofs/perf/fusion_confirm_results.md` |
| Cold compile, short warmed timing h1/h2 | 171.9 s / 419.7 s compile component | `proofs/perf/warmed_timing.json` |
| Persistent cache A/B | 257.7 s cold -> 33.5 s hot, 224.2 s removed | `proofs/perf/v020_wallclock_wins.md` |

Per-phase isolated timings:

| Phase | Isolated min ms | GB | GFLOP | Calls per step | Approx share of 43.56 ms step | Notes |
|---|---:|---:|---:|---:|---:|---|
| Thompson microphysics | 19.959 | 1.769 | 0.720 | 1 | 46% | Largest phase; sedimentation dominates. |
| Surface layer | 3.994 | 0.607 | 0.033 | 1 | 9% | Coupled before MYNN; likely fusion opportunity. |
| MYNN PBL | 2.897 | 1.304 | 0.197 | 1 | 7% | Column physics; output diagnostics recompute PBLH. |
| Advection tendencies | 0.991 | 0.655 | 0.212 | 3 | 7% if summed | RK-stage repeated stencil work. |
| Small-step prep | 0.998 | 0.763 | 0.030 | 3 | 7% if summed | Repeated pressure/geopotential setup. |
| EOS / pressure-density | 1.000 | 0.037 | 0.011 | 3 plus acoustic uses in broader path | launch dominated | Tiny bytes but ~1 ms floor. |
| Vertical Thomas/PCR solve | 0.999 | 0.060 | ~0 | 2 x 16 substeps | launch/solver overhead | Already lowered to cuSPARSE PCR per nsys. |
| Boundary apply | 1.000 | 0.866 | 0.042 | 1 | 2-5% | Full boundary-family updates. |
| Flux-advection augment | 0.678 | 0.786 | 0.288 | 3 | 5% if summed | Repeated mass/face averages. |
| calc_coef_w coefficients | 0.337 | 0.076 | 0.250 | 3 | 2% if summed | Hoist/cache candidate. |
| Halo apply | 0.521 | 0.562 | n/a | ~8 | 1-4% | Repeated stage entry/exit halo traffic. |

Source: `proofs/perf/phase_breakdown.json`. Shares are approximate because the phase timings are isolated and not a mutually exclusive timeline.

Dycore roofline:

| Metric | Value | Source |
|---|---:|---|
| dycore-only wall | 16.898 ms/step | `proofs/perf/roofline_costonly.json` |
| dycore bytes | 5.661 GB/step | same |
| dycore FLOPs | 2.263 GFLOP/step | same |
| arithmetic intensity | 0.400 FLOP/B | same |
| achieved HBM | 0.335 TB/s = 18.7% of 1.792 TB/s peak | same |
| achieved fp64 | 0.134 TF/s = 8.2% of 1.6388 TF/s peak | same |
| HBM floor | 3.159 ms | same |
| wall / HBM floor | 5.35x | same |

Interpretation: the dycore is below the fp64 ridge and far below the fp32 ridge. The dominant gap is launch/serialization and low effective bandwidth, not fp64 arithmetic.

Host materialization and daily wrapper:

- `daily_pipeline._run_forecast_sequence` calls `run_forecast_operational(state, namelist, 1.0)` once per hour, then runs `finite_summary(state)` and output diagnostics. `finite_summary` uses `np.asarray` over every State leaf. That is a full-state D2H check once after forecast and again after hourly land refresh.
- `wrfout_writer.prepare_wrfout_payload` pulls output fields leaf-by-leaf via `_coerce_array -> np.asarray`, and rebuilds static grid-coordinate fields each output.
- Async output already overlaps NetCDF writes, but `proofs/perf/v020_wallclock_wins.md` explicitly says the D2H pull is still synchronous. It estimated synchronous output/CPU work at ~110 s for d02 24 h and ~176 s for d03 24 h before overlap, with 60-85% recoverable by async writing after hour 1. The remaining D2H/CPU pack path is still large enough to deserve v0.10 attention.

## Fusion And Launch Barriers

Main barriers found in v0.9.0 source:

1. Acoustic `lax.scan` over substeps in `operational_mode._acoustic_scan`. The checked-out source has `jax.lax.scan(body, acoustic, xs=None, length=int(stage.number_of_small_timesteps))` with no active `unroll=` hook. This conflicts with published notes that mention `GPUWRF_ACOUSTIC_UNROLL`; the v0.9.0 source itself does not contain that hook.
2. RK/acoustic substep boundaries: 16 acoustic substeps per full step (1 + 5 + 10 across RK stages) multiply EOS, momentum, boundary work, vertical solve and halo traffic. The scan carry includes a large `AcousticCoreState`.
3. Thompson sedimentation in `physics/thompson_column.py`: four independent fixed-length masked scans, `NSED_MAX=64`, `unroll=_sed_unroll()` default 2. The scan is WRF-faithful but still executes the static upper bound even when per-column `nstep` is lower.
4. State is a very large pytree. `State.__slots__` carries live prognostics, base/perturbation/total aliases, boundary arrays, diagnostics and static-ish fields. `OperationalCarry` wraps the whole State plus scratch. This increases carry size and donation pressure.
5. Boundary forcing does separate `_apply_3d` work for `u`, `v`, `w`, `theta`, `qv`, `p`, `pb`, `mu`, `mub`, and optionally `ph`/`phb`; it uses time interpolation, slicing and replacement on full field families.
6. Physics adapters convert between z-major State fields and column-major physics inputs using `jnp.moveaxis`, reshape and reassembly. Some are bitcasts, but HLO must verify whether any become real transposes or layout-changing copies.
7. Surface and MYNN are separate adapters. Surface outputs are written to State and then read by MYNN; output diagnostics recompute surface/MYNN-derived values.
8. Radiation diagnostics call full RRTMG diagnostic kernels at output cadence, separate from the forecast radiation tendency/carry path.
9. Precision enforcement and field-level casts happen through `_enforce_operational_precision` and `State.replace`. With `force_fp64=True`, most fields should already match, but HLO should confirm no residual convert/copy chains.
10. Daily pipeline Python loop and host synchronizations are outside the timestep loop but material for the operational daily wrapper: hourly `block_until_ready`, `np.asarray` finite checks, per-field output `np.asarray`, and hourly public-JIT calls that reinitialize carry.

## Ranked Optimization Table

Estimated gain is warmed coupled-step gain unless marked `daily`, `cold`, or `later`. `Measured` ranges cite existing artifacts; `estimate` means no direct timing artifact exists yet.

| Rank | Inefficiency | Location | Est. gain | Effort | Stability risk | Category | Concrete fix approach |
|---:|---|---|---:|---|---|---|---|
| 1 | Acoustic substep scan is not unrolled/fused in the v0.9.0 source, despite measured prior A/B. | `src/gpuwrf/runtime/operational_mode.py:_acoustic_scan` | 8-22% measured/estimate. Prior artifact reports 1.225x for unroll=4; source mismatch must be revalidated. | M | Med | fusion/launch | Add source-level `unroll` hook or a hand-fused 2-substep/4-substep acoustic kernel. Start with unroll=2, then unroll=4 only if compile/memory gates pass. Validate idealized dycore, WRF fixture, and 24 h free-GPU run. |
| 2 | Thompson sedimentation still executes fixed `NSED_MAX=64` masked loop for each species even when WRF adaptive `nstep_col` may be far lower. | `src/gpuwrf/physics/thompson_column.py:_sed_one_species`, `_nstep_per_column`, `NSED_MAX` | 3-12% estimate; potentially higher in wet cases if histograms prove low max nstep. | M/L | High | fusion/algo/launch | First collect no-GPU-code-change histogram in manager GPU run: max/percentiles/clips by species/domain. If no clips at 16/32, specialize lower caps or bucket by nstep. Must preserve precip oracle and 24 h coupled skill. Do not silently cap. |
| 3 | Full-State hourly finite checks materialize every State leaf to host, often twice per output hour. | `src/gpuwrf/integration/daily_pipeline.py:finite_summary`, `_run_forecast_sequence` | 2-8% daily estimate | S/M | Low | mem/host | Replace with a jitted device finite summary returning scalar booleans/counts/ranges for selected critical leaves. Pull only scalars. Keep an opt-in full host audit for validation runs. |
| 4 | wrfout payload path performs many independent D2H pulls and recomputes static/derived fields on host every output. | `src/gpuwrf/io/wrfout_writer.py:prepare_wrfout_payload`, `_build_output_fields`, `_coerce_array`, `_add_grid_coordinate_fields` | 2-8% daily estimate; v020 output/CPU was ~110 s d02 24 h before async overlap. | M | Low/Med | mem/host | Build a device-side output packer for all hourly fields, `device_get` the packed tree once, cache static lat/lon/map/grid arrays, compute CLDFRA/TH2/unstagger on device or in one pack function. |
| 5 | Surface and MYNN are separate physics adapters with State write/read boundary; output diagnostics recompute PBLH/surface diagnostics. | `operational_mode._physics_boundary_step_with_limiter_diagnostics`; `coupling/physics_couplers.py:surface_adapter`, `mynn_adapter`, `surface_layer_diagnostics` | 3-8% estimate from 4.0 ms surface + 2.9 ms MYNN phase sizes | M | Med | fusion/mem | Implement a combined surface+MYNN adapter that keeps columns/fluxes live and returns State once. Add PBLH/diagnostic side-channel from the MYNN call rather than recomputing at output. |
| 6 | Daily wrapper re-enters public `run_forecast_operational(..., 1.0)` hourly, reinitializing `OperationalCarry`, scratch, held `rthraten`, and step index. | `daily_pipeline._default_forecast_fn`, `_run_forecast_sequence`; `operational_state.initial_operational_carry`; `operational_mode.run_forecast_operational` | 2-10% daily estimate; cold/cache effects larger | L | High | launch/mem/algo | Add a resident carry-threaded daily driver using `_advance_chunk`/single-scan semantics, with land refresh and M9 snapshots applied at hour boundaries without resetting global step/carry. Requires explicit product-equivalence gate because radiation timing/carry semantics can change. |
| 7 | Redundant halo applications around RK stage entry/exit and acoustic return. | `operational_mode._rk_scan_step`, `_acoustic_scan`; `apply_halo` calls | 2-5% estimate; halo phase 0.52 ms isolated with ~8 passes | M | Med | mem/launch | Audit halo-validity contract per stage. Remove stage-entry or stage-exit duplicates where the next consumer already applies halo. Keep boundary-ring tests and idealized gates. |
| 8 | Stage-constant dry arrays, coefficient fields, zero scratch and scalar-base full arrays are rebuilt repeatedly. | `operational_mode._acoustic_core_state_from_prep`, `_acoustic_scan`, `_augment_large_step_tendencies`; `dry_cqw`, `calc_coef_w_wrf_coefficients`, `zeros_like`, `theta_base = scalar * ones_like` | 1-4% estimate; calc_coef_w 0.34 ms x3 plus many tiny kernels | S/M | Low | fusion/mem/compute | Precompute `dry_cqw`, metric inverses, static zero templates and base-column constants in metrics/carry. Use scalar broadcasting instead of full `ones_like` where possible. HLO must show fewer kernels/copies. |
| 9 | `_advance_chunk` lacks donation even though it carries large resident state across host segments. | `operational_mode._advance_chunk`; v020 note says no donation to protect async snapshots | 1-3% estimate | S/M | Med | mem | Add a donating forecast-only `_advance_chunk` variant for non-output segments and keep a non-donating snapshot path. Verify alias report and async output safety. |
| 10 | Boundary apply updates full field families and interpolates boundary leaves each step. | `coupling/boundary_apply.py:apply_lateral_boundaries`, `_apply_3d`, `interpolate_boundary_leaf`; acoustic nested boundary targets in `operational_mode._acoustic_core_state_from_prep` | 1-3% estimate; boundary phase 1-2 ms, 0.866 GB | M | Med/High | mem/launch | Precompute time interpolation coefficients, specialize strip updates, pack boundary strips, and avoid full-family materialization. Nested `ph/w` branches need separate stability proof. |
| 11 | Production guards and limiters still run in the hot path when `disable_guards=False`; some perf proofs use guards off, daily defaults may not. | `operational_mode._positive_definite_theta_increment_limiter`, `_valid_mixing_ratio`, `_finite_or_origin`, calls in `_physics_boundary_step_with_limiter_diagnostics` | 1-4% estimate | S/M | High | compute/mem | Count guard activations over validation cases. For operational performance mode, replace per-field full guards with cheap scalar counters or run guards at proof/output cadence. Only disable if written safety policy accepts it. |
| 12 | Pressure/EOS/geopotential diagnostics are recomputed between large-step tendencies, small-step prep and stage refresh. | `operational_mode._refresh_grid_p_from_finished`, `_augment_large_step_tendencies`, `_acoustic_core_state_from_prep`, `_physics_boundary_step_with_limiter_diagnostics` | 1-3% estimate | M | High | compute/mem | Identify identical pressure/al/alt intermediates and thread them through prep/refresh. Do not remove WRF-required recomputations unless an analytic/fixture proof shows equivalence. |
| 13 | State hot carry includes duplicate aliases and cold/static/boundary leaves. | `contracts/state.py:State.__slots__`, `tree_flatten`; `runtime/operational_state.py:OperationalCarry` | 3-8% estimate | L | High | mem/architecture | ADR-level split: hot prognostic carry, static sidecar, boundary sidecar, diagnostics/output sidecar. Preserve public State API through wrapper conversion. Re-run full validation; large blast radius. |
| 14 | M9/output diagnostics recompute RRTMG and surface/MYNN-derived quantities instead of using held/cached outputs. | `operational_mode.compute_m9_diagnostics`; `physics_couplers.rrtmg_radiation_diagnostics`, `surface_layer_diagnostics`; `daily_pipeline._surface_diagnostics_for_output` | 1-5% daily estimate | M | Med | compute/mem | Persist required radiation/surface/PBL diagnostic outputs in carry at physics cadence. Use held `rthraten`/SWDOWN/GLW where valid. Add PBLH side-channel during MYNN. |
| 15 | Dynamics-only command-buffer win was global-flag rejected for coupled step; scoped dycore-only launch batching remains untested. | XLA/JAX compile configuration around dycore/acoustic call boundary | 5-15% estimate if scoping works; global flag measured -15% to -21% | M | High | launch | Do not set global `--xla_gpu_graph_min_graph_size=1`. Only test per-jit/per-submodule graph capture if JAX supports scoped compile options without introducing a new JIT boundary. Needs A/B proof; lower priority than direct acoustic fusion. |
| 16 | Precision policy currently cannot pay off because launch count and fp64 acoustic conversions dominate. | `contracts/precision.py`; `operational_mode._enforce_operational_precision`; physics adapters | 0% now measured; 5-15% later estimate after fusion | M | Med/High | precision/mem | Defer. After acoustic/Thompson fusion reduces launch tax, move fp32/fp64 boundaries outside inner substeps and retest gated fp32 storage for u/v/theta/moisture. Avoid per-substep f32<->f64 converts. |
| 17 | `time_utc` and static bundle identity fragment the persistent compile cache across daily initializations. | `OperationalNamelist.tree_flatten`, `_StaticHolder`, `time_utc`; `proofs/perf/v020_wallclock_wins.md` | cold-only: huge; 224 s cache win measured for repeat signature, daily different-init currently misses | M | Med/High | compile/launch | Normalize static keys or make radiation clock dynamic only under RMSE-equivalence, not bitwise, gate. Prior dynamic clock failed bit-identical full-forecast gate by max p_pert 4.46 Pa but standalone radiation was bit-identical. |
| 18 | `run_forecast_operational` Python while inside JIT creates one scan per radiation interval for static radiation gating; single-scan warmed unchanged but compile/memory behavior differs. | `operational_mode.run_forecast_operational`, `_scan_forecast_segment`, `run_forecast_operational_single_scan`, `run_forecast_operational_segmented` | 1-3% warm estimate; compile/memory larger for long runs | M | Med | launch/compile | Prefer segmented resident chunks for daily operational use. Keep radiation segments explicit if they reduce non-rad branch cost; use single-scan only where compile stability wins. |
| 19 | Column adapters use `moveaxis`/reshape around Thompson, MYNN and RRTMG; some may lower to copies or fusion barriers. | `physics_couplers._to_columns`, `_from_columns`, `_flatten_columns`, `_unflatten_columns`; RRTMG `moveaxis` patterns | 1-4% estimate | M | Med | mem/fusion | Inspect HLO layouts. If copies exist, standardize physics layout or pack all column fields once per physics block. Avoid repeated z-major/column-major conversions. |
| 20 | Final `_enforce_operational_precision` runs every step and may generate identity casts or converts across many State leaves. | `operational_mode._enforce_operational_precision`, call at end of `_physics_boundary_step_with_limiter_diagnostics`; `State.replace` dtype casts | 1-3% estimate if HLO shows converts; 0 if fully elided | S | Low | precision/mem | StableHLO/HLO audit for `convert` and copy count. Skip enforcement when all fields already have target dtype, or enforce only after known dtype-changing physics. |
| 21 | Mass face averages and metric inverses are recomputed across RK-stage advection, acoustic prep and pressure-gradient work. | `operational_mode._augment_large_step_tendencies`, `_acoustic_core_state_from_prep`, `_acoustic_scan` | 1-3% estimate | M | Med | compute/mem | Hoist `mass_u/v/w`, `muu/muv/muw`, inverse map factors and stage constants into a per-stage prep struct. Recompute cheaper than store only where HLO proves it. |
| 22 | RRTMG g-point/transient layout and output diagnostics are expensive at radiation/output cadence, with large transient memory. | `physics/rrtmg_sw.py`, `physics/rrtmg_lw.py`, `physics_couplers.rrtmg_theta_tendency`, `rrtmg_radiation_diagnostics` | 1-5% amortized estimate | L | High | algo/mem | Profile radiation-only with long warmup. Cache invariant pressure/gas/table transforms, split heating vs diagnostics, and avoid recomputing full diagnostics at output. WRF-fidelity risk is high. |
| 23 | Public daily path blocks and materializes after every hour even when no output/error requires full synchronization. | `daily_pipeline._run_forecast_sequence`; `_default_forecast_fn`; `block_until_ready` | 1-3% daily estimate | S/M | Low/Med | host/launch | Keep synchronization at output boundaries only, not duplicated around finite checks and payload preparation. With device finite summary, pull small scalars asynchronously before output payload. |
| 24 | Static grid coordinate, map and land metadata are rebuilt/pulled for each wrfout. | `wrfout_writer._add_grid_coordinate_fields`, `_build_output_fields`; `daily_pipeline._build_real_case` | 1-2% daily estimate | S | Low | mem/host | Cache host copies of XLAT/XLONG/map factors/static land masks once per run and reuse in every payload. |
| 25 | Boundary interpolation uses dynamic `take` each step over time dimension. | `boundary_apply.interpolate_boundary_leaf`; per-field `_apply_3d` calls | 1-2% estimate | S/M | Med | mem/launch | Precompute lower/upper indices and alpha per step/segment, or pass current boundary strip bundle into the step body. Preserve hourly boundary interpolation semantics. |
| 26 | Thompson per-species scans are independent but not co-scheduled with source/sink or post-update casts; batched 4-species scan regressed, but narrower fusion may remain. | `thompson_column._sedimentation`, `_sed_one_species`, `_state_from_thompson_output` | 1-4% estimate | M/L | Med/High | fusion/mem | Do not repeat rejected 4-species batching. Instead test fusing casts/reassembly and species pairs only after nstep-cap work. Must use Thompson oracle and coupled timing. |
| 27 | Acoustic boundary target construction builds full target/relax arrays each substep/stage even when nested flags are off or self-replay path is simple. | `operational_mode._acoustic_core_state_from_prep`; `AcousticCoreConfig`; boundary work helpers | 1-3% estimate | M | High | mem/compute | Specialize specified/nested/periodic boundary configurations at trace time and elide unused target construction. Requires nested d03 stability proof because comments document prior ph/w boundary failures. |
| 28 | Daily restart/repeat probes and scoring can rerun forecasts or force extra materialization in production-like pipeline runs. | `daily_pipeline.execute_daily_pipeline`, restart/repeat probe sections | variable; >1% when enabled | S | Low | host/algo | Ensure production forecast path disables validation probes by default; run probes only in validation jobs. |
| 29 | Small 3-D zero/save-family fields are recreated or overwritten with zero-like arrays at transitions. | `operational_mode._with_save_family`; `operational_state.initial_operational_carry` | 1-2% estimate | S/M | Low | mem | Reuse persistent zero buffers in carry/metrics or avoid writes for fields not consumed by the next stage. HLO must show fewer memset/copy ops. |
| 30 | Compile/autotuning measurements are not all from a long-warmup Nsight run, so optimization priority could be skewed by Redzone/Delay autotuning artifacts. | Profiling harnesses under `proofs/perf` | planning gain, not direct runtime | S | Low | profiling | Manager should serialize one fresh long-warmup Nsight run after the first v0.10 changes: >=200 warm steps, guards on/off variants, full coupled and dycore-only. This prevents optimizing artifacts. |

## Recommended Sequencing

Do not implement all changes at once. Too many of these change fusion, scheduling, carry layout or output semantics, and the validation/debugging cost would explode. Use phased implementation with proof gates after each group.

Phase 0: serialize missing profiling/proofs before code changes.

- Fresh long-warmup Nsight Systems run on the current v0.9.0 source, full coupled and dycore-only, guards on/off.
- HLO/StableHLO audit for acoustic scan, `_enforce_operational_precision`, column layout transposes, `State.replace` copies, and `_advance_chunk` donation/aliasing.
- Thompson `nstep_col` histogram by species and domain over representative d02/d03 wet columns; report max, P99, P99.9 and clip count for `NSED_MAX=16/32/64`.
- Daily wrapper timing breakdown: forecast, finite checks, M9 diagnostics, output pack D2H, NetCDF write, land refresh.

Phase 1: low-risk host/materialization and obvious HLO cleanup.

- Device-side finite summary instead of hourly full-State D2H.
- wrfout output packer and static-grid cache.
- HLO-confirmed no-op precision-enforcement skip.
- Static zero/dry/metric hoists that do not alter math.

Why first: these are low risk, reduce daily wall immediately, and create cleaner timing for kernel work.

Phase 2: launch/fusion work in the dycore/acoustic loop.

- Add or restore the source-level acoustic scan unroll hook; start with `unroll=2`, then evaluate `unroll=4`.
- Reduce redundant halos after a halo-validity audit.
- Hoist stage constants and mass/metric face averages only where HLO shows material traffic.

Why second: this attacks the known ~6,890 tiny ops/step and the 5.35x dycore wall/HBM-floor gap. It is the main kernel lever. It also unlocks later precision work by reducing launch tax.

Phase 3: physics coupling consolidation.

- Fuse surface+MYNN coupling and preserve PBLH/surface diagnostics as side-channel outputs.
- Reuse RRTMG/M9 diagnostic outputs where forecast carry already has them.
- Inspect and remove physics layout copies.

Why third: surface+MYNN are sizable but less central than acoustic launch count. These changes touch physics contracts, so they need coupled fixture gates.

Phase 4: Thompson sedimentation structural optimization.

- Only proceed if Phase 0 histograms prove lower caps or bucketing can be WRF-faithful with zero clips.
- Test lower static `NSED_MAX` variants or bucketed scans behind a gate.
- Keep implicit sedimentation off by default; it is a fidelity tradeoff, not a v0.10 default optimization.

Why fourth: Thompson is the largest phase, but the safe shipped unroll=2 is already in place, and the remaining faithful options require strong wet-case proof.

Phase 5: daily resident carry and architecture changes.

- Build a resident carry-threaded daily driver with land refresh and M9 snapshots.
- Consider State hot/cold split only after smaller wins and with an ADR.

Why later: these are high blast-radius changes. They can improve daily wall and compile/cache behavior, but they alter product semantics unless carefully gated.

Phase 6: precision re-entry.

- Re-run gated fp32 storage after Phase 2/3/4 reduce launch count and after dtype boundaries are moved outside inner loops.
- Expected current gain is ~0 because fp32 ~= fp64 in existing artifacts. A later gain is plausible only when bandwidth is the exposed bottleneck and conversion traffic is minimized.

## Theoretical-Optimum Estimate

Hardware and current facts:

- RTX 5090 peaks in `roofline_costonly.json`: 104.9 TF/s fp32, 1.6388 TF/s fp64, 1.792 TB/s HBM.
- Dycore clean bandwidth floor: 3.16 ms/step, current dycore 16.90 ms/step, so the dycore has a 5.35x launch/serialization gap.
- Current coupled warmed step: 43.6-46.3 ms/step depending artifact/run, or 15.7-16.7 s/fc-hour.
- Current d02 speedup: ~5.0-5.3x clean CPU denominator, ~7.4-7.8x realistic denominator.

The raw HBM floor for individual phases is not the true theoretical floor, because the acoustic solve, RK stage dependencies, Thompson sedimentation and boundary forcing have causal vertical/time dependencies and required global-memory round trips. A realistic WRF-faithful practical ceiling on one RTX 5090 is:

- dycore/acoustic: 5-8 ms/step if most scan launch tax is eliminated but vertical/PCR and stencil dependencies remain;
- Thompson: 12-16 ms/step for strict explicit WRF-faithful sedimentation unless a no-clipping lower `NSED_MAX`/bucketing proof succeeds; lower than that likely requires an algorithmic sedimentation tradeoff;
- surface+MYNN+boundary+halo+misc: 5-8 ms/step after coupling fusion, halo trimming and hoists;
- radiation amortized and output wrapper: mostly outside the non-radiation step but material for daily wall.

Realistic v0.10 faithful target after top levers: about 25-32 ms/step warmed coupled, or 9.0-11.5 s/fc-hour. That is a 1.35-1.75x improvement over v0.9.0 warmed throughput, roughly 7-9x versus the clean CPU denominator and 11-14x versus the realistic denominator.

Aggressive high-risk ceiling: 18-24 ms/step if acoustic fusion lands, Thompson lower-cap/bucketing proves safe, surface+MYNN fuse cleanly, and output/D2H overhead is removed. That would be 6.5-8.6 s/fc-hour, roughly 10-13x clean and 14-19x realistic. This is not guaranteed and should not be promised without the Phase 0/2/4 proof gates.

Not WRF-faithful default ceiling: implicit Thompson sedimentation can cut the Thompson kernel by ~2.25-2.44x, but existing precip oracle rejects it as default due over-precipitation and diffusion. It belongs in an ADR-gated research branch, not the v0.10 faithful kernel plan.

## Precision Sequencing

Current measured precision result:

- fp32 dynamics ~= 1.00x, because the dycore is launch/memory-bound and the mandatory fp64 acoustic island adds conversion traffic.
- fp32 Thompson ~= 1.0x or worse in tiled tests, because Thompson is dominated by launch/bandwidth-bound sedimentation, not arithmetic.

Precision should be delayed until:

1. acoustic scan fusion/unroll reduces launch count;
2. Thompson sedimentation structural work reduces scan count or cap length;
3. dtype conversions are moved out of RK/acoustic inner loops;
4. HLO proves fp32 storage reduces bytes without adding convert kernels.

Then retest the gated precision matrix with WRF fixtures and daily output scores. Plausible later gain is 5-15% on the full coupled step, mainly from lower HBM traffic in non-acoustic fields and physics, not from fp32 arithmetic throughput.

## Algorithmic Observations

- Thompson implicit sedimentation is the only already-measured large algorithmic speedup, but it is a fidelity tradeoff and currently rejected as default.
- Radiation cadence reduction or cheaper diagnostics would be a fidelity/product tradeoff unless it only reuses already-computed fields.
- Boundary forcing has known d03/nested stability sensitivities around `ph/w` forcing. Any boundary optimization must preserve current specified/nested semantics exactly.
- The vertical implicit solve is already lowered to cuSPARSE PCR; replacing it with a hand Thomas scan is not a promising direction.
- Changing daily carry continuity changes radiation/carry semantics and hourly land-refresh interaction; this is not a pure performance refactor.

## Noted But Out Of Scope For v0.10 (>Risk, <1%, Negative, Or Already Rejected)

| Item | Why not an action item |
|---|---|
| Global `--xla_gpu_graph_min_graph_size=1` | Measured 15-21% slower on coupled operational step, despite dynamics-only win. Do not bake globally. |
| fp32 dynamics before fusion | Measured ~1.00x; conversion traffic and launch-bound behavior erase benefit. |
| fp32 Thompson before sedimentation structural work | Measured ~1.0x or regression; keep opt-in only. |
| Batched four-species Thompson sedimentation | Measured regression: 42.91 ms versus 33.83 ms base on tiled workload. |
| Implicit Thompson sedimentation as default | Kernel 2.25-2.44x faster but fidelity rejected: +47% precip for nsub=1 versus WRF oracle. |
| More Thompson unroll beyond 2 | Existing tiled results show unroll4 worse than unroll2. |
| Debug print path | `jax.debug.print` is guarded by static `debug`; operational path passes `debug=False`. No hot-path issue found. |
| Host callbacks inside timestep JIT | No `device_get`/`np.asarray`/callbacks found inside the jitted timestep body; host sync is in daily/output wrappers. |
| Python branch cleanup with no HLO effect | Only worth doing if HLO or timing shows fewer kernels/compiles. Avoid cosmetic refactors. |
| Static grid-coordinate caching alone | Likely around or below 1% by itself, but should be folded into output-packer work. |
| Inventory/scoring/restart probes | Disable for production runs if enabled; otherwise not part of the operational kernel. |
| One-off compile cache read overhead | Persistent cache already shipped. Further cold-cache work is valuable only for daily different-init reuse, not warmed kernel speed. |

## Required Manager-Serialized Fresh Runs

No GPU runs were performed for this analysis. Recommended serialized GPU/profiler runs:

1. Long-warmup Nsight Systems, >=200 warmed steps, current v0.9.0 source, full coupled and dycore-only, guards on/off.
2. Acoustic unroll source-hook A/B: unroll=1,2,4; full coupled, 24 h, free GPU, compile time, memory, idealized gates.
3. Thompson `nstep_col` histogram and lower-cap A/B if histogram permits.
4. Daily wrapper timing breakdown with timers around forecast, finite summary, M9 diagnostics, payload D2H, NetCDF write and land refresh.
5. HLO/StableHLO and buffer assignment audit for converts, transposes, donation aliasing and scan unroll effects.
6. Guard activation counters over the validation corpus before any guard fast-path default.

## Handoff

Objective: produce a comprehensive, broad, ranked v0.10.0 optimization analysis for the shipped v0.9.0 operational JAX GPU kernel and daily wrapper.

Files changed: `/tmp/gpt_v0100_analysis.md` and `/tmp/gpt_v0100_analysis.done` only. No repository files changed.

Commands run: read-only `rg`, `sed`, `jq`, `git status`, `git rev-parse`, and `git tag --points-at HEAD` against the v0.9.0 worktree and existing proof artifacts.

Proof objects produced: this analysis file and the sentinel file. No new GPU proof artifacts were produced.

Unresolved risks: no fresh Nsight/HLO run was performed; several gains are estimates and must be proven by manager-serialized GPU runs. The acoustic-unroll artifact/source mismatch needs direct verification before implementation.

Next decision needed: choose Phase 1/2 ordering and whether v0.10 accepts RMSE-equivalence rather than bit-identical equivalence for cache-key/dynamic-clock and acoustic-unroll scheduling changes.
