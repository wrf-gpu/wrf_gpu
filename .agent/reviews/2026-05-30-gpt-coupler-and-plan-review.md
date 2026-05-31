# GPT Coupler + Plan Review

Independent adversarial review requested by `.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:1`. Scope was read-only code review of Part A physics/coupling bugs and Part B remaining technical hurdles; validation/publishing was deliberately skipped per `.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:1` and `.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:13`.

Branch/context checked: `manager-2026-05-23`, commit target `17e0039` as specified in `.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:1`. Code was not modified. The only produced artifact is this review file.

## Part A — Findings

### P0-1 — `force_fp64` is still defeated inside the physics couplers

The operational path force-upcasts every state field when `force_fp64=True` (`src/gpuwrf/runtime/operational_mode.py:319`, `src/gpuwrf/runtime/operational_mode.py:325`, `src/gpuwrf/runtime/operational_mode.py:331`), and `State.replace()` was explicitly changed so `_cast=False` can preserve those fp64 updates (`src/gpuwrf/contracts/state.py:566`, `src/gpuwrf/contracts/state.py:574`, `src/gpuwrf/contracts/state.py:576`). However, the physics adapters do not preserve the current field dtype. They call `_field_dtype()`, which returns the frozen default precision matrix (`src/gpuwrf/coupling/physics_couplers.py:346`, `src/gpuwrf/coupling/physics_couplers.py:349`), where `u`, `v`, `theta`, `qv`, hydrometeors, number fields, and `qke` are fp32-gated (`src/gpuwrf/contracts/precision.py:89`, `src/gpuwrf/contracts/precision.py:92`, `src/gpuwrf/contracts/precision.py:100`, `src/gpuwrf/contracts/precision.py:105`).

Concrete downcast sites:

- Thompson writes `theta`, `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr`, `Ns`, and `Ng` through `_field_dtype()` (`src/gpuwrf/coupling/physics_couplers.py:605`, `src/gpuwrf/coupling/physics_couplers.py:608`, `src/gpuwrf/coupling/physics_couplers.py:614`).
- MYNN writes `u`, `v`, `theta`, `qv`, and `qke` through `_field_dtype()` (`src/gpuwrf/coupling/physics_couplers.py:739`, `src/gpuwrf/coupling/physics_couplers.py:742`, `src/gpuwrf/coupling/physics_couplers.py:744`).
- RRTMG writes `theta` through `_field_dtype("theta")` (`src/gpuwrf/coupling/physics_couplers.py:903`, `src/gpuwrf/coupling/physics_couplers.py:910`, `src/gpuwrf/coupling/physics_couplers.py:911`).

In the operational chain, this happens inside a single timestep before the end-of-step precision enforcement (`src/gpuwrf/runtime/operational_mode.py:1477`, `src/gpuwrf/runtime/operational_mode.py:1479`, `src/gpuwrf/runtime/operational_mode.py:1486`, `src/gpuwrf/runtime/operational_mode.py:1513`). So the force-fp64 path is not truly fp64 through the physics sequence; it downcasts active physics tendencies and only re-upcasts after physics/boundaries. That is exactly the bug class called out for review (`.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:10`).

Fix: coupler reassembly must cast to the current state field dtype, not `DEFAULT_DTYPES`, when `force_fp64` has already changed the carry. The minimal pattern is `getattr(state, field).dtype` at adapter boundaries, or rely on `state.replace()` default casting to the current dtype. Keep `_cast=False` only for explicit precision-mode enforcement.

### P0-2 — The radiation-cadence "fix" is still not WRF-equivalent

The task asks to verify that the 180x cadence bug was fixed (`.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:8`). The current code does not hold and apply a radiation tendency every dynamics step. It skips radiation for `cadence-1` steps, then calls RRTMG once and applies `dt * cadence * heating_rate` as a single lump (`src/gpuwrf/runtime/operational_mode.py:1480`, `src/gpuwrf/runtime/operational_mode.py:1486`, `src/gpuwrf/runtime/operational_mode.py:1492`; `src/gpuwrf/coupling/physics_couplers.py:903`, `src/gpuwrf/coupling/physics_couplers.py:910`). The legacy coupled driver has the same "non-radiation block, then one radiation step" structure (`src/gpuwrf/coupling/driver.py:627`, `src/gpuwrf/coupling/driver.py:629`, `src/gpuwrf/coupling/driver.py:641`).

WRF stores radiation as a theta tendency in K/s (`/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/phys/module_radiation_driver.F:415`) and decides when to refresh it by the radiation cadence (`/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/phys/module_radiation_driver.F:1115`, `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/phys/module_radiation_driver.F:1127`). That tendency is then included in physics tendencies (`/home/enric/src/wrf_pristine/WRF/dyn_em/module_first_rk_step_part2.F:392`, `/home/enric/src/wrf_pristine/WRF/dyn_em/module_first_rk_step_part2.F:394`) and is added into `rt_tendf` by `phy_ra_ten()` (`/home/enric/src/wrf_pristine/WRF/phys/module_physics_addtendc.F:131`, `/home/enric/src/wrf_pristine/WRF/phys/module_physics_addtendc.F:133`, `/home/enric/src/wrf_pristine/WRF/phys/module_physics_addtendc.F:229`). `update_phy_ten()` receives `grid%rthraten` in the RK path (`/home/enric/src/wrf_pristine/WRF/dyn_em/module_first_rk_step_part2.F:801`, `/home/enric/src/wrf_pristine/WRF/dyn_em/module_first_rk_step_part2.F:803`).

Lumping the full interval at the cadence step is not equivalent to applying a held rate each step, because the intervening dynamics, microphysics, surface layer, and PBL see a different temperature trajectory. It also changes boundary/guard interaction timing.

Fix: add resident radiation tendency state, for example `rthraten` or `rthratensw/lw`, refreshed only when the radiation driver runs and applied every timestep as `dt * held_rate`. If the project intentionally chooses the lumped approximation, it needs an explicit proof that the altered trajectory is acceptable; this code does not provide one.

### P1-3 — Public coupled driver still runs MYNN before surface

The operational path uses the expected `thompson -> surface -> mynn -> rrtmg -> boundary -> guards -> precision` order (`src/gpuwrf/runtime/operational_mode.py:1469`, `src/gpuwrf/runtime/operational_mode.py:1477`, `src/gpuwrf/runtime/operational_mode.py:1478`, `src/gpuwrf/runtime/operational_mode.py:1479`, `src/gpuwrf/runtime/operational_mode.py:1486`, `src/gpuwrf/runtime/operational_mode.py:1495`, `src/gpuwrf/runtime/operational_mode.py:1513`). But the public coupled driver path still calls MYNN before surface (`src/gpuwrf/coupling/driver.py:897`, `src/gpuwrf/coupling/driver.py:900`, `src/gpuwrf/coupling/driver.py:902`).

That means MYNN consumes stale or zero surface flux handles, because the intended handoff is that `surface_adapter()` writes flux handles (`src/gpuwrf/coupling/physics_couplers.py:793`, `src/gpuwrf/coupling/physics_couplers.py:804`, `src/gpuwrf/coupling/physics_couplers.py:806`, `src/gpuwrf/coupling/physics_couplers.py:809`) and `mynn_adapter()` reads those handles as its bottom boundary (`src/gpuwrf/coupling/physics_couplers.py:686`, `src/gpuwrf/coupling/physics_couplers.py:695`, `src/gpuwrf/coupling/physics_couplers.py:699`, `src/gpuwrf/physics/mynn_pbl.py:493`, `src/gpuwrf/physics/mynn_pbl.py:496`).

Fix: swap `surface_adapter()` before `mynn_adapter()` in `coupling/driver.py`, or deprecate/route the public driver through the operational path. Leaving a stale public path is risky because `src/gpuwrf/coupling/__init__.py:6` exports `run_forecast_segment` and validation modules still import it.

### P1-4 — Thompson rain-freezing lookup collapses WRF's ice-nuclei dimension

The Thompson table loader hardcodes `default_in_index = 27` and slices all rain-freezing tables at that index (`src/gpuwrf/physics/thompson_tables.py:99`, `src/gpuwrf/physics/thompson_tables.py:104`, `src/gpuwrf/physics/thompson_tables.py:107`, `src/gpuwrf/physics/thompson_tables.py:110`). The column kernel then indexes only `idx_r`, `idx_r1`, and `idx_tc` (`src/gpuwrf/physics/thompson_column.py:632`, `src/gpuwrf/physics/thompson_column.py:634`, `src/gpuwrf/physics/thompson_column.py:635`).

WRF computes `idx_IN` dynamically from ice number concentration (`/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:2588`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:2589`) and uses it for `tpg_qrfz`, `tpi_qrfz`, `tni_qrfz`, and `tnr_qrfz` (`/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:2595`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:2599`). The JAX port therefore cannot match WRF in active rain-freezing/ice-nucleation regimes even if the rest of the table interpolation is correct.

There is an adjacent number-concentration gap: graupel mass can be created by table freezing (`src/gpuwrf/physics/thompson_column.py:645`, `src/gpuwrf/physics/thompson_column.py:656`), but `Ng` is not incremented in the state update (`src/gpuwrf/physics/thompson_column.py:653`, `src/gpuwrf/physics/thompson_column.py:657`, `src/gpuwrf/physics/thompson_column.py:658`). Graupel sedimentation later falls back to an artificial number concentration when `Ng <= 0` (`src/gpuwrf/physics/thompson_column.py:870`, `src/gpuwrf/physics/thompson_column.py:873`, `src/gpuwrf/physics/thompson_column.py:876`).

Fix: keep the full `idx_IN` dimension in the table asset and compute the WRF index from `Ni`. Add the missing graupel number tendency for pathways that create graupel, or document and prove why WRF's `ng` pathway is intentionally omitted.

### P1-5 — Thompson sedimentation is not WRF's variable-CFL sedimentation and can lose conservation through flooring

The port explicitly uses a fixed `NSED_SUBSTEPS = 64` (`src/gpuwrf/physics/thompson_column.py:785`, `src/gpuwrf/physics/thompson_column.py:790`). WRF instead computes `nstep` from terminal fall speed and `dzq` per species (`/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3588`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3634`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3637`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3641`) and then loops exactly that `nstep` for rain (`/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3790`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3791`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3792`).

The JAX implementation computes surface precip from raw bottom flux and then floors the column fields (`src/gpuwrf/physics/thompson_column.py:918`, `src/gpuwrf/physics/thompson_column.py:920`, `src/gpuwrf/physics/thompson_column.py:923`). WRF also floors species, but it gates accumulated rain by the post-update surface rain threshold (`/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3811`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3817`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:3818`). The fixed JAX substep can be stable for a subset of profiles, but it is not WRF-faithful and needs a water budget proof for precip plus column hydrometeors.

Fix: implement WRF-style per-species dynamic `nstep` with a static upper bound and masked scan, or produce a conservation proof for the fixed 64-substep approximation across rain/ice/snow/graupel cases. Dry/near-inactive validation does not cover this.

### P1-6 — Thompson validity fallback can return invalid precip side effects

The Thompson kernel builds a thermodynamic validity mask (`src/gpuwrf/physics/thompson_column.py:424`, `src/gpuwrf/physics/thompson_column.py:433`) and saves a fallback state before running physics (`src/gpuwrf/physics/thompson_column.py:989`, `src/gpuwrf/physics/thompson_column.py:991`). It runs the full process chain, including sedimentation/precip (`src/gpuwrf/physics/thompson_column.py:993`, `src/gpuwrf/physics/thompson_column.py:998`), then restores the fallback state in invalid cells (`src/gpuwrf/physics/thompson_column.py:1004`). The precip dictionary is returned unmasked (`src/gpuwrf/physics/thompson_column.py:1005`).

That means an invalid cell can no-op the state but still contribute rain/snow/graupel/ice accumulation. Even if rare, this is a silent accounting bug in exactly the path where guards/fallbacks are active.

Fix: either gate invalid cells before sedimentation or apply the validity mask to every precip channel before returning.

### P1-7 — `sfclayrev` versus `sf_mynn` is a real coupled-skill risk, not just a diagnostic difference

The JAX surface layer is explicitly a `sf_sfclayrev` port (`src/gpuwrf/physics/surface_layer.py:1`, `src/gpuwrf/physics/surface_layer.py:7`, `src/gpuwrf/physics/surface_layer.py:12`). The task notes that the oracle used `sf_mynn` (`.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:6`). These are not interchangeable for coupled skill because the surface layer does not only output diagnostics; it writes flux handles consumed by MYNN (`src/gpuwrf/coupling/physics_couplers.py:793`, `src/gpuwrf/coupling/physics_couplers.py:804`, `src/gpuwrf/coupling/physics_couplers.py:806`, `src/gpuwrf/coupling/physics_couplers.py:809`; `src/gpuwrf/coupling/physics_couplers.py:686`, `src/gpuwrf/coupling/physics_couplers.py:695`, `src/gpuwrf/coupling/physics_couplers.py:699`).

There are concrete algorithmic differences relevant to the reported V10/PBL coupling:

- The JAX port clips reported `zol` to the MYNN band while still identifying as `sfclayrev` (`src/gpuwrf/physics/surface_layer.py:412`, `src/gpuwrf/physics/surface_layer.py:413`, `src/gpuwrf/physics/surface_layer.py:417`).
- It alters WRF's `ust` warm-start averaging for cold-start cases (`src/gpuwrf/physics/surface_layer.py:538`, `src/gpuwrf/physics/surface_layer.py:547`, `src/gpuwrf/physics/surface_layer.py:549`), while WRF `sfclayrev` uses the 50/50 update (`/home/enric/src/wrf_pristine/WRF/phys/physics_mmm/sf_sfclayrev.F90:756`).
- `sf_mynn` computes water roughness and scalar roughness through its own MYNN surface-layer options (`/home/enric/src/wrf_pristine/WRF/phys/module_sf_mynn.F:631`, `/home/enric/src/wrf_pristine/WRF/phys/module_sf_mynn.F:638`, `/home/enric/src/wrf_pristine/WRF/phys/module_sf_mynn.F:675`) and has resolution-dependent 10 m wind diagnostics (`/home/enric/src/wrf_pristine/WRF/phys/module_sf_mynn.F:1108`, `/home/enric/src/wrf_pristine/WRF/phys/module_sf_mynn.F:1124`, `/home/enric/src/wrf_pristine/WRF/phys/module_sf_mynn.F:1129`).

Fix: if the production WRF comparator uses `sf_sfclay_physics=5`, port `module_sf_mynn.F` for the surface layer, or declare the scheme mismatch as an approximation and validate coupled flux budgets, not just T2/U10/V10/HFX/LH one-step diagnostics.

### P2-8 — RRTMG band "scan barriers" do not prevent the heavy band code from being unrolled

SW `_sw_taumol()` uses a Python loop over 14 bands and builds all branch code before stacking (`src/gpuwrf/physics/rrtmg_sw.py:634`, `src/gpuwrf/physics/rrtmg_sw.py:642`, `src/gpuwrf/physics/rrtmg_sw.py:755`, `src/gpuwrf/physics/rrtmg_sw.py:757`). `_sw_taumol_fused()` then scans only over the already-computed stacked arrays (`src/gpuwrf/physics/rrtmg_sw.py:760`, `src/gpuwrf/physics/rrtmg_sw.py:763`, `src/gpuwrf/physics/rrtmg_sw.py:768`).

LW has the same structure: `_lw_taumol()` unrolls 16 Python-band branches (`src/gpuwrf/physics/rrtmg_lw.py:1271`, `src/gpuwrf/physics/rrtmg_lw.py:1285`, `src/gpuwrf/physics/rrtmg_lw.py:1511`, `src/gpuwrf/physics/rrtmg_lw.py:1514`), and `_lw_taumol_fused()` scans over precomputed results (`src/gpuwrf/physics/rrtmg_lw.py:1517`, `src/gpuwrf/physics/rrtmg_lw.py:1520`, `src/gpuwrf/physics/rrtmg_lw.py:1525`). LW also computes fallback optical depths and then selects branch results with an all-true accepted mask (`src/gpuwrf/physics/rrtmg_lw.py:493`, `src/gpuwrf/physics/rrtmg_lw.py:1943`, `src/gpuwrf/physics/rrtmg_lw.py:1944`, `src/gpuwrf/physics/rrtmg_lw.py:1947`).

Fix: if compile memory is a blocker, move per-band work inside a real `lax.switch`/`lax.scan` over table bundles structured as arrays, or explicitly accept band-unrolled RRTMG and solve compile length at the outer timestep segmentation. The current "scan barrier" comments overstate what the HLO is likely to do.

### P2-9 — Pre-physics guards remain load-bearing in the operational chain

The expected adapter chain in the task puts guards after lateral boundaries (`.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:9`). The actual operational path also applies limiter/mixing guards before physics (`src/gpuwrf/runtime/operational_mode.py:1454`, `src/gpuwrf/runtime/operational_mode.py:1455`, `src/gpuwrf/runtime/operational_mode.py:1457`, `src/gpuwrf/runtime/operational_mode.py:1462`), then applies another guard pass after boundaries (`src/gpuwrf/runtime/operational_mode.py:1499`, `src/gpuwrf/runtime/operational_mode.py:1504`, `src/gpuwrf/runtime/operational_mode.py:1512`).

This may be intentional as a production safety net, but it means physics receives a guarded dycore state, not the raw dycore output. That can hide dycore/physics coupling defects and makes guard-off physics parity less direct.

Fix: keep the production guard if required, but report pre-physics guard counters and run a guard-off or assert-no-op proof for the physics-coupled path before using the result as WRF-faithfulness evidence.

## Part B — Technical Hurdles Plan Review

### Perf plan: segmentation is the right first move, but the named fp32 plan artifact is missing

The requested artifact `proofs/perf/fp32_downcast_plan.md` was not present in this checkout. `test -e proofs/perf/fp32_downcast_plan.md` returned not found, and repository search only found references to future fp32/downcast work, not that plan. Because the review task explicitly points to that path (`.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:15`), this is a planning gap.

The segmentation diagnosis is sound. The current production entrypoint is one `jax.jit` with `hours` static (`src/gpuwrf/runtime/operational_mode.py:1725`, `src/gpuwrf/runtime/operational_mode.py:1726`) and the docstring says the whole forecast lowers as one JAX program (`src/gpuwrf/runtime/operational_mode.py:1727`, `src/gpuwrf/runtime/operational_mode.py:1731`). The internal `_scan_forecast_segment()` uses `lax.scan` (`src/gpuwrf/runtime/operational_mode.py:1556`, `src/gpuwrf/runtime/operational_mode.py:1576`), but because it is called inside the outer jitted forecast loop (`src/gpuwrf/runtime/operational_mode.py:1751`, `src/gpuwrf/runtime/operational_mode.py:1756`, `src/gpuwrf/runtime/operational_mode.py:1764`, `src/gpuwrf/runtime/operational_mode.py:1774`), it still contributes to one traced forecast graph. That explains compile time/memory growing with forecast length.

Recommended close plan:

1. Compile a fixed-length segment function outside the `hours`-static whole-forecast JIT. Loop over segments from Python while carrying JAX device arrays. This does not imply host/device state transfers if the carry remains a device array, but it needs a transfer audit.
2. Use `donate_argnums` for the segment carry and fixed static args for segment length/radiation mode. Avoid recompiles by keeping segment shapes and static arguments constant.
3. Use scan `unroll=1` or explicit unroll control where compile size matters.
4. AOT or persistent compilation cache can help repeat runs, but it does not solve an unbounded graph. Segmentation solves graph length; AOT only amortizes compilation.
5. `custom_vjp` is not relevant unless differentiating this forecast. It will not fix compile-bound inference.

Do not start fp32 downcast until P0-1 is fixed. Right now the couplers already downcast fields silently in the force-fp64 path, so any fp32 evidence would be contaminated. The daily pipeline correctly documents that the real-case path needs fp64 for the acoustic solve because fp32 loses perturbations and detonates (`src/gpuwrf/integration/daily_pipeline.py:168`, `src/gpuwrf/integration/daily_pipeline.py:169`, `src/gpuwrf/integration/daily_pipeline.py:170`). The safe downcast boundary is: keep acoustic small-step, implicit `w/ph` solve, EOS/pressure/geopotential, and mass fields fp64; consider fp32 only for fields with explicit per-field parity/conservation proof.

### Open-top W boundary condition: rigid lid is an interim bypass, not a final WRF-faithful answer

The production real-case path currently forces `top_lid=True` (`src/gpuwrf/integration/daily_pipeline.py:182`, `src/gpuwrf/integration/daily_pipeline.py:188`, `src/gpuwrf/integration/daily_pipeline.py:212`) because open-top creates an immediate model-top spike (`src/gpuwrf/integration/daily_pipeline.py:183`, `src/gpuwrf/integration/daily_pipeline.py:185`). The top-face JAX formula structurally mirrors the WRF snippet: JAX lines compute the pressure/metric/buoyancy terms and then zero when `top_lid` is true (`src/gpuwrf/dynamics/core/advance_w.py:345`, `src/gpuwrf/dynamics/core/advance_w.py:356`, `src/gpuwrf/dynamics/core/advance_w.py:360`, `src/gpuwrf/dynamics/core/advance_w.py:364`, `src/gpuwrf/dynamics/core/advance_w.py:365`); WRF does the analogous top-face update and optional lid zero (`/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1420`, `/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1424`, `/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1427`, `/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1430`).

Because the formula is structurally close, the likely bug is not a simple missing term in `advance_w.py:345`. More likely candidates are:

- top face input consistency: `rhs(nz)`/`ph(nz)` and perturbation/geopotential top boundary values are not the same quantities WRF has at `k_end+1`;
- top coefficient placement: `safe_mass_h_mut[km1]`, `c2a[km1]`, `c1f[nz]`, or `rdnw[km1]` may be built on a mass/half-level convention that matches idealized gates but not real d02 top data;
- lateral-boundary interaction at top corners: the daily-pipeline comment says boundaries re-feed the top-corner mode faster than damping removes it (`src/gpuwrf/integration/daily_pipeline.py:186`, `src/gpuwrf/integration/daily_pipeline.py:187`).

Rigid lid is defensible only as a documented stability bypass if the production WRF comparator also uses a rigid lid or if the project accepts a non-WRF top boundary for near-term demos. For the stated endpoint, it must be fixed or explicitly ADR'd as a deliberate configuration difference. A d02 real forecast with upper-level flow and lateral forcing can show skill sensitivity to the top boundary even when surface metrics look stable at +1h.

### V10 weak spot: most likely surface/PBL momentum-flux coupling, then V-boundary, less likely dry dycore V

The V10 diagnostic is directly proportional to the lowest-level `v0` and the surface similarity ratio in the current port (`src/gpuwrf/physics/surface_layer.py:551`, `src/gpuwrf/physics/surface_layer.py:552`). The same surface layer writes `tau_v` from `v0`, wind speed, and `ustar` (`src/gpuwrf/physics/surface_layer.py:575`, `src/gpuwrf/physics/surface_layer.py:577`), and MYNN applies bottom drag using surface fluxes (`src/gpuwrf/physics/mynn_pbl.py:493`, `src/gpuwrf/physics/mynn_pbl.py:494`, `src/gpuwrf/physics/mynn_pbl.py:495`).

Given that:

- the surface scheme is not the same as the noted WRF oracle (`.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:6`);
- the public driver has an order bug where MYNN can run before surface (`src/gpuwrf/coupling/driver.py:897`, `src/gpuwrf/coupling/driver.py:900`, `src/gpuwrf/coupling/driver.py:902`);
- operational MYNN reconstructs V faces with zero-gradient edge placeholders until boundaries overwrite them (`src/gpuwrf/coupling/physics_couplers.py:308`, `src/gpuwrf/coupling/physics_couplers.py:313`, `src/gpuwrf/coupling/physics_couplers.py:740`);

the most likely cause of a southerly V10 bias is surface/PBL momentum-flux coupling, especially `sfclayrev` versus `sf_mynn`, rather than a core dry-dycore V error. The second candidate is V-component lateral-boundary interaction if the bias is spatially concentrated in south/north relaxation zones. A dycore V issue is lower probability after the recent dry operational-hardening work, but it is not eliminated until a V tendency/savepoint parity check is run in the real terrain/boundary configuration.

### Microphysics risk: high for real Canary 3 km, not low-priority for the final endpoint

The task already states cross-species collection is unported and validation was dry/near-inactive (`.agent/reviews/2026-05-30-gpt-coupler-plan-task.md:5`). WRF rain-snow and rain-graupel collection pathways materially move mass and number (`/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:2483`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:2497`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:2504`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:2522`, `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F:2531`). Add the newly found active rain-freezing table bug and sedimentation approximation, and moist/cloudy 3 km Canary cases are not safely covered.

Priority judgment: microphysics can be staged behind dycore/top-boundary and surface/PBL fixes if the immediate target cases are dry/wind-dominant. It cannot be called low-priority for the stated endpoint, because island cloud/precip and radiative/PBL feedbacks are part of real surface forecast skill.

## Highest-Risk Technical Item

The highest-risk remaining item is still precision/coupling correctness, specifically P0-1. The code currently claims a force-fp64 operational path while silently downcasting major physics-coupled fields inside the timestep. That contaminates physics parity, any fp32 downcast plan, V10 diagnosis, and radiation/microphysics error attribution.

The overall technical close plan is directionally sound only after three gaps are closed:

- fix coupler dtype preservation;
- implement held radiation tendencies instead of lumped cadence heating;
- segment the forecast as reusable fixed-length compiled calls before fp32 work.

## Handoff

Objective: perform the requested independent adversarial review of physics couplers and remaining hard technical hurdles.

Files changed: `.agent/reviews/2026-05-30-gpt-coupler-and-plan-review.md` only.

Commands run: repo instruction/context reads; targeted `rg` and `nl -ba` inspections of `src/gpuwrf/coupling/physics_couplers.py`, `src/gpuwrf/coupling/driver.py`, `src/gpuwrf/runtime/operational_mode.py`, Thompson, surface, MYNN, RRTMG, `advance_w.py`, WRF radiation/Thompson/surface/small-step references, and `git status --short`.

Proof objects produced: this review artifact only; no validation or publishing artifacts by request.

Unresolved risks: no runtime validation was executed; no HLO/profile artifacts were generated; `proofs/perf/fp32_downcast_plan.md` was absent, so the fp32 critique is based on code and available references rather than the named plan document.

Next decision needed: choose whether to fix P0-1/P0-2 before any further validation/perf push. My recommendation is yes; otherwise subsequent proof objects can be false-green.

GPT_COUPLER_PLAN_REVIEW_COMPLETE

Top-3 must-fix technical items:

1. Fix physics-coupler dtype preservation so `force_fp64=True` remains fp64 through Thompson, surface/MYNN, and RRTMG, not just at timestep exit.
2. Replace lumped radiation-cadence heating with resident held radiation tendencies applied every dynamics step.
3. Fix or formally ADR the open-top `w` boundary condition; rigid lid is only an interim bypass unless the target WRF configuration also uses it.
