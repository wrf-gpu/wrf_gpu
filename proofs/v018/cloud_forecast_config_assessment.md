# Cloud Forecast Configuration Assessment - v018

Date: 2026-06-17
Author: Codex / GPT-5.5 verifier
Scope: read-only scoping analysis. No code/config changes, no WRF/GPU jobs, no training jobs.

## Executive Verdict

**Do not spend days or weeks of GPU compute mass-producing a cloud downscaler database from the current surface/hourly training contract. It is not cloud-formation-ready as-is.**

This is not because the Canary WRF physics template is empty. The current L2/L3 namelists and retained raw WRF files do compute the main cloud ingredients: Thompson hydrometeors, `CLDFRA`, water vapor, thermodynamic state, geopotential height, PBL/radiation/surface fields, and 1 km island nests. The current `wrf_gpu2` writer also knows how to emit a richer 3-D cloud-capable subset than the active downscaler cache.

The blocker is the **production data contract**:

- The active `wrf_downscale` hot cache is surface-only and keeps `TCC` as the only cloud channel.
- The cache drops the actual 3-D cloud/moist state needed for formation mechanisms: `CLDFRA`, `QCLOUD/QRAIN/QICE/QSNOW/QGRAUP`, `QVAPOR`, `T/P/PB/PH/PHB`, and number concentrations.
- Current retained products are hourly. That is too coarse for cloud onset, breakup, and terrain-slope transitions; MSG cloud truth is 15 min.
- `e_vert=45` gives only about **two median mass levels between 800 and 1500 m AGL** over the sampled Tenerife/La Palma/3 km domains. That is weak for trade-inversion base/top and mar-de-nubes cloud-edge labels.
- The GPU writer's `CLDFRA` path must be audited: radiation can use MYNN `cldfra_bl`, but the main `wrfout` writer falls back to a hydrometeor-occupancy proxy unless a `CLDFRA/cldfra/cloud_fraction` source is present.
- No derived cloud products required by the product layer are frozen into the training contract: cloud base/top, LWP/IWP/RWP, low/mid/high cloud, inversion base/strength, LTS/EIS, LCL/LFC, RH layers, station-relative below/in/above cloud.

**Decision:** re-spec the cloud label/output contract and run one audited pilot case before bulk label production. If the principal only wants surface wind/T2 downscaling, the current cache is acceptable as a surface baseline. If the target is local/hyperlocal cloud formation over Tenerife/La Palma, it is not.

## Evidence Read

Local instructions and sprint constraints:

- `/tmp/gpt_cloud_forecast_brief.txt` was read in full.
- Repository governance read: `PROJECT_CONSTITUTION.md`, `AGENTS.md`, local `.agent/skills/reporting-to-human`, and `.agent/skills/validating-physics`.
- No current v018 sprint contract for this specific cloud assessment was found; the user brief is the operative contract.
- The old global `wrf-gpu-port` skill was deliberately not used, per repo-local instructions.

Primary project context:

- `wrf_downscale` says the long-term target is a practical 1 km downscaler that handles "wind and cloud first" and later T2/RH/gust/precip/moist 3D, while raw WRF NetCDF remains source of truth: `<USER_HOME>/src/wrf_downscale/README.md:3-5`.
- Its initial cache roadmap includes a future native-eta 3-D core and separate moist/cloud/rain cache with hydrometeors, number concentrations, `CLDFRA`, rain increments, masks, and event weights: `<USER_HOME>/src/wrf_downscale/README.md:61-67`, `<USER_HOME>/src/wrf_downscale/docs/cache_design.md:33-53`.
- Gen2's product need is Tenerife first, all Canaries later, with explicit cloud cover / solar branches and terrain-robust local forecasts: `<USER_HOME>/src/canairy_meteo/Gen2/README.md:42-70`.
- Gen2's cloud branch is truth-first: MSG CLM labels at 15 min, HRSEVIRI features at 15 min, MTG FCI around 10 min, and WRF cloud/radiation as future teacher/comparison rather than truth: `<USER_HOME>/src/canairy_meteo/Gen2/README.md:150-158`, `<USER_HOME>/src/canairy_meteo/Gen2/subprojects/cloud_solar_ml_expert/README.md:1-5`, `<USER_HOME>/src/canairy_meteo/Gen2/subprojects/cloud_solar_ml_expert/README.md:50-78`.

## Current Canary WRF Configuration

Representative retained L2 namelist:

- L2 run: 72 h, `max_dom=2`, 9 km / 3 km, `e_vert=45`, hourly history, one frame per file: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260530_18z_l2_72h_20260531T161057Z/namelist.input:1-29`, `:31-54`.
- Physics: Thompson `mp_physics=8`, MYNN `bl_pbl_physics=5`, MYNN surface layer `sf_sfclay_physics=5`, Noah-MP `sf_surface_physics=4`, RRTMG `ra_lw/ra_sw=4`, cumulus `1,0`, `radt=30`, `topo_shading=1`, `slope_rad=1`, `icloud=1`: same file `:56-75`.

Representative retained L3 namelist:

- L3 run: 24 h, `max_dom=5`, 9 km / 3 km / 1 km island nests, `e_we=94,160,94,70,70`, `e_sn=60,67,76,61,58`, `e_vert=45`, hourly history: `<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z/namelist.input:1-54`.
- Physics same as L2, with cumulus on d01 only and off on d02-d05: same file `:56-75`.

Implications:

- `cu_physics=1,0,0,0,0` is the correct broad pattern for this setup: d01 at 9 km has cumulus, 3 km and 1 km nests do not.
- `icloud=1` is explicit in CPU-WRF namelists.
- `radt=30` is defensible for surface production, but likely too coarse for cloud/radiation transition labels over 1 km terrain.

## What The Outputs Actually Contain

I inspected representative retained NetCDF products without running WRF.

Raw CPU-WRF L2/L3 `wrfout` examples:

- L2 d02 raw example: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260521_18z_l2_72h_20260522T133443Z/wrfout_d02_2026-05-22_12:00:00`
- L3 d03 Tenerife raw example: `<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260428_18z_l3_24h_distinct_surface_gate_20260607T003529Z/wrfout_d03_2026-04-29_12:00:00`
- L3 d05 La Palma raw example: `<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260428_18z_l3_24h_distinct_surface_gate_20260607T003529Z/wrfout_d05_2026-04-29_12:00:00`

Observed in raw CPU-WRF files:

- Present: `U,V,W,T,THM,QVAPOR,P,PB,PH,PHB,QCLOUD,QICE,QRAIN,QSNOW,QGRAUP,QNICE,QNRAIN,CLDFRA,T2,Q2,PSFC,PBLH,U10,V10,TSK,SWDOWN,GLW`, RRTMG all-sky fluxes, some clear-sky fluxes, and rain accumulations.
- Missing from raw `wrfout`: `RH/RH2`, `TCC`, low/mid/high cloud cover under the names checked, cloud base/top, LWP/IWP/RWP, LCL/LFC, inversion diagnostics, LTS/EIS.
- Example d03 at 2026-04-29 12Z had nonzero `QCLOUD`, `QRAIN`, `CLDFRA`, and `QNRAIN`; d05 had smaller but nonzero cloud fields.

Thin-gridded sidecar example:

- `<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260429_18z_l3_24h_20260524T204451Z/thin_gridded_d03_tnf_v1.nc`
- Contains 25 hourly times and surface/layer products: `U10,V10,T2,Q2,TSK,PSFC,PBLH,UST,RAINNC,RAINC,RAINSH,SWDOWN,GLW,TCC,CLDLOW,CLDMID,CLDHIGH`.
- Does **not** contain the 3-D state or hydrometeor columns.

Current GPU writer / validation output:

- `wrfout_writer.py` declares downstream critical variables including `CLDFRA`, `QCLOUD`, `QICE`, `QRAIN`, and the 3-D dynamical/thermodynamic state: `.wt-v018-integration/src/gpuwrf/io/wrfout_writer.py:33-80`.
- It also declares extra hydrometeors and number concentrations such as `QSNOW`, `QGRAUP`, `QNICE`, `QNRAIN`, `QNSNOW`, `QNGRAUPEL`, `QNCLOUD`, `QNCCN`, and optional mp28 aerosol fields `QNWFA/QNIFA`: `.wt-v018-integration/src/gpuwrf/io/wrfout_writer.py:92-115`.
- It writes the prepared operational field set when a real source exists: `.wt-v018-integration/src/gpuwrf/io/wrfout_writer.py:283-300`, `:1122-1133`.
- A representative v015 GPU output file contains 107 variables and includes the 3-D state, hydrometeors, `CLDFRA`, `QNSNOW/QNGRAUPEL/QNCLOUD/QNCCN`, all-sky radiation fluxes, and surface fields. It lacks `TCC`, low/mid/high cloud cover, cloud base/top, LCL/LFC, `QNWFA/QNIFA` under mp8, and clear-sky `...C` fluxes.

Important GPU `CLDFRA` caveat:

- The MYNN SGS cloud chain is default-on and documents `icloud_bl=1`: `.wt-v018-integration/src/gpuwrf/physics/mynn_sgs_cloud.py:1-13`, `:40-52`.
- RRTMG radiation can use `state.cldfra_bl` after the first step when SGS clouds are enabled: `.wt-v018-integration/src/gpuwrf/coupling/physics_couplers.py:1016-1034`.
- But the main `wrfout` writer's `CLDFRA` field is sourced only from `CLDFRA/cldfra/cloud_fraction` if present; otherwise it falls back to `where((qc+qi+qr)>1e-8,1,0)`: `.wt-v018-integration/src/gpuwrf/io/wrfout_writer.py:1432-1440`.
- The normal M9 diagnostics routed into output are surface/radiation fields only; no `CLDFRA` or `cldfra_bl` is routed there: `.wt-v018-integration/src/gpuwrf/runtime/operational_mode.py:4315-4350`, `.wt-v018-integration/src/gpuwrf/integration/daily_pipeline.py:809-837`.

Conclusion: CPU raw WRF `CLDFRA` is a valid cloud-fraction history field. Current GPU output `CLDFRA` is not yet proven to be the same diagnostic as WRF `CLDFRA` or as the cloud fraction radiation used. This must be audited and fixed before GPU cloud-label production.

## Downscaler Fit

The actual implemented `wrf_downscale` hot cache is not the same as the future cloud design:

- Manifest/core variables are only `U10,V10,T2,Q2,TSK,PSFC,GLW,SWDOWN,RAINC,RAINNC,RAINSH,TCC`: `<USER_HOME>/src/wrf_downscale/src/wrf_downscale/io/thin_gridded.py:24-37`.
- The cache builder defaults to that exact variable list: `<USER_HOME>/src/wrf_downscale/src/wrf_downscale/cache/thin_gridded_surface.py:48-55`.
- The produced cache has 28 current-shape cases, 2100 samples, and those 12 channels only: `<USER_HOME>/src/wrf_downscale/reports/phase2c_thin_gridded_hot_cache_v1.md:19-32`.
- The common-variable manifest confirms the same safe surface intersection: `<USER_HOME>/src/wrf_downscale/reports/phase2a_thin_gridded_manifest_v1.md:21-24`.

This means the current production cache cannot train a local cloud-formation model. It can train a surface residual/downscaler with a `TCC` proxy. It cannot teach:

- vertical saturation and cloud onset;
- cloud base/top relative to Tenerife/La Palma terrain;
- trade-inversion height/strength;
- cloud water path and drizzle/mizzle;
- station-relative below/in/above cloud state;
- radiation-cloud feedback timing;
- whether WRF cloud fraction came from resolved hydrometeors, MYNN subgrid cloud, or a derived proxy.

The downscale design docs already know this: they list future 3-D and moist/cloud/rain caches separately: `<USER_HOME>/src/wrf_downscale/docs/cache_design.md:33-53`.

## Vertical Resolution

I computed mass-level heights from `PH+PHB` and `HGT` in representative raw `wrfout` files.

Median AGL mass levels below 2.5 km:

| Domain | Terrain max | Median mass levels near low cloud layer |
| --- | ---: | --- |
| L2 d02 3 km | 2987 m | 25.7, 84.3, 159.0, 253.7, 373.1, 522.5, 708.2, 936.3, 1212.3, 1540.6, 1923.7, 2361.5 m |
| L3 d03 Tenerife 1 km | 3462 m | 25.5, 83.5, 157.4, 251.1, 369.0, 516.2, 698.2, 920.4, 1188.3, 1506.3, 1877.1, 2301.3 m |
| L3 d05 La Palma 1 km | 2272 m | 25.4, 83.2, 156.9, 250.3, 367.9, 514.7, 696.0, 917.3, 1184.0, 1501.2, 1871.2, 2294.2 m |

Count of median mass levels in the 800-1500 m trade-inversion band: **2** for each sampled domain.

Implication:

- `e_vert=45` is sufficient to provide broad low-cloud context and can support coarse cloud-base/top derivations.
- It is not strong for sharp inversion-base/top, thin cloud layer thickness, or mar-de-nubes edge formation. Through 800-1500 m, typical median spacing is about 220-320 m between mass levels.
- If the product claim is "local/hyperlocal cloud formation" rather than a broad cloud-cover feature, the mass-production grid should test a refined low-level eta design, likely 60-70 vertical levels or a custom eta distribution with more resolution between 500 and 2000 m MSL/AGL. If compute budget blocks that, the dataset must label inversion/cloud-base uncertainty explicitly.

## Horizontal Resolution

3 km parent:

- Useful as a predictor and large-scale moisture/PBL source.
- Not sufficient as a hyperlocal cloud label for Tenerife/La Palma orographic cloud. Coast/ridge/valley terrain transitions are too sharp.

1 km nests:

- Current L3 d03 covers Tenerife + La Gomera; d05 covers La Palma + north El Hierro. This is the minimum usable WRF label resolution for the requested Tenerife/La Palma problem.
- Even 1 km is still not fully hyperlocal for valley-vs-ridge cloud because grid-cell terrain can differ from station altitude. The Gen2 cloud-cap report explicitly says subgrid orographic cloud requires L3 1 km plus honest station/grid terrain treatment: `<USER_HOME>/src/canairy_meteo/Gen2/reports/opus_cloud_cap_quantification_v1.md:95-106`.

All-islands label product:

- `wrf_downscale` recommends a future contiguous 1 km all-Canaries grid paired with a 3 km parent: `<USER_HOME>/src/wrf_downscale/reports/wrf_gpu2_training_grid_design_20260607.md:3-24`, `:71-83`.
- Later feasibility work says `AC1_FIT` at 520 x 278 mass points is the measured-fit all-islands 1 km target, but with thin headroom and a required full-physics from-IC check before mass production: `<USER_HOME>/src/wrf_downscale/reports/final_canary_training_grid_20260614.md:12-25`, `:176-210`.
- The existing operational 1 km nests do not cover all islands; Fuerteventura/Lanzarote have no 1 km labels in the fallback layout: same report `:49-61`, `:212-228`.

For this user request, Tenerife/La Palma can use current d03/d05 geometry as a first fallback label domain, but a 3 km-only database is wrong for the hyperlocal cloud objective.

## Cadence

Current retained L2/L3 namelists use `history_interval=60` and `frames_per_outfile=1`: L2 namelist `:20-22`, L3 namelist `:20-22`.

Cloud truth cadence in Gen2:

- MSG CLM truth labels are 15 min.
- HRSEVIRI feature input is 15 min.
- MTG FCI is around 10 min.
- Source: `<USER_HOME>/src/canairy_meteo/Gen2/README.md:150-158`.

Assessment:

- Hourly fields can support broad cloud-cover and daily regime labels.
- Hourly fields are weak for onset, breakup, upslope burn-off, and cloud-edge training. A cloud bank can form or dissolve between hourly outputs.
- `wrf_gpu2` has an optional auxhist mechanism capable of genuine 15 min subset outputs, off by default: `.wt-v018-integration/src/gpuwrf/io/auxhist_stream.py:1-27`, `.wt-v018-integration/src/gpuwrf/integration/daily_pipeline.py:129-153`, `:721-733`, `:1084-1094`, `:1150-1158`.
- Tests pin down 15 min auxhist outputs as real sub-hour snapshots, not interpolation: `.wt-v018-integration/tests/test_auxhist_stream.py:1-25`, `:239-255`.

Recommendation:

- Keep the heavy full-state `wrfout` hourly if storage is tight.
- Add a 15 or 30 min cloud auxhist/output product for derived cloud diagnostics: `TCC/CLDLOW/CLDMID/CLDHIGH`, cloud base/top, LWP/IWP/RWP, inversion metrics, PBLH, SWDOWN/GLW, HFX/LH, T2/Q2/TSK, and compact provenance.
- If the model is meant to learn formation dynamics, also retain enough 3-D state at 30 min, at least for pilot cases and high-value cloud regimes.

## Physics Assessment

Current physics template:

- Thompson mp8: good operational default; raw outputs include hydrometeors and `QNICE/QNRAIN`.
- MYNN `bl=5` + MYNN surface layer: the right broad family for marine PBL/orographic cloud, but known to need careful cloud/PBL tuning for this island problem.
- RRTMG SW/LW with `topo_shading=1` and `slope_rad=1`: necessary for terrain cloud/radiation/thermal contrasts.
- Noah-MP `sf_surface=4`: needed for land heating and slope/circulation coupling.
- Cumulus d01 only: defensible. Do not enable cumulus on 1 km nests.

Risks and tuning implications:

- Current `radt=30` is coarse for 1 km cloud/radiation transitions. Gen2's physics second opinion recommends `radt=10` or `9` because terrain shading and cloud-cap radiation evolve fast over 1 km islands and can alias morning/evening transitions: `<USER_HOME>/src/canairy_meteo/Gen2/reports/codex_physics_tuning_second_opinion_v1.md:35-36`, `:55-57`, `:112-129`.
- MYNN cloud internals matter. The same report identifies `bl_mynn_cloudpdf`/`icloud_bl` as directly targeting subgrid low-cloud fraction under the trade inversion and recommends monitoring cloud/T2/RH and north/east cloud-cap slices: same file `:40-43`, `:83-99`, `:121-129`.
- The current GPU port implements MYNN SGS cloud chain default-on, but output of actual cloud fraction needs the CLDFRA audit described above.
- `mp_physics=28` aerosol-aware Thompson exists in `wrf_gpu2`, with `QNWFA/QNIFA` prognostics and cold-start self-init support: `.wt-v018-integration/src/gpuwrf/coupling/physics_dispatch.py:168-172`, `.wt-v018-integration/src/gpuwrf/runtime/operational_mode.py:3367-3374`, `:3715-3722`, and scheme limits at `.wt-v018-integration/src/gpuwrf/io/scheme_catalog.py:1118-1148`.
- However, current Canary L2/L3 namelists use mp8, not mp28. Do not switch to mp28 for bulk production without a cloud/precip/radiation pilot, because mp28 changes the physics and adds aerosol assumptions. It is a later sensitivity, not a blocker for first WRF cloud labels.

Gen2 already treats WRF cloud/radiation as a future teacher, not truth:

- `<USER_HOME>/src/canairy_meteo/Gen2/subprojects/cloud_solar_ml_expert/README.md:50-78`, `:150-170`, `:212-216`.

Therefore, the label database must preserve source flags and enable comparison to MSG CLM, not train as if WRF cloud is observational truth.

## Required Cloud Label Contract Before Bulk Compute

Minimum fields to retain per output time for cloud-formation training:

3-D state on native eta:

- `QVAPOR`
- `T` or dry/moist theta plus actual temperature derivation metadata
- `P`, `PB`
- `PH`, `PHB`
- `U`, `V`, `W`
- `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, `QGRAUP`
- `CLDFRA` with source semantics frozen and audited
- `QNICE`, `QNRAIN`; plus `QNCLOUD/QNCCN/QNSNOW/QNGRAUPEL` only where physically active/non-fabricated
- `QKE` / turbulence state where available

2-D surface/PBL/radiation:

- `U10`, `V10`, `T2`, `Q2`, `PSFC`, `TSK`
- `PBLH`, `UST`, `HFX`, `LH`, `QFX` if available
- `SWDOWN`, `GLW`, `SWDNB/LWDNB`, `SWNORM`, `OLR`
- `RAINC`, `RAINNC`, `RAINSH` and hourly/subhour increments
- static terrain, landmask, land use, slope, aspect, curvature, coast distance, map factors

Derived cloud products:

- total, low, mid, high cloud fraction from the same `CLDFRA` convention
- cloud base/top in MSL and AGL
- cloud thickness
- column LWP/IWP/RWP
- peak `CLDFRA`
- station-relative class: clear / below cloud / in cloud / above cloud / partial
- inversion base/top/strength and capping-inversion height
- LTS/EIS
- LCL/LFC/CAPE/CIN if used downstream
- RH profiles and layer RH summaries

The Gen2 cloud-cap quantification report already defines a practical subset and thresholds using `CLDFRA(k)`, `QCLOUD(k)`, `QICE/QRAIN`, `T/P/PB`, `PH/PHB`, `HGT`, `XLAT/XLONG`, and `SWDOWN`: `<USER_HOME>/src/canairy_meteo/Gen2/reports/opus_cloud_cap_quantification_v1.md:22-60`, `:76-93`, `:235-256`.

## Must-Have Changes Before Mass Production

1. **Freeze a cloud data contract.**
   Do this before any long GPU campaign. The active 12-channel surface cache is not the cloud contract.

2. **Fix/audit GPU `CLDFRA`.**
   Output `CLDFRA` must be the same diagnostic intended for cloud labels, not a fallback hydrometeor occupancy mask. If radiation uses MYNN `cldfra_bl`, either emit `CLDFRA_BL/QC_BL/QI_BL` or route the exact WRF-equivalent cloud fraction into `CLDFRA`, then compare CPU-WRF vs GPU on a cloud-cap fixture.

3. **Use 1 km child labels for Tenerife/La Palma.**
   3 km is acceptable as parent predictor, not as hyperlocal cloud target. Use current d03/d05 for a first Tenerife/La Palma pilot, or AC1_FIT after the full-physics fit check.

4. **Add 15-30 min cloud output cadence.**
   Prefer 15 min for MSG alignment, 30 min if storage is constrained. Hourly-only is a broad-regime product, not formation timing.

5. **Add derived cloud/inversion diagnostics.**
   Do not force every trainer to rediscover cloud base/top/LWP/inversion from raw 3-D arrays without a canonical audited derivation.

6. **Improve or explicitly qualify vertical resolution.**
   Either test a denser low-level eta grid before bulk production, or mark inversion/cloud-base products as coarse with only two levels in 800-1500 m.

7. **Retain raw/restart-compatible artifacts until cache audit passes.**
   Current raw WRF is often removed after thin sidecars. That is fine for surface work, but not for first cloud database construction.

8. **Run one cloud-regime pilot before weeks of compute.**
   Use a known cap-nubes / trade-inversion case. Produce CPU/GPU field inventory, CLDFRA audit, derived diagnostics, MSG CLM alignment, and cache-loader smoke before scaling.

## Nice-To-Have / Later Sensitivities

- `radt=9-10` min instead of 30 min for cloud/radiation transition cases.
- `bl_mynn_mixlength=2` and `bl_mynn_cloudpdf` sensitivity after a baseline cloud scorer exists.
- mp28 aerosol-aware Thompson only after a pilot, especially for calima/dust/cloud-radiation cases.
- CAMS AOD / aerosol radiation coupling for calima.
- Additional 1 km nests or AC1 all-islands for Fuerteventura/Lanzarote if all-islands cloud product matters.
- Probabilistic heads using MSG CLM as truth and WRF as teacher/feature, not truth.

## Cost Implications

Output/storage scaling:

- AC1_FIT is about 520 x 278 = 144,560 columns.
- One float32 3-D mass variable at 44 levels is about **24.3 MiB per time**.
- One float32 2-D variable is about **0.55 MiB per time**.
- A core cloud subset with 10 3-D variables and 20 2-D variables is about **254 MiB per output time**, or **7.4 GiB for 30 hourly outputs**, or **29.7 GiB for 30 h at 15 min** before compression/chunking.
- A richer 18 3-D + 30 2-D subset is about **453 MiB per output time**, or **13.3 GiB hourly / 53.1 GiB at 15 min** for 30 h before compression/chunking.

Compute scaling:

- Existing all-islands feasibility work projects AC1_FIT 24 h around 4.5 h and 30 h around 5.6 h forecast-only, fp64, with thin full-physics headroom caveats: `<USER_HOME>/src/wrf_downscale/reports/final_canary_training_grid_20260614.md:176-210`.
- Lowering radiation cadence from 30 min to 9-10 min adds physics cost; prior Gen2 estimate for CPU sensitivity was +10-25% for the run: `<USER_HOME>/src/canairy_meteo/Gen2/reports/codex_physics_tuning_second_opinion_v1.md:35-36`.
- Increasing vertical levels from 44 mass levels to roughly 60-70 mass levels will scale memory/I/O and much column physics approximately with the vertical count, likely +35-60% for vertical-array components.
- Subhourly output mostly multiplies I/O/storage and output-boundary work. Use compact auxhist/diagnostic products to avoid writing the entire 3-D state every 15 min.

The expensive mistake is not output volume. The expensive mistake is generating many 1 km days with only hourly/surface/TCC labels and no way to recover the cloud formation state later.

## Final Recommendation

**Re-spec before compute.**

Recommended immediate sequence:

1. Freeze `cloud_label_contract_v1`: variables, cadence, derivations, provenance, and CLDFRA semantics.
2. Implement/verify a compact cloud auxhist/derived output stream and a raw 3-D retention policy for pilot cases.
3. Run one Tenerife/La Palma cloud-cap pilot only after the contract exists.
4. Audit against raw CPU-WRF, GPU output, and MSG CLM/station solar where available.
5. Only then start bulk GPU label production.

If the principal proceeds with the current cache/config unchanged, the result will be a useful surface downscaler corpus, but not a defensible local/hyperlocal cloud-formation training database.

## Commands / Proof Objects Produced

No WRF, GPU forecast, training, or code-generation job was run.

Read/inspection commands included:

- `nl -ba` on retained L2/L3 `namelist.input` files.
- `rg` over `.wt-v018-integration/src`, `.wt-v018-integration/docs`, `wrf_downscale`, and Gen2 reports.
- Python `netCDF4.Dataset` read-only inspection of representative raw CPU-WRF, thin-gridded, and GPU validation `wrfout` files.
- Python read-only computation of mass-level AGL heights from `PH+PHB` and `HGT`.
- Python read-only inspection of `wrf_downscale` cache summaries/manifests.

Proof object produced:

- `.wt-v018-integration/proofs/v018/cloud_forecast_config_assessment.md`

Unresolved risks:

- No new WRF/GPU pilot was run, by instruction. Therefore this report identifies configuration/data-contract risk; it does not close a cloud-skill gate.
- The GPU `CLDFRA` semantics are code-inferred and require a direct CPU/GPU cloudy-case audit before production.
- Vertical-level recommendations are based on sampled retained runs and domain medians; a custom eta-design sprint should quantify exact inversion-layer levels over target stations and terrain bands.

Next decision needed:

- Approve a cloud-label contract/pilot sprint before bulk GPU database generation, or explicitly limit the next compute campaign to surface downscaling rather than cloud formation.
