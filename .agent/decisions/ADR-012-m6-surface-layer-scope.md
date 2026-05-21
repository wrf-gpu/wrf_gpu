# ADR-012 - M6 Surface-Layer Scope

Date: 2026-05-21
Author: M6-S3 worker (Codex gpt-5.5)
Status: PROPOSED worker draft
Scope: M6-S3 real surface-layer fold-in for operational `U10/V10/T2/Q2` diagnostics and MYNN surface-flux handles.

## Decision

M6-S3 implements the manager-selected **Option A: prescribed land state**. The atmosphere reads the Gen2 d02 land/surface state and uses it as a prescribed lower boundary for an MM5 sfclay-style Monin-Obukhov surface-layer kernel. M6-S3 does not add prognostic soil, canopy, snow, or irrigation evolution.

Included WRF features:

- MM5 `module_sf_sfclay.F` surface-layer similarity theory for friction velocity, stability functions, sensible/moisture flux exchange, and 10 m / 2 m diagnostics.
- Lowest-model-layer mass-point winds, temperature, water vapor, and pressure as atmospheric inputs.
- Prescribed skin temperature, surface pressure, land/water mask, soil moisture availability, and roughness surrogate from Gen2 `wrfinput_d02`.
- Output `SurfaceFluxes(ustar, theta_flux, qv_flux, tau_u, tau_v, rhosfc, fltv)` in FP64 at the adapter boundary, preserving the M5-S2.x surface-layer contract.

Excluded WRF features:

- Prognostic Noah-MP soil moisture, soil temperature, canopy water, snowpack, crop, irrigation, runoff, groundwater, and 4D-Var coupling.
- Online sea-surface roughness evolution beyond the sfclay Charnock update in the local kernel.
- RRTMG online radiation conditioning inside M6-S3; the pinned d02 Gen2 tree does not expose `wrfout_d02_*` history files or `RTHRATEN/RTHRATSW/RTHRATLW`.

## Prescribed Data Source

The source is the pinned Gen2 run:

`/mnt/data/canairy_meteo/runs/wrf_l3/20260519_18z_l3_24h_20260520T025228Z/wrfinput_d02`

The required prescribed-land inventory is:

- Land/category: `XLAND`, `LANDMASK`, `LAKEMASK`, `IVGTYP`, `ISLTYP`, `LU_INDEX`.
- Thermal state: `TSK`, `SST`, `TSLB`, plus Noah-MP diagnostics `TG`, `TV`, and `TRAD` when present.
- Moisture state: `SMOIS`, `SH2O`, `Q2`, and Noah-MP canopy/ground diagnostics when present.
- Surface exchange/roughness surrogate: `CM` and `CH`.

Local inventory caveat: `wrfinput_d02` does **not** contain `ZNT`. M6-S3 therefore derives a bounded roughness surrogate from the prescribed exchange coefficient `CM` and records that derivation in the land-state manifest. This is the smallest honest path for Option A in the local data tree; it is not a claim that direct WRF `ZNT` history was available.

## WRF Source Mapping

The local WRF source is `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF`.

- The sfclay public wrapper passes lowest-level `U/V/T/QV/P`, `dz8w`, `PSFC`, `ZNT`, `MAVAIL`, `XLAND`, `TSK`, and diagnostics through to `SFCLAY1D` at `phys/module_sf_sfclay.F:234-255`.
- `SFCLAY1D` declares the lower-boundary inputs and diagnostics at `phys/module_sf_sfclay.F:263-334`.
- Ground potential temperature is computed from `TSK` and surface pressure at `phys/module_sf_sfclay.F:399-406`.
- Air potential temperature, virtual temperature, saturated surface mixing ratio, density, and first-level height are computed at `phys/module_sf_sfclay.F:428-483`.
- Bulk Richardson number and convective/subgrid wind augmentation are computed at `phys/module_sf_sfclay.F:490-535`.
- Stable, neutral, and unstable Monin-Obukhov regimes are selected at `phys/module_sf_sfclay.F:563-695`, with unstable stability-function tables initialized at `phys/module_sf_sfclay.F:954-967`.
- Friction velocity and 10 m / 2 m diagnostics are computed at `phys/module_sf_sfclay.F:697-825`.
- Surface moisture and heat flux coefficients/fluxes are computed at `phys/module_sf_sfclay.F:830-936`.

## Consequences

Positive: M6-S3 replaces the M5 neutral-bulk placeholder at the coupled adapter boundary with a real stability-aware surface layer and prescribed Gen2 land state, without expanding to a full Noah-MP port.

Risk: without `wrfout_d02_*` history, prescribed land state is time-zero only and d02 radiation tendencies are unavailable. Operational deltas from this sprint are therefore a surface-layer fold-in signal, not proof of fully conditioned land/radiation coupling.
