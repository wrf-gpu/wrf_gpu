# ADR-013 - M6 Noah-MP Prescribed Subset

Date: 2026-05-21
Author: M6-S3 worker (Codex gpt-5.5)
Status: PROPOSED worker draft
Scope: M6-S3 Noah-MP minimum needed to feed the real surface-layer path.

## Decision

M6-S3 implements a **prescribed Noah-MP subset**, not a prognostic land-surface model. The subset loads and validates the Noah-MP state fields already present in Gen2 `wrfinput_d02`, then exposes bounded surface inputs to the sfclay kernel:

- `TSK` as skin temperature.
- `SMOIS` as total soil moisture and `SH2O` as liquid soil moisture.
- `TSLB` as soil temperature.
- `IVGTYP`, `ISLTYP`, `LU_INDEX`, `XLAND`, `LANDMASK`, and `LAKEMASK` as category/mask fields.
- `SST` for water points when available.
- `CM` and `CH` as prescribed surface exchange-coefficient provenance; `CM` is inverted to a bounded roughness surrogate because direct `ZNT` is absent in the local `wrfinput_d02`.

This subset intentionally does not call WRF `NOAHMP_SFLX` or evolve Noah-MP state.

## WRF Source Mapping

The local WRF Noah-MP driver at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_sf_noahmpdrv.F` treats `TSK`, `SMOIS`, `SH2O`, and `TSLB` as inout Noah-MP state fields (`:224`, `:235-237`). It reads vegetation and soil category fields `IVGTYP` and `ISLTYP` at `:106-107`.

In the active land-tile path, the driver copies prescribed grid state into tile-local variables:

- `VEGTYP = IVGTYP(I,J)` and `SOILTYP = ISLTYP(I,J)` at `module_sf_noahmpdrv.F:1241-1252`.
- `SMC`, `SMH2O`, and `STC` receive `SMOIS`, `SH2O`, and `TSLB` at `module_sf_noahmpdrv.F:1626-1629` and again in the stock path at `:2202-2205`.
- After a full Noah-MP call, prognostic results are copied back to `TSK`, `SMOIS`, `SH2O`, and `TSLB` at `module_sf_noahmpdrv.F:2921-2935`.

M6-S3 stops before the prognostic call and uses the copied Gen2 fields as prescribed lower-boundary inputs. This matches Option A from the M6-S3 dispatch and keeps the operational path bounded.

## Boundaries

The prescribed subset may be time varying only if real d02 history is available through Gen2. In the pinned local tree there are no `wrfout_d02_*` files, so the M6-S3 implementation uses time-zero `wrfinput_d02` land state and records this in the land-state manifest.

The subset may not modify `/mnt/data/canairy_meteo/**`; all generated manifests and proof artifacts live under this repository.

## Consequences

Positive: surface-layer diagnostics become stability-aware and tied to real Gen2 lower-boundary fields while avoiding a false claim of Noah-MP parity.

Risk: this is not a prognostic land feedback. If M6-S8 surface RMSE still fails after radiation and dycore follow-ups, the next decision is whether to promote Option B bounded prognostic Noah-MP in M6.5/M7.
