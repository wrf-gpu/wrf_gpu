# ADR-009 - RRTMG Real-Driver Column Binding And Reduced-G-Point JAX Tables

Date: 2026-05-21
Author: M5-S3 attempt-3 worker amendment (Codex gpt-5.5)
Status: PROPOSED worker draft, pending mandatory Claude Opus reviewer pass
Scope: implementation record for the M5-S3 RRTMG shortwave and longwave radiation column sprint.

## Decision

M5-S3 attempt 3 preserves the attempt-2 WRF wrapper-driver binding: the Fortran harness initializes local Gen2 RRTMG tables and calls the real WRF `RRTMG_SWRAD` and `RRTMG_LWRAD` wrapper surfaces. The JAX table asset still stores raw `RRTMG_SW_DATA` and `RRTMG_LW_DATA` payload bytes for provenance, but the compact A2 median/quantile clipped reductions are removed.

The table extractor now consumes WRF spectral data using WRF's own reduced-g-point grouping. For SW, it parses the 14 unformatted band records, extracts KAO/KBO reference-pressure absorption arrays, Rayleigh/source arrays, and applies the WRF 16-to-band-dependent g-point weights from `module_ra_rrtmg_sw.F:4763-4784` and `module_ra_rrtmg_sw.F:4927-5027`. For LW, it uses the WRF 16-to-band-dependent g-point map and weights from `module_ra_rrtmg_lw.F:8073-8104` and `module_ra_rrtmg_lw.F:8244-8315`. Cloud optical coefficients are sourced from the WRF source tables and delta-scaling/absorption formulas at `module_ra_rrtmg_sw.F:2388-2428` and `module_ra_rrtmg_lw.F:2997-3018`.

This is a real spectral-table consumption fix for A2's R-2. It is not yet a full RRTMG port: the JAX kernels still do not implement full gas-species interpolation, McICA cloud sampling, or the complete AER transfer solvers.

## WRF Source Mapping

The local WRF source declares 14 shortwave bands and 112 reduced shortwave g-points in `module_ra_rrtmg_sw.F:31-37`. It declares 16 longwave bands and 140 reduced longwave g-points in `module_ra_rrtmg_lw.F:76-82`. This ADR follows the discovered local build rather than the sprint text's incorrect "32 bands" wording.

The harness binds the WRF wrapper-driver surfaces at `module_ra_rrtmg_sw.F:10034-10100` and `module_ra_rrtmg_lw.F:11570-11607`. Those wrappers call the internal AER RRTMG transfer drivers at `module_ra_rrtmg_sw.F:11462-11484` and `module_ra_rrtmg_lw.F:12768-12778`. Initialization opens `RRTMG_SW_DATA` and `RRTMG_LW_DATA` at `module_ra_rrtmg_sw.F:11667-11685` and `module_ra_rrtmg_lw.F:13046-13067`. The data-record comments state 14 SW read records and 16 LW read records at `module_ra_rrtmg_sw.F:11705-11710` and `module_ra_rrtmg_lw.F:13085-13090`.

Pressure-layer mass in the JAX compact kernels follows the local harness interface reconstruction at `scripts/wrf_rrtmg_harness.f90:294-321`. WRF's SW and LW heating-factor comments define the flux-over-pressure-to-K/day conversion at `module_ra_rrtmg_sw.F:4880-4901` and `module_ra_rrtmg_lw.F:8212-8233`. The LW wrapper lines `module_ra_rrtmg_lw.F:12823-12829` only convert `hr` from K/day to K/s and Exner-normalized tendency; they are not a pressure-thickness heating formula.

## Tables And JAX Use

`scripts/extract_rrtmg_tables.py` parses the real big-endian Fortran sequential-unformatted records, verifies 14 SW and 16 LW records, stores raw payload bytes/offsets, and pins SHA-256 for data and source files. The regenerated NPZ is 1,747,092 bytes with SHA-256 `cffd87d494e3f8c2da6bedac42d6626a993bdcd777dcd0bad53dee5e4f7f96c8`.

The active spectral coefficient values are no longer pinned to the A2 clip floors: the regenerated active-value fraction at old floors `0.0025`, `1e-5`, `0.25`, `0.16`, `0.003`, and `0.2` is `0.0`. The JAX bundle exposes SW absorption as `(14,59,12)` over bands, WRF reference pressure levels, and padded reduced g-points, plus SW g-point weights/masks, Rayleigh coefficients, and cloud extinction/SSA. LW now also exposes pressure-resolved absorption as `(16,59,16)`, plus LW g-point weights/masks, band weights from WRF `delwave`, and LW cloud absorption.

## Validation And Gate Status

Attempt 3 restores non-vacuous Tier-1 tolerances in the manifests: flux outputs use `abs=1.0 W m-2, rel=0.05`, and heating-rate outputs use `abs=1.0e-4 K s-1, rel=0.05`. These replace the A2 `abs=1200 W m-2` and `rel=15.0` carry-forward regime.

Under those strict tolerances, the current compact JAX kernels do not pass Tier-1. Regenerated artifacts report SW max absolute errors of `flux_down=863.4149601378938 W m-2`, `flux_up=1578.875792806668 W m-2`, and `heating_rate=6.909078736834584e-4 K s-1`. LW max absolute errors are `flux_down=228.98306589556717 W m-2`, `flux_up=176.3891440048552 W m-2`, and `heating_rate=1.5330015754548213e-4 K s-1`. The gate is therefore `FALLBACK` for correctness, not a performance GO.

Tier-2 was updated to remove the A2 JAX-side tautological energy record. It now checks JAX SW and LW heating against candidate flux divergence using the WRF fixture pressure-layer mass, keeps real-driver SW/LW closure checks, and checks LW Stefan-Boltzmann surface emission against `sigma * emissivity * T_sfc^4`. The regenerated Tier-2 artifact passes.

Profile reporting remains honest: `kernel_launches_per_step` equals `raw_hlo_launch_marker_count`, now `28` after the reduced-g-point table expansion (`15` SW plus `13` LW). This preserves the raw-count reporting rule, but it no longer preserves A2's raw value of 22.

## Consequences

Positive consequence: the A2 R-2 clipped coefficient defect is removed, and the A2 R-3 vacuous tolerance defect is removed. The repository now has real WRF driver fixtures, real table provenance, real reduced-g-point spectral coefficients, strict manifest tolerances, and non-tautological Tier-2 records.

Cost: the compact JAX transfer kernels remain inadequate against the real WRF RRTMG fixture once tolerances are meaningful. The next decision should be a dedicated M5-S3.x implementation sprint for full gas-species interpolation plus SW/LW transfer parity, or an explicit architecture decision that RRTMG is outside the compact-column JAX scope.
