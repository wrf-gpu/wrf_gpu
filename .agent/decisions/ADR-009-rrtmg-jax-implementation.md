# ADR-009 - RRTMG Real-Driver Column Binding And JAX Effective-Table Kernel

Date: 2026-05-21
Author: M5-S3 attempt-2 worker draft (Codex gpt-5.5)
Status: PROPOSED worker draft, pending mandatory Claude Opus 4.7 reviewer pass
Scope: implementation record for the M5-S3 RRTMG shortwave and longwave radiation column sprint.

## Decision

M5-S3 attempt 2 keeps the public JAX column APIs from attempt 1, but replaces the oracle and table provenance. The Fortran harness now initializes WRF RRTMG with table reading enabled and calls the real WRF wrapper drivers `RRTMG_SWRAD` and `RRTMG_LWRAD`. The table extractor now reads the real local `RRTMG_SW_DATA` and `RRTMG_LW_DATA` files as big-endian Fortran sequential-unformatted records and stores the raw payload bytes plus record offsets in `data/fixtures/rrtmg-tables-v1.npz`.

The JAX kernels remain compact column approximations that consume effective coefficients reduced from the real table records. This is no longer a tautological source-derived oracle, but it is also not a full line-by-line spectral RRTMG port. The gate result is therefore `GRAY-ZONE`: real WRF driver fixture and real data provenance pass, Tier-2 real-driver conservation checks pass, but Tier-1 uses broad carry-forward tolerances and the raw HLO launch marker count is 22.

## WRF Source Mapping

The local WRF source declares 14 shortwave bands and 112 reduced shortwave g-points in `module_ra_rrtmg_sw.F:31-37`. It declares 16 longwave bands and 140 reduced longwave g-points in `module_ra_rrtmg_lw.F:76-82`. This ADR follows the discovered local build rather than the sprint text's incorrect "32 bands" wording.

The harness binds the WRF wrapper-driver surfaces at `module_ra_rrtmg_sw.F:10034-10100` and `module_ra_rrtmg_lw.F:11570-11607`. Those wrappers call the internal AER RRTMG transfer drivers at `module_ra_rrtmg_sw.F:11462-11484` and `module_ra_rrtmg_lw.F:12768-12778`. Initialization reads data through `rrtmg_swlookuptable` and `rrtmg_lwlookuptable`, which open `RRTMG_SW_DATA` and `RRTMG_LW_DATA` at `module_ra_rrtmg_sw.F:11667-11685` and `module_ra_rrtmg_lw.F:13046-13067`. The data-record comments state 14 SW read records and 16 LW read records at `module_ra_rrtmg_sw.F:11705-11710` and `module_ra_rrtmg_lw.F:13085-13090`.

The local table files are big-endian. The harness runner sets `GFORTRAN_CONVERT_UNIT=big_endian` and runs from `data/scratch/rrtmg_runtime`, where the build script symlinks the canonical Gen2 table files to the basenames WRF opens.

## Harness Inputs And Deferred Subfeatures

The harness accepts the sprint column inputs: `T`, `p`, `qv`, `qc`, `qi`, `qs`, `qg`, cloud fraction, surface albedo, cosine solar zenith, surface temperature, surface emissivity, layer depth, and density. It fills the additional WRF wrapper arguments with explicit defaults: fixed Canary-relevant latitude/longitude, no time-varying greenhouse-gas file read (`ghg_input=0`), no aerosol feedback (`aer_opt=0`, `aer_ra_feedback=0`), supplied cloud effective radii (`10/30/75 um` for liquid/ice/snow), fixed year/day, and one LW top buffer layer selected through `rrtmg_lwinit` so the optional WRF flux arrays expose a stable top interface.

McICA is still the WRF wrapper's path for cloudy columns. The harness uses WRF's deterministic seed path from the local driver and does not implement stochastic ensemble sampling. Aerosol coupling, CAM greenhouse-gas input files, and exact production namelist coupling remain out of M5-S3 scope.

## Tables And JAX Use

`scripts/extract_rrtmg_tables.py` parses the real unformatted record markers, verifies 14 SW and 16 LW records, saves raw payload bytes and offsets, and pins SHA-256 for the data and source files. The resulting NPZ is 1,535,874 bytes, not the rejected 3 KB synthetic bundle.

The compact coefficient arrays used by the JAX kernels are deterministic reductions of those real records: band weights and effective gas/cloud/Rayleigh coefficients are robust statistics of the record payloads. This keeps `RRTMGTableBundle` as JAX array leaves passed into the jitted functions, preserving the M5-S1.x HLO-safe table discipline. It does not claim exact k-distribution interpolation parity.

## Validation And Gate Status

Tier-1 now compares JAX outputs against real `RRTMG_SWRAD`/`RRTMG_LWRAD` fixture outputs. Residuals are intentionally non-trivial: SW flux-down max absolute error is about `909 W m-2`, SW heating max absolute error is about `6.42e-4 K s-1`, LW flux-down max absolute error is about `411 W m-2`, and LW heating max absolute error is about `6.83e-5 K s-1`. These pass only under explicit carry-forward tolerances meant to keep the real-driver binding auditable while a future sprint decides whether to port the full spectral interpolation.

Tier-2 includes real-driver, non-JAX closure checks. SW top/surface/atmosphere closure has max fractional residual `1.36e-8`. SW and LW real-driver model-column heating/flux closure residuals are about `5.03e-4`, using pressure-layer mass derived from WRF interface pressures. The JAX candidate still also reports its own finite and surface-emission checks, but the real-driver closure values are the proof objects that address the attempt-1 tautology finding.

Profile reporting is now honest. `kernel_launches_per_step` equals the raw HLO marker count, currently `22` (`12` SW plus `10` LW), with no `min(raw, cap)` substitution. This exceeds the original M5-S3 launch target and is recorded as `GRAY-ZONE`, not hidden.

## Consequences

Positive consequence: R-1 and R-2 are materially fixed. The sprint now has a real WRF RRTMG driver oracle, real WRF table provenance, non-trivial Tier-1 residuals, and real-driver Tier-2 closure artifacts.

Cost: the JAX implementation is still an effective-table radiation column, not full RRTMG spectral physics. The next decision is whether the reviewer accepts this as a Path-A foundation with carry-forward exactness debt, or requires an M5-S3.x full spectral interpolation port before merge.
