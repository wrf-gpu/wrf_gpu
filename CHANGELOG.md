# Changelog

All notable changes to `wrf_gpu` will be documented here. The project uses semantic versioning as described in [README.md](README.md#versioning-policy).

## [Unreleased]

## [0.0.1] - 2026-05-28

### Added
- Initial public release.
- JAX-native implementation of WRF v4 ARW dynamical core (RK3 + acoustic substep, C-grid staggering, terrain-following mass coordinate).
- Minimum operational physics suite: Thompson microphysics, MYNN PBL, RRTMG radiation, Noah/Noah-MP-style surface.
- Whole-state device residency: zero inter-kernel D2H transfers inside the forecast loop (Nsight Systems verified).
- WRF-compatible NetCDF wrfout writer (41-variable minimum subset).
- Restart/checkpoint with bitwise continuity.
- AEMET station-observation verification scaffold (BIAS / MAE / RMSE / Fractions Skill Score).
- Multi-day batch pipeline driver.
- Idealized test cases (warm bubble, density current, Schaer mountain wave) with analytic IC builders.
- Comprehensive unit-test suite (~150 tests) covering savepoint parity, conservation, repeatability, restart, D2H invariant.
- Documentation: README, NOTICE, CITATION.cff, CONTRIBUTING, SECURITY, AI_USE.

### Validated
- 24h 3km regional forecast: ~12 min wall-clock on a single NVIDIA RTX 5090.
- Apples-to-apples speedup vs 28-rank CPU WRF (same machine, d02-only): **22.26×**.
- B6 savepoint parity vs WRF Fortran: 0.0 bitwise.
- Restart bitwise: max delta 0.0 across all 47 State fields.
- 1km full-domain memory probe: 7278 MiB / 32607 MiB (78% headroom).

### Known limitations
- Tested on a single workstation only (AMD Ryzen 9 + NVIDIA RTX 5090, 32 cores, 32 GB VRAM).
- T2 / U10 / V10 station-skill comparison vs CPU WRF on a single-day side-by-side shows the GPU forecast is materially less skilful than CPU WRF; the project is not yet an operational-replacement candidate. See preprint §8 and README "What remains" section.
- Single-GPU only. Multi-GPU halo exchange exists as a placeholder; no validated multi-node result.
- Data path is replay-based (boundary forcing from retained CPU WRF outputs); direct AIFS / global-model IC/BC ingestion is not implemented.

## Versioning policy

- `0.0.x` — pre-arXiv-preprint releases. Public reproducibility snapshots; no operational-quality claims.
- `0.1.0` — arXiv preprint companion release. Locked source corresponding to the cited paper version.
- `1.0.0` — reserved for when the operational-skill blockers are closed and a peer-reviewed paper has accepted the claim.

[Unreleased]: https://github.com/wrf-gpu/wrf_gpu/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/wrf-gpu/wrf_gpu/releases/tag/v0.0.1
