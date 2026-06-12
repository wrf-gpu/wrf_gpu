# License Notes

This file is not legal advice. The project's own license is in
[`LICENSE`](LICENSE); these notes cover the WRF-derived-material question on top
of it.

- **wrf_gpu is a clean reimplementation**, not a Fortran-source port of WRF. It
  validates against WRF as an oracle and does not redistribute WRF source.
- **WRF naming.** Do not brand the project as WRF. Use *WRF-compatible*,
  *WRF-like*, or *WRF-derived fixture* language. WRF is a registered name of
  UCAR/NCAR.
- **WRF-derived material.** Where any copied WRF-derived content survives
  (variable mappings, fixture metadata, documentation excerpts, lookup tables
  regenerated from WRF data files), preserve the required upstream notices.
  WRF source is in the public domain (UCAR/NCAR); confirm the current terms from
  the official UCAR/NCAR sources before relying on that for redistribution.
- **Third-party data.** Observation datasets (e.g. AEMET station data) and any
  met_em/CPU-WRF reference outputs are **not** redistributed in this repository;
  users supply their own (see `docs/quickstart.md`).
