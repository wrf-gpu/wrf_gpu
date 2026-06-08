# gpuwrf — source tree

`gpuwrf` is the JAX/XLA, GPU-resident, WRF-compatible NWP implementation. It runs a
standalone WRF v4 ARW forecast end-to-end on a single GPU (native init or replay),
reads a standard WRF `namelist.input`, and writes a WRF-compatible `wrfout`.

Top-level layout:

- `dynamics/` — the ARW dynamical core (RK3 + acoustic substeps, flux-form advection,
  geopotential/`w` implicit solve), fp64-operational.
- `physics/` — microphysics, PBL + surface layer (MYNN-EDMF, sfclay), radiation (RRTMG
  SW/LW, Dudhia SW, classic RRTM LW), Noah-MP land, cumulus (KF/BMJ/Tiedtke/Grell-Freitas),
  gravity-wave drag.
- `coupling/` — physics↔dynamics couplers (tendency assembly + cadence).
- `io/` — `wrfinput`/`wrfbdy`/met_em readers, `wrfout`/`auxhist` writers, namelist parsing,
  and the fail-closed scheme catalog + validator.
- `integration/` — the operational scan loop, the standalone native-init and live-nested
  pipelines, and the daily/replay pipeline.
- `runtime/` — device/sharding config, the persistent JIT cache, state layout (SoA pytree).
- `contracts/` — frozen interface contracts (state, halo, physics interfaces).

Validation is against an unmodified WRF build as the oracle (proof objects under `proofs/`),
not bitwise Fortran-source parity. See the top-level `README.md` and `docs/`.
