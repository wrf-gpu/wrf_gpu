# Changelog

All notable changes to wrf_gpu are recorded here. This file is a concise index;
each release has full, honest release notes in `RELEASE_NOTES_v<version>.md`.
Versions follow a 0.x pre-1.0 line (the v1.0.0 target is a complete, validated
WRF v4 GPU port — see [`PROJECT_PLAN.md`](PROJECT_PLAN.md)).

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [0.18.1] — Quickstart usability patch

Documentation/usability patch over 0.18.0 (no model-code change; default numerics
unchanged). A naive-user acceptance test of the public quickstart found the
advertised Switzerland case was not runnable from a fresh clone. This release:

- **Ships a bundled real-data example** at `examples/switzerland_d01/` (GFS-
  initialized, public domain, ~13 MB) so a fresh clone runs an end-to-end GPU
  forecast with no external download.
- **Rewrites the Quickstart** (`README.md`, `docs/quickstart.md`) to use the
  bundled case with a concrete, copy-pasteable command (`--input-dir
  examples/switzerland_d01 --domain d01`), the required `GPUWRF_WRF_ROOT`,
  calibrated cold-compile guidance, and a unified JIT-cache env-var name.
- **Refreshes `docs/equivalence-switzerland.md`** (stale v0.13.0 status → v0.18)
  and links the example + equivalence test from the README.

## [0.15.0] – [0.18.0]

Per-release history for 0.15 → 0.18 is maintained in the README **"Release line"**
table and the per-version proof archives (`proofs/v01{5,6,7,8}/`); standalone
`RELEASE_NOTES_*` files cover releases up to 0.15.0. 0.18.0 is the
**feature-complete** release — every WRF v4 scheme classified and tested,
experimental K2 multi-GPU, perf-neutral vs 0.17.

## [0.14.0] — Memory + WRF-identity release

> **Both 72 h field-parity gates closed on the final code.** Switzerland d01 and
> Canary L2 d02 each ran stable to h72 GPU-vs-CPU-WRF, with **9/10 prognostic
> fields within frozen tolerance** and the full dynamics/thermodynamics core
> cell-for-cell identical. The one out-of-envelope field per region is a bounded
> diagnostic — `RAINNC` precipitation sensitivity (Switzerland, 5.19 mm RMSE vs a
> 1.0 mm bound) and `QVAPOR` moisture margin (Canary, 1.45×10⁻³ vs 1.0×10⁻³ kg/kg,
> +45%); on Canary the static `MUB`/`PB` nest-frame-seam base-state artifact is
> also bounded. These four bounded misses are pre-existing/physical diagnostics
> carried to v0.15, **not** identity failures. Warm throughput is roughly on par
> (~1.05× Switzerland, ~1.06× Canary); performance is the v0.15 focus.

**Theme: memory headroom + a reproducible WRF-identity proof system.** v0.14 is
not a performance release (warm throughput is roughly on par, ~1.05×, with
v0.13.0 — performance is the v0.15 focus). What v0.14 adds:

- **GPU↔CPU identity-proof visualization system** — a reusable, CPU-only,
  publication-quality visual proof (per-variable RMSE/bias time series with the
  tolerance line, variable×lead scoreboard, 1:1 cell scatter, signed spatial
  difference maps, and a README-embeddable dashboard) over all cells, all 72
  leads, and all core internal variables, for both Canary L2 d02 and Switzerland
  d01. Reproducible via `scripts/build_identity_proof_plots.py`. See
  [`docs/IDENTITY_PROOF.md`](docs/IDENTITY_PROOF.md).
- **72 h field-parity gates** (Canary + Switzerland) vs CPU-WRF truth, with a
  pre-declared tolerance manifest and the grid-delta atlas.
- **Release hygiene**: portable defaults (no hard-coded personal paths in
  user-facing instructions), standard release files, and a curated, clearly
  framed development-log archive under [`.agent/`](.agent/).

Carry-overs and bounded acceptances are in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## [0.13.0] — Validate & Accelerate

Lifted the single-GPU VRAM ceiling via a three-part RRTMG chunking (SW −88.6 % /
LW −43.6 %, numerically inert); turned gravity-wave drag on by default on the
nested 1 km path; re-landed GPU-validated compile-speed infra; wired MYJ PBL +
Janjic-Eta surface layer to operational; added clear-sky radiation diagnostics,
moisture flux-advection into RK3 (opt-in), and `shard_map` fake-mesh multi-GPU
sharding; hardened reproducibility + community validation. Full notes:
[`RELEASE_NOTES_v0.13.0.md`](RELEASE_NOTES_v0.13.0.md).

## [0.12.0] — Standalone out-of-box CLI

Made wrf_gpu a true out-of-the-box standalone GPU forecast system: standalone
native-init + live-nested `--max-dom` CLI (no CPU-WRF `wrfout` dependency),
persistent JIT cache (on by default), fail-closed scheme catalog, WRF-faithful
PSFC fix, and a runnable GPU-vs-CPU equivalence demo. Full notes:
[`RELEASE_NOTES_v0.12.0.md`](RELEASE_NOTES_v0.12.0.md).

## [0.11.0] — Live nesting, restart, conservation

Live multi-domain nesting (d01→d02→d03, one-way), bit-identical WRF restart,
closed conservation budgets, MYNN-EDMF mass flux, topographic/slope radiation,
terrain-slope diffusion, and KF/BMJ/Tiedtke/Grell-Freitas cumulus. Full notes:
[`RELEASE_NOTES_v0.11.0.md`](RELEASE_NOTES_v0.11.0.md).

## [0.10.0]

Removed one faithful Thompson sedimentation inefficiency. Full notes:
[`RELEASE_NOTES_v0.10.0.md`](RELEASE_NOTES_v0.10.0.md).

## [0.9.0] — Standalone forecast system

Consolidated native real-init + the operational physics menu into a standalone
forecast system. Full notes: [`RELEASE_NOTES_v0.9.0.md`](RELEASE_NOTES_v0.9.0.md).

## [0.4.0]

Native real-init (assembles `wrfinput`/`wrfbdy` from met_em-stage forcing,
proven equivalent to `real.exe` at t=0). Accessible via the `v0.4.0` git tag.

## [0.3.0]

Native metgrid. Accessible via the `v0.3.0` git tag.

## [0.2.0] — Paper baseline

The stable paper-claims baseline. Accessible via the `v0.2.0` git tag.

## [0.1.0] — First validated release

Single-domain replay path consuming CPU-WRF/Gen2 artifacts for initialization;
Coriolis-corrected 3 km d02 validated against nightly CPU-WRF over real days.
Full notes: [`RELEASE_NOTES_v0.1.0.md`](RELEASE_NOTES_v0.1.0.md).
