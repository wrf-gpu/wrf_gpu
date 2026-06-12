# Changelog

All notable changes to wrf_gpu are recorded here. This file is a concise index;
each release has full, honest release notes in `RELEASE_NOTES_v<version>.md`.
Versions follow a 0.x pre-1.0 line (the v1.0.0 target is a complete, validated
WRF v4 GPU port — see [`PROJECT_PLAN.md`](PROJECT_PLAN.md)).

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [0.14.0] — Memory + WRF-identity release

> **Status placeholder.** The two 72 h field-parity gates (Canary L2 d02,
> Switzerland d01) are the v0.14 release gate. Final highlights and gate numbers
> are pending: `<manager: final 72h gate numbers + headline highlights>`.

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
