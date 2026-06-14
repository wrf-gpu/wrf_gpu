# Changelog

All notable changes to wrf_gpu are recorded here. This file is a concise index;
each release has full, honest release notes in `RELEASE_NOTES_v<version>.md`.
Versions follow a 0.x pre-1.0 line (the v1.0.0 target is a complete, validated
WRF v4 GPU port — see [`PROJECT_PLAN.md`](PROJECT_PLAN.md)).

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [0.16.0] — Stability release (+aerosol "+1", 1 km-unlock)

> **Coupled stability across the whole implemented physics menu.** Every
> implemented L2 scheme runs coupled-green on a real Switzerland d01 case inside
> the frozen v0.14 tolerance band — **24 of 25 L2 targets GREEN**, the 25th
> (Noah-classic land surface) an honest scope-carry → rollup
> **`ALL_GREEN_OR_CARRIED`**.

- **Stability coverage:** per-scheme COUPLED real-case gate
  (`proofs/v016/coupled_coverage_gate.py`); 24 GREEN + 1 scope-carry (lsm2);
  dashboard `proofs/v016/dashboard/`.
- **Aerosol-aware Thompson (`mp_physics=28`), the "+1":** `QNWFA`/`QNIFA`
  prognostics end-to-end (62-leaf State, append-only at the end), per-scheme
  **oracle PASS** vs the unmodified pristine WRF `module_mp_thompson.F`
  (5187-col, GPU). Coupled field-gate carried (GPU time only).
- **1 km-unlock:** chunked MYNN BouLac (`GPUWRF_MYNN_BOULAC_CHUNKED=1`, default
  off) makes a **1 km single-domain fp64 forecast fit on one RTX 5090** (dense
  OOMs 147 k cols at ≈ 18.8 GiB; chunked fits at 18.25 GiB), **bit-identical** to
  dense (`max_abs == 0.0`). Measured in a **fresh process per grid**; repeated
  multi-grid runs in one process can fragment allocator memory (isolate grids per
  process or recycle the process).
- **Honest performance — fp32 make-or-break CONCLUDED (double-confirmed):** fp64
  GPU ≈ CPU-WRF parity (GeForce fp64 1/64 hardware law); the **valid-numerics fp32
  ceiling is ~1.1×** (full-ws 16k 1.107× / 65k 1.110×, VRAM ratio 1.000; GPT
  independently reproduced 1.105× / 1.111×) — larger fp32 speedups are
  **precluded** by the conservation/cancellation fp64 pins (the ~4.3× "cost proxy"
  is numerically invalid: corrupts conservation; qke non-finite at 1 km). The
  genuine ~1.1× fp32 lane ships now; fusion ≈ 0% (XLA optimal). No end-to-end
  speedup claimed. Evidence: `proofs/v016/fp32_verdict/`.
- Scope-carries to v0.17: Noah-classic land bundle, RRTMG variants,
  LW31/MP97/New-Tiedtke. (Full-working-set fp32 is **CONCLUDED in v0.16, not a
  carry** — the valid-numerics ceiling is proven ~1.1×.)

Full notes: [`RELEASE_NOTES_v0.16.0.md`](RELEASE_NOTES_v0.16.0.md).

## [0.15.0] — Final fp64 kernel + WRF-fidelity release

> **Final fp64 GPU kernel (adversarially confirmed near-optimal, device-bound)**
> + WRF-fidelity fixes (MYNN-condensation `niter` 50→16, Thompson
> cold-collection) + MUB/PB nest-base-state seam fix; both 72 h GPU-vs-CPU-WRF
> field-parity gates re-closed (Switzerland d01 + Canary L2 d02), 9/10 fields
> within frozen tolerance, cleaner than v0.14 at the atlas level. Honestly
> ~parity total-wall (0.99×/1.04×); no multi-× or large-grid speedup claimed.

Full notes: [`RELEASE_NOTES_v0.15.0.md`](RELEASE_NOTES_v0.15.0.md).

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
