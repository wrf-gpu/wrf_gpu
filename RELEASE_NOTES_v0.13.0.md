# Release notes — wrf_gpu v0.13.0

**v0.13.0 is "Validate & Accelerate": it lifts the single-GPU VRAM ceiling that
gated v0.12.0, turns gravity-wave drag on by default on the nested 1 km path, and
hardens the validation/reproducibility story so an outside reviewer can reproduce
the numeric-correctness core from a clone alone.**

The keystone is a three-part RRTMG VRAM reduction (g-point band-tiling +
taumol/optics construction chunking + leading-column tiling) that cut peak
shortwave VRAM **−88.6 %** and longwave **−43.6 %** at production column depth,
then capped the formerly full-column g-point transient — bit-identical
(`max_rel = 0.0`). The large-column GPU suite records LW untiled OOM on a
32.11 GiB allocation, LW tiled 5.37 GiB; SW 10.03 → 1.62 GiB
(`proofs/v013/rrtmg_column_tile_vram_suite.json`).
That headroom unblocked the v0.12.0-deferred **24 h nested 1 km + GWD** run, which
now passes `PIPELINE_GREEN`, so gravity-wave drag (`gwd_opt=1`) is **default-on on
the nested path** in v0.13.0 (it was gated-off in v0.12.0 because it OOM'd at
~sim-hr 7).

This release builds on the full v0.12.0 capability set (standalone out-of-box CLI,
standalone live-nested `--max-dom`, persistent JIT cache, fail-closed scheme
catalog, WRF-faithful PSFC fix, runnable equivalence demo) and the v0.11.0 set
(live multi-domain nesting, restart continuity, conservation-closed budgets,
MYNN-EDMF, topographic/slope radiation, terrain-slope diffusion,
KF/BMJ/Tiedtke/Grell-Freitas cumulus), all of which carries forward unchanged.

> **Honest framing — read this first.** wrf_gpu is a **WRF-compatible
> reimplementation** (a clean JAX rewrite validated against WRF as an oracle),
> **not a Fortran-source port**, and it is a **transparent research artifact, not a
> full WRF replacement.** The credibility gate for any "operational / replacement"
> claim — **24 h forecast-skill closure (T2/U10/V10) vs CPU-WRF — is NOT closed in
> v0.13.0.** It is a hard dynamics/MYNN/`*_tendf` GPU problem with no cheap knob,
> and it is the dominant carry-over (see KI-9 / KI-4 below and
> [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)).

## Headline features

- **RRTMG VRAM-floor chunking + column tiling — the keystone (numerically inert).** A three-part
  reduction of the dominant fp64 VRAM consumer: (1) **g-point band-tiling** of the
  SW two-stream/g-point reduction and the LW `rtrnmc` band loop via `lax.scan` over
  the band axis, (2) **taumol/optics construction chunking** so the per-band gas
  optical depths are built one tile at a time instead of stacking all bands upfront,
  and (3) **leading-column tiling** so the largest g-point temporary is not
  materialized over every column at once.
  Combined peak VRAM at production depth (nlev 48 / ncol 24576): **SW
  16730 → 1906 MiB (−88.6 %)**, **LW 17854 → 10068 MiB (−43.6 %)**. The deep
  nlev 64 / ncol 49152 case (the GWD-nested-1km OOM family) **previously OOM'd and
  now fits**. **Bit-identical** across all tested chunk widths (SW 1/2/3/4/5/7/14,
  LW 1/2/4/8/16; `max_rel = 0.0`), all-sky and clear-sky; energy-closure invariant
  3.6×10⁻¹⁵ (fp64 machine-eps). The final large-column GPU suite shows LW untiled
  OOM on a 32.11 GiB allocation, LW tiled peak 5374.84 MiB; SW untiled 10033.1 MiB,
  SW tiled 1619.54 MiB. Public API + defaults unchanged. Proofs:
  `proofs/v013/gpoint_chunk_rrtmg.json`, `proofs/v013/optics_taumol_chunk.json`,
  `proofs/v013/rrtmg_column_tile.json`, `proofs/v013/rrtmg_column_tile_vram_suite.json`.
- **Gravity-wave drag operational coupling — now default-ON on the nested path.**
  With the chunked RRTMG temporary, the **24 h nested 1 km + GWD** run that OOM'd at
  step 0 / hr 7 in v0.12.0 now fits and runs clean: `PIPELINE_GREEN`, 24/24 `wrfout`
  per domain, all fields finite at +24 h (d03 T2 ∈ [279.6, 300.9] K), forecast-only
  ≈ 1.86 h. `gwd_opt=1` is **honoured by default**; `GPUWRF_GWD_NESTED=0` forces it
  off. Completion + finiteness gate on the prod-failing case, **not** a skill-vs-truth
  claim. Proof: `proofs/v013/gwd_nested_24h_gate.json`.
- **Compile-speed infra re-landed and GPU-validated.** AOT precompile + a persistent
  XLA autotune cache + compile-cache hardening. The v0.12.0 GPU-abort (the XLA
  autotune-flag injection that broke the GPU path and forced the v0.12.0 revert) is
  **fixed**: real-GPU import is clean (no 3 s abort, `XLA_FLAGS=None`), and a
  subprocess flag-probe **drops** unsupported flags instead of aborting. The
  persistent autotune cache is **opt-in, default-off** (`GPUWRF_XLA_AUTOTUNE_CACHE`)
  and its measured warm-cache **effect is gated/unadvertised until measured on the
  integrated GPU smoke** (the cold→warm autotune speedup is a labelled follow-up, not
  a shipped headline). CPU cold→warm cache-hit speedup ~4.5× on the representative
  graph; 22 tests. Proof: `proofs/v0130/compile_speed.json`.
- **RRTM-LW (classic AER `ra_lw=1`) hardening — skeptic-passed + 2 fixes.** An
  independent cross-model skeptic audit of the band/laytrop vectorization (the author
  wrote both kernel and oracle) found **no JAX port bug** (max divergence 2.7×10⁻¹³,
  oracle integrity clean). Two findings fixed: **F1** `_nbuf` is now grid-aware (real
  `top_pressure_pa`, WRF `nint(p_top_mb/4)`; `None`→5000 reproduces production
  bit-identical, 7+7 cases max diff 0.0) and **F2/F3** replaced masking-clamps with
  **fail-loud NaN guards** (a forbidden-pattern removal). New pristine-WRF non-5000-ptop
  oracle (100 mb / 20 mb): grid-aware rel ~2×10⁻¹³ vs the old hardcoded 4.8×10⁻²/NaN.
- **Multi-GPU domain decomposition (fake-mesh, bit-identical).** `shard_map` +
  `lax.ppermute` periodic-ring halo shards the 5th-order advection and 6th-order
  diffusion stencils; partition-invariance is **bit-identical** (P = 2/4/8 == P = 1,
  max abs diff 0.0) on a CPU fake mesh of up to 8 devices. **HONEST:** this
  workstation has one physical RTX 5090, so **real multi-GPU throughput, NVLink/NCCL
  bandwidth, and collective overlap are UNMEASURED** — the per-watt / whole-Earth
  claims stay **PROJECTED, never MEASURED**. Default-off, 27 tests, 0 regressions.
  Proof: `proofs/v013/multigpu_fakemesh.json`.
- **Moisture flux-advection wired into RK3 (correctness, opt-in).** Condensates
  (`qv`/`qc`/`qr`/`qi`/`qs`/`qg`) can now be flux-advected by the resolved wind in the
  RK3 large step via `moist_adv_opt` — closing the gap where condensates previously
  had **zero resolved-wind advection** (moisture moved only through the physics
  boundary). **Default `moist_adv_opt=0` is byte-identical** by construction
  (production unchanged); the opt-in path is conservation-closed (8.2×10⁻¹⁶),
  WRF-parity bit-exact (1.7×10⁻¹⁶), and leaves the Straka/Skamarock idealized gates
  unchanged. This is a fidelity lever for the skill-closure carry-over (KI-9), shipped
  off-by-default. Proof: `proofs/v013/moisture_advection_wiring.json`.
- **MYJ PBL + Janjic-Eta surface layer — reference-only → operational.**
  `bl_pbl_physics=2` + its mandatory partner `sf_sfclay_physics=2` are now scan-wired
  via a jit/vmap-traceable MYJ rewrite + `vmap` Janjic + State adapters (TKE coupling
  faithful via `qke` carry). Oracle PASS vs v0.6.0 pristine-WRF savepoints (worst PBL
  2.7×10⁻¹¹ / SFC 1.6×10⁻¹⁰, 6 regimes, **not** a self-compare). Default suite
  byte-unchanged; the mandatory MYJ↔Janjic pairing is **fail-closed**. 101 tests.
  Per-scheme parity only — end-to-end coupled-RMSE vs CPU-WRF is a carry-over.
  Proof: `proofs/v013/myj_janjic_oracle.json`.
- **Positive-definite / monotonic moisture advection** (`scalar_adv_opt`
  extended to moisture species). The PD/monotonic limiter now covers the moisture
  species, not just theta: positivity enforced (vs unlimited −2.4×10⁻⁴), per-species
  conservation ~10⁻¹⁷–10⁻¹⁸ relative; default-off byte-identical. Proof:
  `proofs/v013/pd_moisture.json`.
- **Clear-sky radiation diagnostics — the 8 `…C` flux vars.** `SWUPTC/SWDNTC/
  SWUPBC/SWDNBC/LWUPTC/LWDNTC/LWUPBC/LWDNBC` (TOA + surface, up + down) via a
  WRF-faithful second clear-sky RT stream (clear vrtqdr_sw quadrature + clear
  `rtrnmc`), built on the new g-point band-tiling. Oracle PASS vs pristine-CPU-WRF
  d03 `…C` (clear-sky JAX-vs-WRF RMSE ~1.5 W/m² on SWUPTC; **not** a self-compare);
  all-sky byte-unchanged; default-off (`with_clear_sky`). Follow-up: thread
  `with_clear_sky` through `M9Diagnostics` for operational `wrfout`. Proof:
  `proofs/v013/clearsky_radiation.json`.
- **Outsider-runnable reproducibility.** 45 proof `.py` runners sanitized of
  hard-coded `<USER_HOME>` paths → repo-root resolvers + `WRF_PRISTINE_ROOT` env
  (semantics-preserving, `py_compile`-clean); the Thompson table assets are vendored
  and pinned (`manifest/reproducibility_assets.json`); a new
  `scripts/verify_reproducibility.sh` is **GREEN 11/11 outsider-runnable** on
  CPU-only. New docs: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). No `src`
  changes.
- **Community-standard validation suite.** A new `scripts/community_validation.sh`
  (CPU-only, like `verify_reproducibility.sh`) re-runs the community-standard tests an
  external NWP reviewer expects: published idealized dycore benchmarks (Straka
  density current + Skamarock/Bryan-Fritsch warm bubble vs WRF spec), closed-domain
  mass/energy conservation budgets, and bitwise `wrfrst` restart — all PASS, with an
  honest CPU-vs-GPU / corpus gap list. New docs:
  [`docs/VALIDATION.md`](docs/VALIDATION.md).
- **Powered n=15 TOST — scoring path unblocked (rc=2 fix).** The GPU
  `daily_pipeline` / `run_one_case` `rc=2` that blocked the powered TOST campaign in
  v0.12.0 was root-caused (two conflated sources: a per-case `L2_D02_BLOCKED` and an
  orchestrator `<2-scored` conflation) and **fixed**; the scoring path is proven
  `rc=0` on a real GPU `wrfout` vs CPU-WRF (`SCORING_PATH_RC0_PROVEN`), 7 tests. The
  **powered n=15 TOST campaign itself was not run for this release** — see the
  Resolved-at-tag section. No "TOST PASS" / statistical-equivalence is claimed.
  Proofs: `proofs/v013/tost_rc2_fix.json`,
  `proofs/v013/tost_scoring_path_cpu_proof.json`.

## Honest equivalence framing — the credibility gate is NOT closed

The v0.12.0 runnable equivalence demo verdict on the default **24 h d02** case
**carries forward unchanged**: `NOT_EQUIVALENT` (6 of 10 fields exceed the
predeclared operational tolerances), **dominated by lead-time wind divergence** (3D
meridional wind **V** pooled RMSE 8.13 m/s vs a 1.8 m/s bar). v0.13.0 ships fidelity
levers toward this gap (moisture flux-advection into RK3, MYJ+Janjic, clear-sky
diagnostics) **off-by-default**, but **does not close it**. Closing it is a hard
dycore-`ph'` / MYNN / `*_tendf` GPU effort with no cheap knob, and it remains the
single most important carry-over (KI-9). Full numbers and framing:
[`docs/equivalence-demo.md`](docs/equivalence-demo.md), tracked as **KI-9**.

## Speedup (carried from v0.12.0; no new headline number)

All numbers are one RTX 5090 vs 28-rank CPU-WRF on the same workstation, both
fp64, same d02 3 km grid, per forecast-hour. Full reconciliation:
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md). v0.13.0 makes **no new shipped speedup
claim**: the compile-speed autotune-cache *effect* stays gated/unadvertised until
measured on the integrated GPU smoke, and real multi-GPU throughput is unmeasured.

- **Warm kernel (apples-to-apples): ~5×** (band 5–8×, strict dt-parity floor ~3.2×).
- **Warm real-user wall: ~2.5×** — full command-to-finish wall, persistent cache warm.
- **Equivalence-demo real-user: ~4.26× warm-cached / ~1.70× cold** — the 24 h d02 demo.

## Known issues (carried + new)

See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) for full detail.

| ID | Summary | Severity |
|---|---|---|
| **KI-9** | **The credibility gate.** Equivalence demo: **24 h d02 `NOT_EQUIVALENT`**, dominated by **lead-time wind divergence** (3D V pooled RMSE 8.13 m/s). v0.13.0 ships off-by-default fidelity levers but does **not** close it (hard dycore-`ph'`/MYNN/`*_tendf` GPU work, no cheap knob). | Documented gap |
| KI-4 | d02 **U10** episodic final-lead under-prediction (8.06 vs 7.5 m/s bar); within bar at all other leads, beats persistence 23/24. Tied to KI-9 / MYNN cloud PDF. | Documented residual |
| KI-3 | Focused **104-variable** `wrfout` (vs WRF's full 375-var schema). Remaining gap is mostly stochastic-seed + less-common diagnostics. | Scope boundary |
| KI-5 | Powered **n=15 TOST**: scoring path **unblocked** (rc=2 fixed); the powered campaign was not run for this release. **No TOST PASS is claimed** (n=15 underpowered; n≈27 for full power). | Scope boundary |
| KI-6 | RRTMG SW intermediate `taug` differs in 4 UV bands; integrated fluxes pass tier-1 (< 0.05 % rel). Pre-existing; carried. | Isolated |
| KI-7 | Free-running (`run_boundary=False`) on **wide domains** (nx≈160+) can go unstable beyond ~14 h. Operational path uses boundary forcing. | Robustness edge |
| **KI-10** | **Moisture advection cadence refinements** (GPT Q1/Q3): the opt-in moisture flux-advection shares the theta acoustic-cadence rather than accumulating acoustic fluxes, and physics-tendency folding is not yet WRF-cadence-exact. Default-off, so no shipped-behavior impact. | Fidelity refinement |
| **KI-11** | **2-way nesting equivalence vs CPU-WRF untested.** One-way 24 h is proven; the 2-way feedback path is finite/stable but its 24 h real-GPU equivalence vs CPU-WRF is unmeasured. | Scope boundary |

## Resolved at tag (or pending the final gates)

- **GWD on the nested 1 km path — RESOLVED (now default-on).** The v0.12.0 deferral
  (24 h nested 1 km + GWD OOM'd at ~sim-hr 7) is closed by the RRTMG VRAM chunking:
  `PIPELINE_GREEN`, 24/24 `wrfout` per domain, all-finite at +24 h, forecast-only
  ≈ 1.86 h (`proofs/v013/gwd_nested_24h_gate.json`). `gwd_opt=1` honoured by default.
- **Integrated GPU smoke gate (Step A) — represented by the nested 24 h 1 km gate
  above** (`PIPELINE_GREEN`, 24/24 `wrfout` per domain, all-finite at +24 h,
  forecast-only ≈ 1.86 h; `proofs/v013/gwd_nested_24h_gate.json`). This is a "does
  the integrated v0.13 trunk run clean + fast end-to-end" gate (no CPU compare).
- **Powered n=15 TOST — not scored for this release.** The scoring path is proven
  `rc=0`; the powered campaign was not run. **No equivalence / TOST PASS is claimed.**
- **Compile-speed autotune-cache effect — gated/unadvertised, not yet measured on
  GPU.** The infra is GPU-validated (clean import, no abort); the *effect* is gated
  until measured.

## Deliberately deferred to v0.14+ (deliberate scope boundaries, not silent gaps)

The next roadmap is **Tier 3 (the scheme long-tail toward v1.0.0) + the v0.13.0
carry-overs**. See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) and
[`.agent/decisions/V0130-ROADMAP.md`](.agent/decisions/V0130-ROADMAP.md).

- **24 h forecast-skill closure (T2/U10/V10) vs CPU-WRF** (KI-9) — the credibility
  gate for any "operational / replacement" claim; hard dycore/MYNN/`*_tendf` GPU work.
- **Moisture-advection cadence refinements** (KI-10) — acoustic-accumulated fluxes +
  WRF-cadence-exact physics-tendency folding.
- **2-way nesting 24 h real-GPU equivalence vs CPU-WRF** (KI-11) — only finite/stable
  is proven today.
- **Tier-2 speed/architecture remainder** — sub-jit split + recompile hygiene,
  `--xla_gpu_force_compilation_parallelism` + dev `--fast-compile`, CPU-flock for
  idle nightly cores.
- **Multi-hardware / independent reproduction** — v0.13.0 is one RTX 5090, one
  JAX/CUDA stack; a second GPU/driver/stack + an independent reproduction run is a
  carry-over.
- **Gotthard / Switzerland operational validation** — not a v0.13.0 pass. Case
  generation and CPU truth exist; the v0.12 128²/150² attempt is documented as
  fp64 OOM/grid-ceiling evidence. The post-memory-fix GPU-vs-CPU-WRF Switzerland
  run is v0.14 B7.
- **Tier-3 scheme long-tail** — ~22 microphysics, ~10 cumulus, ~8 PBL, ~12 radiation,
  ~4 surface-layer + ~6 LSM families, each opt-in / fail-closed until oracle-proven.
- **Full 375-variable `wrfout`** (KI-3), **RRTMG SW `taug` UV-band fix** (KI-6), and
  the **`*_tendf` source-tendency adapter** for RK-stage physics.

## Out of scope (documented boundary, not v0.13.0 and not the roadmap)

WRF-Chem · WRF-Fire · WRF-Hydro · coupled ocean · urban canopy (UCM/BEP/BEM) ·
moving nests · FDDA/DA · stochastic physics. These are rejected, not roadmap.

## Reverted / carried from v0.13.0

- The compile-speed **autotune-cache effect** is gated/unadvertised until measured on
  the integrated GPU smoke (the infra is GPU-validated; only the *effect* number is
  pending).
- The moisture flux-advection and clear-sky-radiation diagnostics are **wired but
  default-off** (opt-in); operationalizing them on the default path (and the cadence
  refinements, KI-10) is a v0.14+ item.
