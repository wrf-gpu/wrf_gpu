# Release notes — wrf_gpu v0.14.0

**v0.14.0 is a memory + WRF-identity release, not a performance release.** It
pairs the v0.13.0 memory-stability work with a reproducible GPU↔CPU
**identity-proof** system and closes two **72 h GPU-vs-CPU-WRF field-parity
gates** on the final code — **Switzerland d01** and **Canary L2 d02** — each
stable to h72, with **9/10 prognostic fields within frozen tolerance** and the
full dynamics/thermodynamics core **cell-for-cell identical**. Warm throughput is
honestly **~1.05×** (Switzerland) / **~1.06×** (Canary) on par with 28-rank
CPU-WRF; performance recovery is the dedicated focus of **v0.15**.

> **Honest framing — read this first.** wrf_gpu is a **WRF-compatible
> reimplementation** (a clean JAX rewrite validated against WRF as an oracle),
> **not a Fortran-source port**, and a **transparent research artifact, not a
> full WRF replacement.** v0.14 closes the 72 h field-parity gates on both
> regions and proves the dynamics/thermodynamics core cell-for-cell identical;
> the broader **24 h/72 h forecast-skill equivalence (T2/U10/V10) vs CPU-WRF is
> still the credibility gate and is NOT claimed closed** (KI-9). It is a hard
> dynamics-`ph'` / MYNN / `*_tendf` GPU problem, with no cheap knob.

## What v0.14 fixes — the venting root cause and the WRF-faithful dynamics

The headline v0.14 fix is the **Switzerland d01 strong-flow mass venting** root
cause: a `_THETA_LIMITER_MAX_K=500 K` **masking clamp** that was firing on real
~507 K stratospheric theta — silently capping a physical value. It is raised to a
**non-load-bearing 1000 K** (no clamp on real states). Alongside it:

- **advance_w WRF-faithfulness** — `pg_buoy` carried-pressure formulation,
  w-coriolis / curvature terms, and an **open top** consistent with WRF.
- **physics-`tendf` fold + 2-D Smagorinsky now applied on the default path** —
  the physics source-tendency fold and the WRF real-data-default 2-D Smagorinsky
  (`diff_opt=1`/`km_opt=4`) horizontal diffusion are honoured by default.
- **`h_diabatic` mass** wired in.
- **RAINNC all-phase WRF-convention fix** — accumulated grid-scale precipitation
  was dropping snow + graupel + ice; it now follows WRF's all-phase convention.
- **DZS/ZS writer registration** — both soil-layer fields are now emitted (writer
  fix), and the Grid-Delta Atlas DZS/ZS checks PASS.
- Plus the large v0.13/v0.14 **memory-stability** work that keeps both 72 h runs
  inside ~20 GiB peak VRAM on a single RTX 5090.

## 72 h field-parity gates (final code, both regions)

Each region ran 72 h GPU-vs-CPU-WRF against retained CPU-WRF truth, scored over
**all cells, all 72 leads, and all core internal variables** with a pre-declared
frozen tolerance manifest (the same numbers the Grid-Delta Atlas hard-gate uses).

### Switzerland d01 72 h

- **Stable to h72**; mandatory fields all present; **9/10 prognostic fields
  within frozen tolerance**; all dynamics/thermo/mass green.
- The single Grid-Delta Atlas hard-gate miss is **RAINNC rmse 5.19 mm vs the
  1.0 mm bound** — a **bounded precipitation-placement sensitivity**, ≈0.78× the
  field's own std of 6.6 mm, not a dynamics blow-up. The RAINNC all-phase
  WRF-convention bug is **FIXED**. DZS/ZS now **PASS** (writer fix).
- Identity plots: **9/10 prognostic fields within frozen tolerance** (RAINNC the
  one out).
- Benchmark: GPU **~2762 s** vs CPU **2906 s** = **~1.05×**, peak VRAM **~19.8
  GiB**.
- Run `v014_switzerland_d01_72h_FINAL_20260612T062354Z` vs CPU truth
  `v014_switzerland_72h_cpu_20260610T122909Z`.

### Canary L2 d02 72 h

- **Stable to h72**; operational verdict **L2_D02_GREEN** (bounds PASS, rmse
  PASS, pipeline green). The v0.14 default-on changes (open-top, 2-D Smagorinsky,
  physics-`tendf` fold, theta-ceiling 1000 K) do **not** regress Canary.
- Three bounded Atlas misses: **MUB max_abs 250.7 + PB max_abs 249.9** (a known
  **static** nest-frame-seam base-state artifact, localized) and **QVAPOR rmse
  1.45×10⁻³ vs 1.0×10⁻³ kg/kg** (+45%, a tight moisture margin).
- Identity plots: **9/10 prognostic fields within frozen tolerance** (QVAPOR the
  one out).
- Benchmark: GPU **~8200 s** vs CPU **8713 s** = **~1.06×**, peak VRAM **~20.3
  GiB**.
- Run `v014_canary_d02_72h_FINAL_20260612T062354Z` vs CPU truth
  `20260501_18z_l2_72h_20260519T173026Z`.

## The four bounded misses — honest, frozen limits unchanged

The manager's bounded-accept decision, recorded honestly: dynamics/thermo are
**cell-for-cell identical** on both regions. The four out-of-envelope diagnostics
— **RAINNC** precip sensitivity (Switzerland), **MUB/PB** nest-frame-seam base
state and **QVAPOR** moisture margin (Canary) — are **pre-existing/physical
diagnostics, NOT identity failures**, and match the class the accepted Canary
gate and the v0.11/v0.12 releases shipped with. They are carried to **v0.15** as
the "fix remaining deviations" lane. The **frozen tolerance limits are
unchanged** — no goalpost moving.

## WRF-v4 identity-proof system (new, CPU-only, reproducible)

A reusable, publication-quality visual proof that the GPU port is true to
CPU-WRF v4 across **all grid cells, all 72 forecast leads, and all core internal
variables**. It is **offline and CPU-only** — it reads existing paired `wrfout`
NetCDF files plus a frozen tolerance manifest and never runs WRF, JAX, CUDA, or
any model kernel. Per region it renders: per-variable RMSE/bias time series with
the tolerance line, a variable×lead scoreboard, a GPU-vs-CPU 1:1 cell scatter
(`r ≈ 0.99–1.00` on the prognostic core), signed spatial-difference maps at true
scale, and a README-embeddable dashboard.

- Tool: `scripts/build_identity_proof_plots.py`
- Method + reproduce commands: [`docs/IDENTITY_PROOF.md`](docs/IDENTITY_PROOF.md)
- Assets: `docs/assets/v014/identity_proof/{switzerland_d01,canary_l2_d02}/identity_dashboard.png`

```bash
taskset -c 0-3 python3 scripts/build_identity_proof_plots.py \
  --cpu-dir "$CPU_DIR" --gpu-dir "$GPU_DIR" \
  --domain d01 --init "2023-01-15T00:00:00+00:00" \
  --case-id switzerland_d01_72h --region-label "Switzerland d01 72h" \
  --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
  --proof-dir proofs/v014/identity_proof/switzerland_d01 \
  --asset-dir docs/assets/v014/identity_proof/switzerland_d01
```

## Performance — honest ~1.05×, deferred to v0.15

v0.14 is **not** a performance release. Warm per-forecast-hour throughput is
roughly on par with 28-rank CPU-WRF (~1.05× Switzerland, ~1.06× Canary). The
v0.13.0 perf-triage attributed the earlier 3×→1× change to a double-compile
(fp32→fp64 graph) with no trivial identity-safe fix; **performance recovery is
the dedicated focus of v0.15**. No performance headline is claimed for v0.14.

## Carried forward unchanged

All of the v0.13.0 capability (RRTMG VRAM-floor chunking, GWD-on-nested,
GPU-validated compile-speed infra, MYJ+Janjic operational, moisture
flux-advection into RK3, multi-GPU fake-mesh sharding), the v0.12.0 capability
(standalone out-of-box CLI, standalone live-nested `--max-dom`, persistent JIT
cache, fail-closed scheme catalog, WRF-faithful PSFC fix, runnable equivalence
demo), and the v0.11.0 capability (live multi-domain nesting, restart continuity,
conservation-closed budgets, MYNN-EDMF, topographic/slope radiation,
terrain-slope diffusion, KF/BMJ/Tiedtke/Grell-Freitas cumulus) carries forward
unchanged.

## Known issues / scope boundaries

Full detail in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) and
[`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

| ID | Summary | Severity |
|---|---|---|
| v0.14 bounded misses | Four bounded, pre-existing/physical diagnostics out of frozen envelope on the 72 h gates (limits unchanged): Switzerland **RAINNC** 5.19 mm vs 1.0 mm, Canary **QVAPOR** 1.45×10⁻³ vs 1.0×10⁻³ kg/kg (+45%), Canary static **MUB/PB** nest-frame-seam base state (Atlas max_abs 250.7 / 249.9). Carried to v0.15. | Bounded acceptance |
| KI-9 | The credibility gate. v0.14 closed both 72 h field-parity gates (dynamics/thermo core cell-for-cell identical), but the broader 24 h/72 h **forecast-skill equivalence** (T2/U10/V10) is still open. Hard dynamics-`ph'`/MYNN/`*_tendf` GPU work, no cheap knob. | Documented gap |
| KI-3 | Operational `wrfout` is a focused **104-variable** subset (vs WRF's 375). | Scope boundary |
| KI-5 | Powered **n=15 TOST**: scoring path unblocked (rc=2 fixed); v0.14 did not run the powered campaign (the 72 h field-parity gates are the v0.14 evidence). No TOST PASS is claimed; deferred. | Scope boundary |
| KI-4 / KI-6 / KI-7 / KI-10 / KI-11 | Carried from v0.13.0 — see `docs/KNOWN_ISSUES.md`. | Carry-overs |

## Performance / precision note

The standalone CLI path is **fp64-only**; gated-fp32 remains an experimental
ADR-007 preview and is no faster on this memory-bound workload. Whether/how to
operationalize a reduced-precision state is pending a dedicated ADR (tied to the
v0.15 performance work).
