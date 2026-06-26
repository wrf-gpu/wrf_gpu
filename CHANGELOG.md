# Changelog

All notable changes to wrf_gpu are recorded here. This file is a concise index;
each release has full, honest release notes in `RELEASE_NOTES_v<version>.md`.
Versions follow a 0.x pre-1.0 line (the v1.0.0 target is a complete, validated
WRF v4 GPU port — see [`PROJECT_PLAN.md`](PROJECT_PLAN.md)).

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [0.21.0] - 2026-06-26

Stability + compile-cache-speed release. Priority order: **STABILITY > IDENTITY >
SPEED > MEMORY**. The fp64 default path stays byte-identical and warm forecast
throughput is unchanged from v0.20; the speed win is **compile / warm-start time**.
Full notes: [`RELEASE_NOTES_v0.21.0.md`](RELEASE_NOTES_v0.21.0.md).

### Added
- **AOT cheap-key cross-process warm-start of the FUSED cascade (default on).** After a
  one-time cold compile, a fresh process loads the compiled GPU executable from disk via
  a cheap metadata key (computed **without lowering**, scoped to the trace-import closure
  of the traced body, with the fused cascade's **edge geometry** — child
  weights/ratios/cadences/bdy-widths/count/order + namelist aux — folded into the key) and
  **skips the multi-tens-of-minutes re-lower** the old persistent cache still paid. MEASURED
  3-domain fused+AOT cold→warm gate (RTX 5090, 20240901): the **fused** `d01` + `fused/d02`
  blobs load `loaded=true source=aot_blob` cross-process with **0 re-lower** of `jit_fused`,
  **warm output byte-identical to cold** (`REF_COMPARE` equal, max_abs_diff 0), **steady
  s/step 1.20 ≤ the fused baseline ⇒ NO runtime regression**, warm peak host RSS **~10 GB**
  (vs ~21 GB cold), load in seconds, finite. **9-nest fused stress re-confirm (MEASURED,
  RTX 5090, 20240901):** both fused phases (`ed303`+`1db7`, ~1.7 GB each) load
  `source=aot_blob` cross-process with **0 `jit_fused` re-lower**, a **bounded K=2** phase
  set (no cached-call thrash, 0 cached-call-error), all nine domains finite; the warm-vs-jit
  bit-identity rides the **same fused executable path** the 3-dom `REF_COMPARE` already proved
  byte-identical. A fresh load is numerically inert (the key only
  *locates* the blob; the loaded executable is byte-identical to a cold compile);
  `GPUWRF_AOT_VERIFY=1` is a fail-closed lower-once + HLO-digest backstop (default off,
  quarantine on mismatch). Opt out of AOT with `GPUWRF_NESTED_AOT=0`.
- **Stability: dycore boundary mechanism fix.** An acoustic dry-mass-drain limiter
  plus positive `c2a`/`alt` conditioning fix the diagnosed steep-terrain failure
  mechanism. Identity-preserving on the integrated CPU baseline; the 9-nest Canary
  gate-case (max_dom=9) is now **finite past its step-67 divergence window (all nine
  domains, MEASURED)**.
- **Stability: default-on fail-fast finite detector.** `GPUWRF_FINITE_CHECK` guards
  the nested forecast path and reports the first non-finite prognostic
  `{domain, field, level, step, sim_time_s, index}`. Observational on finite states
  (does not mutate values); opt out with `GPUWRF_FINITE_CHECK=0`.
- **Stability: steep-terrain GPU regression gate.** Opt-in two-domain nested gate via
  `tests/test_v021_steep_terrain_stability_gate.py` (real `execute_nested_pipeline`
  path, both domains required finite).
- **Compile-cache safety.** Default persistent JAX cache directories are now
  version/backend-keyed; `GPUWRF_CACHE` roots get `jit/<version-tag>`, so a stale
  older-release cache is never mistaken for a warm one.
- **Autotune cache.** XLA autotune cache is default-on when the compile cache is on,
  with fail-open subprocess flag probing; prewarm `info`/`pack`/`unpack` is available.
- **Operational hardening.** Nested namelist scalar coercion is more robust;
  grid-scaled `GPUWRF_MIN_FREE_VRAM_GIB` preflight guidance; the next large-GPU
  activation path uses the Python nested API runner, `cuda_async`, no preallocation,
  version-keyed cache/prewarm, and read-back-before-delete output hygiene.

### Changed
- **Nest runtime default stays FUSED; AOT warm-start bolted onto the fused executable.**
  The fused single-module cascade remains the runtime (byte-identical to v0.20, **zero
  throughput change**); AOT now serializes/loads that fused executable for warm-start.
  **De-fuse** (per-domain compile — the same bit-identical eager code path,
  `GPUWRF_NESTED_DEFUSE_COMPILE=1` / `GPUWRF_NESTED_FUSE=0`) remains an **opt-in
  low-host-RAM compile lever** (9-nest ~20–27 GB VmHWM vs ~60 GB fused) **with a documented
  runtime cost**: it was briefly trialed as the default and **reverted** after the canary
  benchmark measured **+18.8 % s/step** (the 9-nest de-fuse was host-bound, GPU ~0 %).
  De-fuse cold (~70–75 min) is also slower than fused — a RAM-for-WALL trade, not a faster
  cold compile. Optional parallel prewarm for the de-fuse path via
  `GPUWRF_NESTED_PARALLEL_COMPILE=N` (`=0` opts out).
- Release framing is **STABILITY > IDENTITY > SPEED > MEMORY**; the speed win is
  compile/warm-start time, not warm forecast throughput.
- Full CPU suite A/B (per-file-isolated, baseline vs edited): **zero new failures** vs
  the v0.20.2 known-red baseline. 3-domain de-fuse CPU tol-match worst T2 2.96 K /
  1.03%.

### Carried limitations
- **Most-extreme 1 km Mont-Blanc (~1042 m/cell) terrain not fully stabilized.** The
  dycore fix stabilizes the standard 9-nest Canary gate-case but relocates the failure
  on the extreme case → deep boundary-stability fix is **v0.21.1**.
- **#123 long-horizon 9-nest GPU-VRAM OOM mitigated-not-eliminated.** In de-fuse mode
  the single-card 32 GB run can still OOM around the ~90 min integration horizon
  (mitigated by the RRTMG-transient cap + fail-closed preflight). For VRAM-bound long
  integration use `GPUWRF_NESTED_FUSE=1`. B200 / fp32 / VRAM work is v0.21.1+.
- The ≥1 h-finite + all-fields CPU-match Canary gate is a **local** gate, not a default
  v0.21.0 claim. No new physics features (per `proofs/v021/WRF_V4_FEATURE_AUDIT.md`).

### Follow-ups (carried, non-blocking)
- AOT cheap-key hardening: `_walk`-repr robustness and an import-time env scanner.

## [0.20.2]

### Added
- **#122 training subset — cloud-validation fields** (output-only, opt-in, default byte-identical,
  cache-neutral): `MINIMAL_TRAINING_SET` gains **OLR** (TOA outgoing LW = cloud-top, MSG-observable),
  **RAINC** (convective precip; completes `RAINNC` for the 3 km cumulus parent), **SWDNB** (RRTMG
  surface downwelling SW). Subset is now 39 vars (was 36). No HLO/compile change. `tests/
  test_v0201_training_output_subset.py` 10/10 incl. default-byte-identity.

## [0.20.1] — reliability + I/O readiness + honesty refresh (default path byte-identical)

A **bit-identical-safe** patch release on top of v0.20.0: it hardens the nested
GPU path against the OOM class (co-resident headroom solved; solo fragmentation
mitigated and carried), makes the nested compile cache hit across forecast dates,
adds an opt-in compact training-output mode, lands the paid-B200 I/O readiness
tooling, and applies a no-fabrication honesty refresh to the public claims. The
fp64 default path stays byte-for-byte unchanged. Full notes:
[`RELEASE_NOTES_v0.20.1.md`](RELEASE_NOTES_v0.20.1.md).

- **#114 — cross-date warm NEST compile cache (bit-identical). CONFIRMED.** The
  nested path carried a residual baked date scalar in the pytree treedef; it is now
  wrapped in a `_DateClockAux` holder so the **treedef is date-invariant** and a
  new/leap-year date no longer re-traces or re-lowers the fused nest, removing the
  **~50-min per-date nested recompile**. Bit-identical on the default nested path.
  **GPU gate PASSED:** cold `DATE_A` GREEN (cache_delta ≈ 6497, 26 `wrfout`) → warm
  different-date `DATE_B` GREEN (cache_delta = 2, 26 `wrfout`); CPU treedef
  date-invariance tests 9/9 (`tests/test_operational_namelist_cache_key.py`).
  **Honest:** warm removes the recompile but still pays a fused-module load+link
  (~19 min × 2 ≈ 38 min) → ~30 min net saving per new date, **not "instant"**.
  *[Gate driver `scripts/gate_n114_n122.py`; CPU tests
  `tests/test_operational_namelist_cache_key.py` (9/9); cache_delta verdict
  in the v0.20.1 release-prep acceptance matrix.]*
- **#122 — opt-in compact training-ready nest output.** `GPUWRF_TRAINING_OUTPUT_SUBSET`
  (opt-in) writes a **36-variable MINIMAL** training set (+ mandatory coords, 44 names
  total) with **zlib-lossless** compression. **Default OFF → output byte-identical to
  the legacy writer.** Validated on a real GPU nest run; CPU 10/10
  (`tests/test_v0201_training_output_subset.py`).
  *[CPU tests `tests/test_v0201_training_output_subset.py` (10/10).]*
- **#123 — GPU OOM hardening (nested path); two modes, one solved + one mitigated.**
  (a) **Co-resident headroom → SOLVED.** Launch-time nested **preflight** fail-closes
  with **exit 75 before** the ~50-min compile when the GPU lock or **card-relative
  free VRAM** gate fails (`max(GPUWRF_MIN_FREE_VRAM_GIB=24,
  GPUWRF_MIN_FREE_VRAM_FRACTION×total)`, default fraction 0.50; multi-GPU select via
  `CUDA_VISIBLE_DEVICES`; `--force-gpu-run` bypass; launch-time-only guard).
  GPU-validated fail-closed + happy-path; CPU 9/9.
  (b) **Solo cuda_async fragmentation → MITIGATED, not fixed.** RRTMG column-tile cap
  default **2048 → 1024, bit-identical** (column-local; CPU proof `max_abs = 0.0`);
  GPU-measured largest allocation **0.432 → 0.271 GiB (−37 %)** + a shipped RRTMG
  transient reproducer. **Carried limitation:** solo fragmentation can still OOM the
  full fp64 nest — **no OOM-proof / fp32-nest / 24 h large-nest claim.**
  *[Commits `b1c44524`, `4c9588db`; `proofs/v013/rrtmg_column_tile.json`
  (`max_abs = 0.0`), `proofs/v020/oom_hardening/rrtmg_transient_cpu_smoke.json`,
  reproducer `scripts/rrtmg_transient_reproducer.py`; GPU A/B largest-alloc
  (0.432→0.271 GiB) in the v0.20.1 release-prep acceptance matrix.]*
- **S2 — paid-B200 I/O-readiness tooling (CPU-only).** Manifest + WRF-dimension
  validator (rejects 897 / old 181 mini-nest, accepts 898/369 @ ratio 3, streamed
  SHA-256) + block drain/backpressure/resume/stop-pull with **S3
  read-back-verify-before-delete**. 22/22 CPU tests green, zero GPU imports.
  **Carried:** tested on synthetic/local dry-runs only — the **real S3 path is not
  yet exercised**, so the next paid pod needs a pre-pod read-back/delete dry-run
  against a disposable prefix.
  *[Commits `56fb28b1`, `3c0933b2`; `tests/test_b200_manifest_validator.py`,
  `tests/test_b200_drain.py`.]*
- **Honesty refresh (no fabricated numbers; S3 pass).** Identity **TH2 step-1
  divergence is a metric artifact, not a bug** (I1: recomputed TH2 reproduces the
  stored field to ~10⁻³ K on both runs; its absolute error equals T2's; the
  collapse is low-variance Pearson over the shared near-surface T2 offset, not a
  TH2 defect). Report **nRMSE alongside r** (I2). Annotate the **d01 mean-r
  2-short-lead coverage** caveat (~+0.018 surface-wind inflation, I3). **#119
  closed:** "Switzerland worse than Canary" is a presentation/metric artifact — two
  hash-verified analyses converged that Swiss surface fields are comparable/better
  and no model/port bug or paper claim is falsified (paper-figure provenance fix is
  paper-track, not this release).
  *[Source: S3 release-prep honesty pass + the v0.20.1 release-prep Pearson
  consolidation (two hash-verified analyses); numbers recomputed from the paired
  GPU/CPU `wrfout` trees.]*

## [0.20.0] — correctness + stability + capability + reliability (modest measured nest speedup)

A **bit-identical-safe** release: the fp64 default path is byte-for-byte unchanged,
and v0.20 adds a modest measured nest speedup, an opt-in fp32 capability mode, and a
compile cache that just works across runs and across forecast dates. v0.20.0
branches off **0.19.1** (0.19.2 was never shipped). This release:

- **Makes the all-7 nested fast path faster than v0.19 and CPU — byte-identically.**
  The default fused all-7-island, 9-domain run measures **~668 s/forecast-hour warm**
  (range 645–680) on the reference GPU against **713 s/forecast-hour for v0.19** and
  the canonical **12-rank CPU-WRF baseline at 1020 s/forecast-hour** — **~1.07× faster
  than v0.19** and **~1.53× faster than CPU**. The output is **byte-identical to
  v0.19 (1926/1926 vars, maxΔ=0.000e+00)**: the gain comes entirely from a
  **numerics-free CUDA stream-ordered allocator** (`cuda_async`, now the default),
  which carries essentially the whole improvement (~+44 s/forecast-hour); the
  async-output and host-RAM-event-tail-guard levers are stability/neutral by design.
  Peak VRAM is **122 MiB leaner** than v0.19 and stays flat over 12 output groups
  (1 km fit, no OOM). The nest is host-bound (~7–8% GPU duty) and the headline is a
  **range, not a point** (measured under live CPU contention, ±2–3% noise); the
  identity / fit / VRAM gates are robust regardless. Larger structural multipliers
  (acoustic-substep config, fused megakernel, fp32-relaxed) are an explicit
  **future wave, not in v0.20**.
  *[MEASURED: `proofs/v020/lowhang/COMBINED_SPEEDUP.md` §4–§8.]*

- **Keeps the fp64 default byte-identical and same-speed.** The fp64_default GPU
  all-7 9-domain output is **963/963 vars maxΔ=0.000e+00, byte-identical** across
  all 9 domain files; warm fp64 forecast-only time is **HEAD ≈ baseline** (within
  noise). v0.20 adds no numerics risk on the default path.
  *[MEASURED: `proofs/v020/fp32_integration/FP32_INTEGRATION_REPORT.md` §4.1/§4.1b.]*

- **Adds an opt-in fp32 mixed-precision mode for capability + VRAM (not speed).**
  `GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020` runs a
  perturbation-authoritative fp32 acoustic path (fp64 totals). It cuts whole-run
  VRAM **−14.4%** (aggressive) and extends full-physics cell capability **~1.16×**
  (fits 700² where fp64 caps 650²; in a dynamics-only stress fp64 OOMs at 1M
  columns where fp32 fits). On the single RTX 5090 it is **NOT a speedup** —
  fp32/fp64 throughput-ceiling ratio **≈0.91 (≈1, not ≈2)**, and there is no
  peak-VRAM win on small single domains (radiation-transient-bounded below ~384²).
  **Honest scope:** fp32 tolerance is checked at the **1 h lead** (19/19 fields
  green) — real but **not stringent**; the **24–120 h skill gate is future work,
  out of v0.20 scope**. fp32 is strictly opt-in; `fp64_default` remains the default.
  *[MEASURED VRAM/capability + 1h tolerance: FP32_INTEGRATION_REPORT.md §4.2.
  INCONCLUSIVE single-card speed: `proofs/v020/benchmark/T2T3_REPORT.md` R∞
  ratio ≈0.91.]*

- **Makes the compile cache hit across forecast dates (#91), zero config.** The
  persistent per-user on-disk JIT cache (default on since v0.12.0) previously baked
  the forecast date into the radiation/solar HLO, so every new date missed the cache
  and recompiled. The date is now a **runtime argument** (`clock_base`), so the
  **lowered HLO is identical across dates** (sha256 identical across 3 dates,
  including a leap year) and re-running on a new or leap-year date is a **warm cache
  hit with 0 new cache entries**. The **default RRTMG path stays bit-identical**
  (traced vs baked clock: **64/64 State leaves byte-identical**, 3 steps, RRTMG
  SW+LW). One documented non-default residual: GSFC SW (`ra_sw=2`) keeps a seasonal
  ozone-band index, so its HLO still date-varies.
  *[MEASURED: `proofs/v020/julday_cache/JULDAY_CACHE_FIX_REPORT.md`.]*

- **Re-certifies single-domain scaling honestly.** A parametrized single-domain
  scaling study (Swiss base, tiled, dt=10s) shows throughput saturating at a ceiling
  of **~9.6e6 cells/s (fp32) / ~1.06e7 cells/s (fp64)** by ~384²; fp32 **fits 512²
  (11.5 M cells) at 16.2 GB where fp64 OOMs** (capability), with **no single-domain
  speed win** from fp32 (t-ratio 0.86–1.07). On a tiny single-domain 129² grid the
  GPU is **~2.3× slower** than 24-rank CPU-WRF (host/launch-bound); GPU vs CPU
  identity at 1 h is tight (T2 RMSE 0.79 K, U10 0.56 m/s, PSFC corr 0.999996). The
  harness is parametrized to lift to H200/GB300 (those results are **PROJECTED**).
  *[MEASURED: `proofs/v020/benchmark/T2T3_REPORT.md` G-series + Swiss-CPU-match +
  identity.]*

- **All-7 24 h GPU-vs-CPU identity (fp64 vs CPU-WRF, 9 domains):** the
  dynamics/thermo **core is EXCELLENT** — **T corr 0.9999 (RMSE 0.69 K), PH and
  PSFC corr 0.9997, U 0.991, V 0.968, QVAPOR 0.964** (mean correlation across the 9
  domains). The **surface diagnostics are looser** (the most
  parameterization-sensitive fields): **T2 0.944 (RMSE 0.78 K), TH2 0.878, U10
  0.855, V10 0.852** (mean corr); on the **inner 1 km Alpine nests d06/d07 the 10 m
  winds spread to corr ~0.48–0.65 / RMSE ~4–5 m/s** — honestly the expected
  most-sensitive field on complex terrain, **byte-identical to the validated v0.19
  output (this is NOT a v0.20 regression)**, with divergence growing with lead time.
  Logged as a v0.20.1 characterization item (#119).
  *[MEASURED: `proofs/v020/validation/identity/identity_metrics.json` — 24 h,
  init 2026-02-14 18Z, fp64 GPU vs CPU-WRF, 9 domains, 10 fields.]*

- **Keeps opt-outs explicit.** `GPUWRF_BITWISE=1` or `GPUWRF_NESTED_FUSE=0` selects
  the eager non-fused bitwise/debug path; `GPUWRF_ALLOCATOR=platform` restores the
  pre-v0.20 synchronous allocator; `fp64_default` (default) is the byte-identical
  precision mode. See the v0.20.0 release notes (archived at tag v0.20.0).

## v0.19.1 — 24-hour VRAM-stability fix (nested fast path)

- Fix a GPU memory leak in the default fused nested cascade that blocked sustained
  24-hour all-7-island `max_dom=9` runs. A self-referential recursive closure
  (`integrate` in `runtime/domain_tree.py`) pinned one per-domain carry set per
  output group via a reference cycle that the cyclic GC does not reclaim during
  the device-bound loop, so VRAM grew ~1 GB/output group → CUDA OOM at ~9.7 h.
  Break the closure cycle (memory-only; no kernel/numerics/throughput change).
  Validated on the canonical all-7 `max_dom=9` real case: VRAM flat over 9 output
  groups, warm 721 s/forecast-hour (= v0.19.0, no regression), all 9 domains
  tolerance-green. See `RELEASE_NOTES_v0.19.1.md`.

## [0.19.0] — fast all-7 nested fusion + terrain-blend fidelity

Performance and fidelity release for the all-7-island `max_dom=9` nested path.
This release ships the tested tree `7a84e519`:

- **Makes fused nested cascade the default fast path.** The all-7-island,
  9-domain case now measures **713 s/forecast-hour warm** on the reference GPU
  against the canonical **12-rank CPU-WRF baseline at 1020 s/forecast-hour** —
  **1.43x faster than CPU**. The best warm segment measured **683
  s/forecast-hour**. The first fused run still pays the one-time XLA megacompile
  (cold segment **7348 s/forecast-hour**, about **41 min wall**) and is cached
  after that.
- **Restores the fast `_advance_chunk` body.** The v0.18.0 `fori_loop` to `scan`
  rewrite made the warm leaf body much slower; v0.19.0 returns to the traced-count
  `fori_loop` form while preserving the v0.18.3 `max_dom=9` compile-bounded fix.
- **Fixes live-nest terrain/base-state initialization.** Nested runtime
  initialization now matches WRF's `blend_terrain` ordering for terrain and base
  state, eliminating the d02-d09 HGT/MUB/PB/PHB red-field class that blocked the
  speed gate.
- **Ships the all-fields release gate green.** The GPU run writes all expected
  `wrfout` files for all 9 domains, all fields are finite, and the established
  grid-delta atlas comparator reports **PASS on all 9 domains** against the
  frozen v0.14 tolerance manifest (**102 compared numeric fields/domain, 0
  tolerance failures**).
- **Keeps opt-outs explicit.** `GPUWRF_BITWISE=1` or `GPUWRF_NESTED_FUSE=0`
  selects the eager non-fused path for bit-identical/debug comparisons. Fused
  mode is tolerance-green vs CPU-WRF, not bitwise-identical to the eager path.

## [0.18.3] — max_dom=9 compile fix + nested history_interval cadence fix (bit-identical)

Two bugfixes; default forecast numerics unchanged (bit-identical: 26/26 `wrfout`
fields exact, `max_abs_diff 0.0`). This release:

- **Fixes the `--max-dom 9` (all-7-island nest) compile blowup.** Thompson
  sedimentation/fall-speed scans materialized static `s64[nz]` vertical-index
  arrays (`jnp.arange(nz)`); across the 9 distinct domain shapes XLA folded them
  unbounded, so `jit__advance_chunk` compiled forever and never integrated
  (HLO-confirmed: `proofs/v018/maxdom9_fix/report.md`). The scans now thread a
  scalar `int32` counter in the carry (`scan(..., None, length=nz)`) — identical
  iteration order/values, no static index vector. All 9 domain-shape compiles now
  complete **bounded** (≤409 s cold, ≤22 s warm cache-hit), reach stable
  integration (~85 % GPU util), and write output.
- **Fixes nested output ignoring `history_interval`.** The nested pipeline was
  hardcoded to hourly output; it now honors the per-domain namelist
  `history_interval` (valid time / `XTIME` from `own_step·dt`). Hourly gates are
  unchanged (`ceil(3600/dt)` equals the old `round(3600/dt)` for the v0.18.2
  timesteps). Proven: a d03 `wrfout` at forecast **+5 min** (`XTIME=5.0`, 106
  fields finite) under `history_interval=5`.

## [0.18.2] — 1 km nested VRAM-efficiency fix (bit-identical)

Memory-efficiency patch over 0.18.1. Default numerics are unchanged and the
validated outputs are bit-identical on the default path. This release:

- **Makes the AC1_FIT 9/3/1 nested 1 km all-island case fit one RTX 5090**:
  the prior path OOMed on the 32 GiB card; the fixed warm-cache 1-forecast-hour
  run passes at **18.1 GiB peak VRAM**.
- **Realizes the algorithmic VRAM lever bit-identically** via RRTMG radiation
  column-tile defaults **16384→2048** plus tiled MYNN cold-start BouLac
  initialization; 26/26 `wrfout` fields compare exact (`max_abs_diff=0.0`) and
  MYNN cold-start `qke`/`pblh` diffs are 0.0.
- **Clarifies utilization honestly**: warm steady-state nested 1 km execution is
  ~85–88% GPU utilization; the lower full-run aggregate is the one-time
  load/cold-JIT prefix, not steady-state idleness.
- **Restores `data/fixtures` runtime tables** for Thompson aerosol and cold
  collection so Thompson cases import from a source-only tree.

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
table and the per-version proof archives (`proofs/v01{5,7,8}/`); standalone
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
  framed development-log archive in the source tree.

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
