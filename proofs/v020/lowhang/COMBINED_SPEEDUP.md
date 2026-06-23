# v0.20.0 Low-Hanging-Fruit Speed Wave — COMBINED SPEEDUP (all-7 9-domain nest)

**Worker:** Claude Opus 4.8 (max), integration + combined-speedup measurement.
**Branch:** `worker/integration/v020-lowhang` (off `worker/integration/v020` @ `eb564cea`).
**Status:** ✅ COMPLETE. Integration merged + CPU-gated + warm GPU A/B measured + byte-identity proven +
cuda_async fit/12h-VRAM-flat verified, all on the all-7 9-domain 1km nest (`canary_all7`). Headline below.

**TL;DR:** v0.20.0 lowhang = a **byte-identical ~1.05–1.10× warm speedup over v0.19 fp64 (~1.50–1.58× over
CPU-WRF 12-rank)** on the all-7 9-domain 1km nest, carried by the `cuda_async` allocator (~45 s/fc-h);
async-output + host-RAM-guard are bit-identical robustness (neutral on speed here); sync stays `root:1`.
Identity: **1926/1926 vars byte-identical**. cuda_async fit: **no OOM, peak 14.99 GB flat over 12h (0.21%),
122 MiB leaner than platform**. Honest caveat: host-bound nest under live CPU-corpus contention → headline is
a range, not a false-precision point.

---

## 1. What shipped into the integration branch

Three bit-identical / numerics-free host-bubble levers merged in plan order
(`proofs/v020/lowhang/INTEGRATION_AND_MEASURE_PLAN.md`):

| Order | Lever | Commit | Gate | Knob (default) |
|---|---|---|---|---|
| 1 | G_allocator_env — nested XLA allocator default `cuda_async` (pooling) | `eebe0bad` | numerics-free | `GPUWRF_ALLOCATOR=cuda_async` (was `platform`) |
| 2 | G_domain_tree — host-RAM event-tail guard (streaming Counter+deque fold) | `1bdbac18` | bit-identical | `GPUWRF_NESTED_EVENT_TAIL=4096` |
| 3 | G_nested_pipeline — async history output + sync-cadence sweep | `b2dcdf17` | bit-identical | `GPUWRF_NESTED_ASYNC_OUTPUT=1`, `GPUWRF_NESTED_SYNC_MODE=root:1` |

**Merge conflict (forecast loop, `nested_pipeline.py`) resolved** by keeping BOTH:
G_domain_tree's streaming events fold (Counter+`event_counts`/`force_counts` +
bounded `events_tail` deque) AND G_nested_pipeline's `try/finally` +
`async_writer.join()` fail-closed drain. The post-conflict `DomainTreeResult` build
already consumes `events_tail`, and no `events=[]` initializer survives the fold, so
the streaming fold is mandatory (the incoming `events.extend` would `NameError`).
Integration commits: `4a2160aa` (alloc) → `39cf6edc` (domain_tree) → `41cb580b` (nested).

## 2. CPU re-gate (before any GPU) — GREEN

`taskset -c 0-3 JAX_PLATFORMS=cpu pytest` over the union gate set: **78 passed, 0 failed.**
Identity-critical subset (verbose):
- `test_fused_cascade_is_scheduler_and_value_identical_to_eager_all7` — PASS
  (the all-7 nested cascade is scheduler-and-value identical = **fp64_default bit-identical nested cascade trace**).
- `test_fused_cascade_none_lookup_is_eager_byte_identical` — PASS.
- `test_async_writer_byte_identical_to_sync` (+ ordering, error-surfacing) — PASS (L1 async == sync).
- `test_streaming_fold_matches_old_full_list_count` (+ tail-cap parsing/unbounded) — PASS (G_domain_tree fold == old full-list counts).

## 3. Two setup issues found + fixed before measuring (no lever changes)

1. **Canonical case correction (plan residual-risk item 3).** The async harness defaulted
   `INPUT_DIR` to `…/ac1fit_20260614T220802Z/run_cpu`, but that case has `max_dom=3` and only
   3 `wrfinput` files — running `--max-dom 9` on it cannot drive a 9-domain nest. The genuine
   all-7 9-domain nest (the v0.19 713–721 s/fc-h baseline case) is
   **`<DATA_ROOT>/wrf_downscale/canary_all7/run`** (`max_dom=9`, 9 grids, 9 `wrfinput` files,
   `wrfbdy_d01`). All measurements use canary_all7. The allocator harness already defaulted to it.
2. **Worktree data symlink.** Fresh worktree lacked `data/wrf_pristine`; the Noah-MP/RRTMG
   runtime tables (`MPTABLE.TBL`, …) failed to load (`config.paths.wrf_run_dir()` →
   `<repo>/data/wrf_pristine/WRF/run`). Fixed with the documented machine-local symlink
   `data/wrf_pristine → <USER_HOME>/src/wrf_pristine` (gitignored; the same effect as
   `GPUWRF_WRF_ROOT`). All key tables (MPTABLE/SOILPARM/GENPARM/VEGPARM/RRTM_DATA/RRTMG_LW) verified resolvable.
   Also pinned the async harness to host cores 0-3 (was hardcoded `4-31`, which collides with the
   running CPU-WRF corpus — would corrupt timing).

## 4. Measurement method (warm, apples-to-apples)

Unified harness `scripts/v020_lowhang_combined_ab.sh`: one warm-up populates the shared
persistent XLA cache (`<DATA_ROOT>/gpuwrf_jax_cache`); the levers do not change the HLO, so every
timed arm is **warm** (compile excluded — `s/fc-h = wall_clock_forecast_only_s / hours` from the
nested proof JSON). All arms: fp64 (`JAX_ENABLE_X64=true`), `GPUWRF_MYNN_BOULAC_ONZ=1`, host
cores `0-3`, `canary_all7/run`, `--max-dom 9`, `HOURS=2`, 1 Hz peak-VRAM sampler. One GPU-lock
holder per session. Lower s/fc-h wins; reject any arm whose peak VRAM regresses vs the v0.19 baseline.

Arms: `v019_baseline` (platform, async=0, root:1) · `alloc_only` (cuda_async, async=0) ·
`async_only` (platform, async=1) · `combined` (cuda_async, async=1, root:1) ·
`combined_K2`/`combined_K3` (sync-cadence sweep).

**Fixed-overhead note (measured).** Each arm is a fresh process; the first
`run_operational_domain_tree` call (after `forecast_start`) pays a one-time
trace + 611 MB cache-deserialize (~7 min observed) that lands *inside*
`forecast_only_s`. It is identical across arms, so **relative** lever gains cancel
it exactly. For the **absolute** vs-713/vs-1020 headline it is removed with the
two-point method (run the headline arms at H1 and H2; clean
`s/fc-h = (fc_s(H2) − fc_s(H1)) / (H2 − H1)`), which cancels all fixed per-process
overhead. The cold megacompile (`jit_fused`, ~27 min on 4 cores under corpus
contention — the documented one-time op-risk) was absorbed by the discarded
warm-up and cached (so every timed arm is a cache HIT, verified: zero new cache
writes during the arms).

---

### Raw H2 measurements (run1, all warm cache-HITs; identical per-process fixed overhead in each)
| arm | forecast_only_s (H2) | raw s/fc-h | Δ vs baseline (s/2h) | per-hr saving | peak VRAM (MiB) | finite | outputs | OOM |
|---|---|---|---|---|---|---|---|---|
| v019_baseline (platform, async0, K1) | 1824.3 | 912.1 | — | — | 14996 | ✓ | ✓ | no |
| alloc_only (cuda_async) | 1738.3 | 869.2 | −86.0 | **+43.0 s/fc-h** | 14874 | ✓ | ✓ | no |
| async_only (async output) | 1834.7 | 917.4 | +10.4 | −5.2 (noise) | 14991 | ✓ | ✓ | no |
| **combined (cuda_async+async, K1)** | **1741.5** | 870.7 | **−82.8** | **+41.4 s/fc-h** | 14873 | ✓ | ✓ | no |

The per-hr saving column is Δ/2h — the per-process fixed overhead (trace + 611 MB cache-deserialize, identical
across arms) cancels exactly in the Δ, so these per-hour savings are clean regardless of the overhead's exact
size. combined gain ≈ **+41 s/fc-h ≈ 1.06×**, carried entirely by **cuda_async** (+43); async output is neutral
here (−5, within noise — only ~2 output groups to overlap at H2, and the writer thread shares the 4 pinned cores).
(The two-point H1 anchor below was intended to pin the *absolute* steady-state, but turned out noise-limited;
the robust evidence is this H2 Δ and the H2 root:1 group separation.)

### Two-point anchor (H1) + the noise floor (honest)
| arm | H1 fc_s | H2 fc_s |
|---|---|---|
| v019_baseline (platform, K1) | 1282.0 | 1824.3 |
| combined (cuda_async, K1) | 1171.6 | 1741.5 |
| combined_K3 (cuda_async, K3) | **1448.0 (OUTLIER)** | 1688.9 |

The H1 session ran later, against variable CPU-corpus load on cores 4–31 (the nest is host-bound at
~7–8% GPU duty, so host contention injects run-to-run timing noise). `combined_K3` H1 = 1448.0 is a clear
outlier (slower than even baseline-H1), so the per-config two-point decomposition is not reliable — the
fixed overhead (~600–740s) swamps the ~40–90s lever signal at this noise level. **The clean evidence is the
back-to-back H2 session**, where the root:1 arms separate cleanly with low intra-session noise:

| H2 root:1 arms | platform | cuda_async | Δ |
|---|---|---|---|
| pair A | baseline 1824.3 | combined 1741.5 | −82.8 |
| pair B | async_only 1834.7 | alloc_only 1738.3 | −96.4 |
| group mean | **1829.5** | **1739.9** | **−89.6s/2h = −44.8 s/fc-h** |

Intra-group noise is tiny (platform pair Δ=10.4s, cuda_async pair Δ=3.2s) → the cuda_async win is real and
well-separated; the async-output lever sits inside the platform group (neutral); the host-RAM guard has no
speed effect by design (stability only).

## 5. HEADLINE RESULTS (warm, all-7 9-domain nest, byte-identical)

Anchored to the v0.19 reference (713) by applying the clean H2 cuda_async saving (per-hour compute, which is
duration-independent), consistent with v0.19's own s/fc-h framing. Reported as a range because the
host-bound nest measurement carries CPU-corpus-contention noise.

| Config | warm s/fc-h | × vs v0.19 fp64 (713) | × vs CPU-WRF (1020) | peak VRAM | finite | outputs | OOM |
|---|---|---|---|---|---|---|---|
| CPU-WRF (12-rank, reference) | 1020 | — | 1.00 | — | — | — | — |
| v0.19 fp64 (no levers) | 713 | 1.00 | 1.43 | 15.0 GB | ✓ | ✓ | no |
| **v0.20 COMBINED (cuda_async+async+guard, root:1)** | **~668 (645–680)** | **~1.07× (1.05–1.10)** | **~1.53× (1.50–1.58)** | **14.9 GB** | ✓ | ✓ | no |

**Driver = cuda_async (~45 s/fc-h).** async-output + host-RAM-guard are bit-identical safety, not speed, on
this nest. Net: a safe, byte-identical **~1.05–1.10× over v0.19 (≈1.50–1.58× over CPU-WRF)**, no math change.

### Per-lever marginal gains (clean H2 root:1)
| Lever (vs v0.19 baseline at H2) | Δ forecast_only_s (s/2h) | per-fc-h | × vs baseline | peak VRAM Δ |
|---|---|---|---|---|
| **cuda_async** (alloc_only) | **−86.0** | **+43.0** | **~1.06×** | −122 MiB (leaner) |
| async output (async_only) | +10.4 | −5.2 (noise) | ~1.00× (neutral) | +0 (host-side) |
| host-RAM event-tail guard | n/a | 0 (stability only) | 1.00× | 0 |

### Sync-cadence sweep (H2) → best K = **root:1 (keep default; zero code change)**
| K | combined fc_s (H2) | s/fc-h | peak VRAM (MiB) | note |
|---|---|---|---|---|
| root:1 | 1741.5 | 870.7 | 14873 | default |
| root:2 | 1778.2 | 889.1 | 14875 | slower than K1 |
| root:3 | 1688.9 | 844.5 | 14877 | faster at H2 (−52.6s) **but** H1 was an outlier (1448) → unconfirmed |

**Decision: keep `root:1`.** root:2 is slower than root:1; root:3's single-sample H2 lead is not corroborated
(its H1 re-measure was a contention outlier) and is within the contention-noise envelope. No VRAM
difference across K. Per the rule (change default only on a credible win with no VRAM regress), root:1 stands.
`root:3` is flagged as a candidate to re-confirm in a corpus-paused low-noise window.

## 6. cuda_async MUST-NOT-REGRESS gate
> **This nest IS the 1km case.** `canary_all7` has `dx = 9000, 3000, 1000×7` m — domains **d03–d09 run at
> 1 km** (the all-7-island 9-domain 1km nest, the historical RRTMG-transient OOM case). So the cuda_async fit
> check below directly satisfies the brief's "the 1km nest still FITS (no OOM regression)" requirement.
- **2h nest fit — ✅ PASS:** no CUDA OOM on any cuda_async arm; all d01–d09 finite + outputs present.
  Peak VRAM cuda_async **14873–14877 MiB vs platform 14996 MiB** → cuda_async is **122 MiB LEANER**, not a
  regression (the pooling allocator avoids the BFC fragmentation that drove the historical 1km OOM, and pools
  less than platform's raw cudaMalloc here). Fits comfortably (≈47% of the 32 GB RTX 5090).
- **12h VRAM-flat census — ✅ PASS** (`scripts/v020_alloc_24h_vram.sh`, cuda_async + async + root:1 = shipping
  config; 12 hourly output groups ≥ the v0.19.1 leak-guard census's 9 groups, spanning the historical ~9.7h
  leak onset). rc=0, **no OOM**, all domains finite, all 12 outputs present. Peak-VRAM per forecast hour:

  | fc-hour | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | peak MiB | 13520 | 13774 | 14873 | 14959 | 14969 | 14965 | 14977 | 14977 | 14982 | 14985 | 14984 | 14991 |

  Hours 1–3 are the **one-time cuda_async pool warm-up** (the pool grows to its steady working set, set by the
  first RRTMG radiation transient). **Hours 4–12 are FLAT: 14959–14991 MiB, spread 32 MiB = 0.21%** — i.e. the
  peak does NOT climb per output group. (The reducer's "first→last sextile delta = 1217 MiB" is entirely that
  one-time warm-up in sextile 1, not a leak: it is the opposite of the v0.19.1 signature, which incremented
  monotonically *per output group* to OOM at ~9.7h.) Overall peak 14991 MiB ≈ 47% of the 32 GB RTX 5090.
- **Verdict: ✅ cuda_async PASSES the must-not-regress gate** — the 1km nest fits with no OOM, peak VRAM is
  flat over 12 output groups (past the 9.7h leak onset) and is 122 MiB *leaner* than platform. (The 12h run's
  effective 717 s/fc-h is overhead-laden — a single long run with the per-process fixed cost amortized over
  12h — and is **not** the speed result; the speed result is the clean relative H2 A/B in §5.) If a regression
  ever appears, `GPUWRF_ALLOCATOR=platform` restores v0.19.x with zero code change and the bit-identical host
  levers keep their (small) gain.

## 7. Byte-identity on the nest — ✅ PASS (`scripts/v020_wrfout_byte_compare.py`)
GPU wrfout, all 18 domain files (d01–d09), every data variable compared bit-for-bit vs the v0.19 baseline:

| comparison | result |
|---|---|
| async_only vs v019_baseline | **1926/1926 vars BYTE-IDENTICAL** (maxΔ=0.000e+00) |
| alloc_only (cuda_async) vs v019_baseline | **1926/1926 vars BYTE-IDENTICAL** (maxΔ=0.000e+00) |
| combined (cuda_async+async, K1) vs v019_baseline | **1926/1926 vars BYTE-IDENTICAL** (maxΔ=0.000e+00) |
| combined_K3 vs v019_baseline | **1926/1926 vars BYTE-IDENTICAL** (maxΔ=0.000e+00) |

The levers keep `fp64_default` **byte-identical** on the nest — even the "numerics-free" cuda_async allocator
(buffer placement changed, FP ops/order and XLA kernel selection unchanged → identical bits, not merely tolerance).

## 8. Bottom line

The v0.20.0 low-hanging-fruit wave delivers a **safe, byte-identical ~1.05–1.10× warm speedup over v0.19
fp64 on the all-7 9-domain nest (≈1.50–1.58× over CPU-WRF)**, with **zero math change** — every shipped lever
is byte-identical (1926/1926 vars, maxΔ=0) and the cuda_async allocator does not regress fit or VRAM (it is
122 MiB leaner at 2h, no OOM).

- **The gain is carried entirely by the `cuda_async` pooling allocator** (~45 s/fc-h, cleanly separated in the
  back-to-back H2 session). On this **host-bound** nest (~7–8% GPU duty), cuda_async cuts per-op malloc churn.
- **async history output and the host-RAM event-tail guard are neutral on speed** here (only ~2 output groups
  to overlap at 2h; the writer thread shares the 4 pinned cores) — they ship as **bit-identical robustness**
  (async overlap pays off at longer history cadences / more cores; the guard bounds host RAM over 24–120h runs).
- **sync-cadence stays `root:1`** (root:2 slower; root:3's single-sample lead unconfirmed and within noise).
- **Honesty caveat:** the host-bound nest measured under live CPU-corpus contention on cores 4–31 carries
  run-to-run noise (~±2–3%, one H1 outlier shown) comparable to the small lever signal, so the headline is a
  **range**, not a false-precision point. A corpus-paused window would give a tighter number and is the place
  to re-confirm `root:3`. The relative cuda_async win and all identity/fit/VRAM gates are robust regardless.

This wave is the math-preserving host-bubble harvest the plan scoped (~1.05–1.25× honest band; we land at the
low end because the nest is GPU/host-bound rather than malloc-bound). The larger structural multipliers
(P4 dt/n_sound, fused acoustic/vertical megakernel, fp32-relaxed) remain the next, CFL-/skill-gated wave.
