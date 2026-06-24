# Release notes — wrf_gpu v0.20.1

**v0.20.1 is a reliability + I/O-readiness patch release.** It hardens the nested
GPU path against the out-of-memory class of failures (#123 — co-resident headroom
solved, solo fragmentation mitigated and carried), makes the persistent compile
cache hit across forecast dates for the nested path (#114), adds an opt-in
compact training-ready output mode (#122), and lands the paid-B200 I/O-readiness
tooling (manifest/dimension validation + block drain/resume/stop-pull). It also
applies an **honesty refresh** to the public-facing claims (perf headline framing,
memory accounting, identity-metric interpretation) with no fabricated numbers.

> **Compatibility — read this first.** The **fp64 default path stays
> byte-identical to v0.20.0**: the RRTMG column-tile cap change is bit-identical
> (CPU-proven `max_abs = 0.0`), the #122 training subset is **opt-in and the
> default output is byte-unchanged**, and the #114 cache change is a treedef
> date-invariance fix that does **not** alter the lowered HLO numerics. No
> namelist, interface, or numerical change on the default path.

> **Honest framing.** wrf_gpu is a **WRF-compatible reimplementation** (a clean
> JAX rewrite validated against WRF as an oracle), **not a Fortran-source port**,
> and a **transparent research artifact, not a full WRF replacement**. v0.20.1 is
> a reliability/readiness patch; the open credibility gate (24–120 h
> forecast-skill equivalence on the surface fields) remains future work and is
> **not** claimed closed here.

---

## What v0.20.1 fixes / adds

### #114 — cross-date warm NEST compile cache (bit-identical)

**CONFIRMED (GPU-validated + CPU treedef tests).** v0.20.0 closed the single-domain
across-date cache miss (#91) by passing the forecast date as a runtime argument. The
**nested** path still carried a residual baked date scalar in the pytree treedef
static-aux, so a new forecast date missed the warm persistent cache and paid the full
nested **cold compile (~50 min)**. v0.20.1 wraps the date/clock scalars in a
`_DateClockAux` holder so the **treedef is date-invariant** and the lowered HLO is
identical across dates — a new or leap-year forecast date no longer re-traces or
re-lowers the fused nest.

- The change is **bit-identical** on the default nested path (the same compiled
  executable; the date is data, not a graph constant). CPU treedef date-invariance
  is covered by `tests/test_operational_namelist_cache_key.py` (**9/9 green**,
  including an explicit leap-year `noahmp_yearlen` case).
- **GPU acceptance gate PASSED.** Cold `DATE_A` (20260512) compiled GREEN
  (persistent cache_delta ≈ 6497, 26 `wrfout` files), then a different-date warm
  `DATE_B` (20260228) ran GREEN with persistent **cache_delta = 2** (26 `wrfout`
  files) — the fused nest was **not re-traced/re-lowered** across the date change.

> **Warm is faster, not "instant" — read this honestly.** Removing the cross-date
> recompile saves the **~50 min cold compile**, but a warm cross-date run still pays
> a fused-module **load + link** of the cached executable (~19 min × 2 ≈ **38 min**).
> The net saving is **~30 min per new date**, not a zero-cost warm start. Do not read
> #114 as "instant warm" — it removes the per-date *recompile*, not the per-process
> module load/link.

*[Validated by an internal cross-date acceptance gate; CPU treedef tests:
`tests/test_operational_namelist_cache_key.py` (9/9). The cold/warm cache_delta
verdict is recorded in the v0.20.1 release-prep acceptance matrix (release-prep
tree). Code: branch `worker/opus/v0201-nest-prep`, merged into
`worker/integration/v0201`.]*

### #122 — opt-in compact training-ready nest output

A new **opt-in** compact `wrfout` mode for building training corpora at much lower
disk cost:

- Enabled by `GPUWRF_TRAINING_OUTPUT_SUBSET` (env, opt-in). When **off (the
  default), output is byte-identical to the legacy writer** (unchanged-legacy-path
  by design, `variable_subset=None`).
- Writes a focused **36-variable MINIMAL training set** (+ mandatory coordinate
  variables, 44 names total in the validated nest `wrfout`) instead of the full
  variable list, with **zlib-lossless** compression (no value change — the kept
  variables are bit-for-bit the same numbers, just fewer of them and compressed,
  round-trip exact).
- Cuts per-day output volume substantially for large nested corpora; the default
  full-output path is untouched.
- **Validated on a real GPU nest run** during the #114/#122 acceptance gate: the
  subset `wrfout` carried exactly the expected names, all levels present, finite
  and sample-buildable. CPU coverage: **10/10 green**
  (`tests/test_v0201_training_output_subset.py`).

*[Code: branch `worker/opus/v0201-nest-prep`. CPU coverage:
`tests/test_v0201_training_output_subset.py` (10/10). Default-path byte-identity +
the real-nest subset validation are recorded in the v0.20.1 release-prep acceptance
matrix.]*

### #123 — GPU OOM hardening for the nested path (mitigation, not a blanket fix)

The nested out-of-memory failures come in **two distinct modes**, and v0.20.1
addresses them separately. **One mode is solved; the other is mitigated and carried
as a documented limitation** — read both before drawing any conclusion.

1. **Co-resident-headroom OOM → SOLVED (GPU-validated).** A launch-time nested
   preflight. For `--max-dom > 1` the CLI now runs a CPU-side preflight that
   validates the cooperative GPU lock and checks **card-relative free VRAM**
   *before* JAX backend init or the ~50-min nested compile. If the headroom gate
   fails it **exits 75** before any expensive GPU work, instead of OOMing tens of
   minutes in.
   - Threshold is `max(GPUWRF_MIN_FREE_VRAM_GIB default 24 GiB,
     GPUWRF_MIN_FREE_VRAM_FRACTION × total VRAM)`, default fraction `0.50`
     (so 24 GiB on a 32 GiB 5090, ~90 GiB on a 180 GiB B200 — the absolute
     24 GiB is meaningless on a large card).
   - Multi-GPU: device selection honors `CUDA_VISIBLE_DEVICES` (physical index or
     GPU UUID/prefix), failing closed if the named device is absent.
   - `--force-gpu-run` / `GPUWRF_FORCE_GPU_RUN=1` intentionally bypass the gate.
   - **Limitation (documented):** this is a *launch-time* guard; it does not
     re-check if another process grows GPU memory mid-forecast.
   - **GPU-validated:** fail-closed path exits **75** on an under-headroom card,
     and the happy path returns `status = PASS` on a healthy card (it is **not**
     release-breaking). CPU coverage: `tests/test_gpu_preflight.py` /
     `tests/test_rrtmg_oom_hardening.py` (**9/9 green**).
   *[Code commits `b1c44524`, `4c9588db`.]*

2. **Solo `cuda_async` RRTMG-fragmentation OOM → MITIGATED (tier-1), NOT "fixed".**
   The RRTMG column-tile cap default is lowered **2048 → 1024**. This lowers the
   transient radiation working-set **without changing physics**: the RRTMG tiling
   path is column-local (columns do not couple; padded tail columns are discarded),
   so the result is **bit-identical** — CPU inertness proof
   `proofs/v013/rrtmg_column_tile.json` reports `max_abs = 0.0`, `max_rel = 0.0`
   on all required SW/LW all-sky and clear-sky cases. Shared
   `GPUWRF_RRTMG_COLUMN_TILE_COLS` override plus per-physics overrides remain.
   A standalone **RRTMG transient reproducer**
   (`scripts/rrtmg_transient_reproducer.py`) is shipped for LW/SW transient memory
   measurement.
   - **GPU-measured A/B (`cuda_async`, 144801 columns):** the 1024 cap cuts the
     radiation transient's **largest single allocation from 0.432 → 0.271 GiB
     (−37 %)** and lowers peak transient use, while remaining bit-identical.

   > **Carried limitation — do NOT over-claim.** The 1024 cap is bit-identical and
   > measurably reduces the radiation transient, but it does **not** make the nest
   > OOM-proof: **solo `cuda_async` fragmentation can still OOM the full fp64 nest**.
   > A reproducer is shipped so the failure can be reproduced and tracked. v0.20.1
   > therefore makes **no "OOM-proof", no fp32-nest, and no 24 h large-nest claim** —
   > the co-resident mode is solved by the preflight; the solo-fragmentation mode is
   > mitigated and remains an open limitation.

   *[Bit-identity: `proofs/v013/rrtmg_column_tile.json` (`max_abs = 0.0`). CPU
   smoke: a CPU-class RRTMG transient smoke (recorded internally). Reproducer:
   `scripts/rrtmg_transient_reproducer.py`. The GPU A/B largest-alloc numbers
   (0.432 → 0.271 GiB, `cuda_async`) are recorded in the v0.20.1 release-prep
   acceptance matrix (#123 mode-1 disposition). Commit `b1c44524`.]*

The nest default remains **fp64 + `cuda_async`**, which is the bounded path on the
reference 5090; the preflight closes the co-resident-headroom failure mode and the
1024 cap reduces (but does not eliminate) the solo-fragmentation failure mode.

### S2 — paid-B200 I/O-readiness tooling (CPU-only)

Tooling so a paid B200 nest run can stream and verify training output safely
without paying to rerun lost hours:

- **Manifest + WRF-dimension validator** (`scripts/b200_validate_manifest.py`):
  fail-closed on missing SHA-256 / expected domains; **rejects** an `897` child or
  the **old 181 mini-nest** grid and **accepts** `898`/`369` at `parent_grid_ratio
  = 3`; streamed SHA-256 of staged inputs.
- **Block drain / backpressure / resume / stop-pull**
  (`scripts/b200_drain.py`): verified copy with **S3 read-back-verify-before-delete**
  (`aws s3api head-object` confirms size + SHA-256 before the source is unlinked),
  done-marker validation that alarms on post-delete target loss, enforced S3 byte
  budget, atomic local copy, and rejection of truncated/empty blocks. Default
  policy `allow-partial` (every verified block is kept training-ready, never
  rerun); `--training-policy contiguous --expected-block-count N` is the stricter
  mode.
- Synthetic dry-run proof exercises the full copy → checksum → smoke → safe-delete
  path. **22/22 CPU tests green** (including S3-path data-loss-blocker coverage);
  zero GPU imports.

> **Carried limitation — tested, not yet exercised against real S3.** The tooling is
> proven on **synthetic / local** dry-runs only; the **real S3 read-back / delete
> path has not been exercised end-to-end**. The next paid B200 pod must run a
> **pre-pod read-back + delete dry-run against a disposable prefix** before trusting
> the safe-delete on real training output.

*[Code commits `56fb28b1` + `3c0933b2`. Tests: `tests/test_b200_manifest_validator.py`,
`tests/test_b200_drain.py` (22/22 green).]*

## Honesty refresh (public claims — no fabricated numbers)

Applied from the S3 paper/identity honesty pass (the internal release-prep
honesty analysis; every number below traces to a recompute over the on-disk
paired GPU/CPU `wrfout` trees):

- **Identity TH2 is a metric artifact, not a bug (I1).** The inner-nest 2 m
  potential-temperature `TH2` step-1 correlation looks worse than `T2`, but
  recomputing `TH2 = T2·(P0/PSFC)^{R/cp}` from the stored fields reproduces the
  written `TH2` to within `~10⁻³ K` on **both** the GPU and CPU runs, and `TH2`'s
  absolute GPU-vs-CPU error **equals** `T2`'s (≈0.78 vs 0.77 K RMSE). The θ
  conversion strips the terrain-correlated variance that keeps `T2`'s correlation
  high (the θ field's spread is ~4× smaller), so the **same shared near-surface
  offset** that leaves `r ≈ 0.98` on `T2` collapses Pearson `r` on the near-flat θ
  field. This is **low-variance Pearson collapse over the shared T2 offset, not a
  TH2-specific defect** — the earlier "suspected TH2 bug" framing is withdrawn, no
  code locus.
- **Report a variance-robust metric alongside r (I2).** Identity reporting now
  carries **nRMSE = RMSE/field-std** (and RMSE / max-abs / bias) next to Pearson
  `r`, so a near-flat field is not scored as broken (T2 nRMSE ≈ 0.28 → r 0.98;
  TH2 nRMSE > 1 → r collapses with near-identical absolute error).
- **d01 mean-r coverage caveat (I3).** The d01 domain contributes only **2 short
  leads (≤ 0.67 h)** versus 72 for d02–d09; the "mean r over domains" is **not
  lead-matched** and d01 inflates the surface-wind mean by **~+0.018**. This is
  annotated; no bare lead-unmatched mean is reported without the caveat.
- **"Switzerland worse than Canary" is a presentation artifact, not a model/port
  bug (#119, closed).** Two independent analyses (hash-verified) converged: on the
  72 h identity dashboards Switzerland is **comparable or better** — it has the
  *higher* `r` on every near-surface field (T2, U10, V10, W) and is only modestly
  worse on the 3-D winds (still `r > 0.98`). The apparently low `r` on accumulator
  and low-variance fields (e.g. RAINNC, flat θ) is the **same low-variance Pearson
  metric effect** described above, not a solver defect; no paper claim is falsified.
  This is reported here only as identity-wording honesty (the paper-figure provenance
  fix is a separate paper-track item, not part of this release).

*[I1/I2/I3 traced in the S3 release-prep honesty pass, recomputed from the paired
GPU/CPU `wrfout` trees with numpy + netCDF4 (no JAX/CUDA). The #119 Swiss-vs-Canary
verdict traces to the v0.20.1 release-prep Pearson consolidation (two hash-verified
analyses). The dynamics/thermo fp64 core remains bit-faithful: T, PH corr ≈ 1.0,
PSFC ≥ 0.999.]*

## Carried forward unchanged

All of the v0.20.0 capability — the default fused all-7 9-domain nest fast path
(~1.07× vs v0.19 / ~1.53× vs 12-rank CPU-WRF, byte-identical to v0.19), the
opt-in fp32 mixed-precision capability mode (−14.4% VRAM / ~1.16× cells), the
across-date single-domain compile cache (#91), and the fp64-default byte-identity
— carries forward unchanged.

## Known issues / scope boundaries

Full detail in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

| ID | Summary | Severity |
|---|---|---|
| #123 solo-fragmentation OOM | The co-resident-headroom mode is **SOLVED** by the launch-time preflight (GPU-validated fail-closed + happy-path). The *solo cuda_async fragmentation* mode is **MITIGATED, not fixed**: the bit-identical 1024 RRTMG cap cuts the transient's largest allocation **0.432 → 0.271 GiB (−37 %, GPU-measured)**, but solo fragmentation **can still OOM the full fp64 nest** (a reproducer is shipped). **No OOM-proof / fp32-nest / 24 h large-nest claim.** | Mitigated + carried limitation |
| High resident host RAM (~36 GB) | A run holds ~36 GB host RAM resident (Noah-MP/physics tables baked as static aux). Precision-independent; does not affect correctness/stability/results; limits pod density. Root-caused; the tables-as-runtime-args fix is **not** in v0.20.1 (deferred). | Root-caused, deferred |
| fp32 1 h-only fidelity | The opt-in fp32 mode is tolerance-checked only at the 1 h lead; the 24–120 h skill gate is future work. fp64 stays the byte-identical default. | Documented scope |
| 24 h/72 h forecast-skill closure | The open credibility gate (T2/U10/V10 broad skill-equivalence) remains future work; cell-identity on the dynamics/thermo core is proven. | Documented gap |

## Performance / precision note

v0.20.1 is **not** a performance release — it adds no single-card speedup over
v0.20.0. The fp64 default path is byte-identical; fp32 remains opt-in and is **not**
a single-card speedup. The large-GPU scaling/energy figures are characterized in
the project paper, not this release; the B200 perf headline is reported there as
peak (768²) + largest-grid-sustained (1024²) + roofline-fit asymptote together,
with the explicit working-set-vs-resident memory caveat (see the public-paper
honesty refresh).
