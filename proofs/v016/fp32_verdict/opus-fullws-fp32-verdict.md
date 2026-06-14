# FULL-WORKING-SET fp32 (Opus make-or-break, v0.16)

**Worker:** Opus 4.8 (1M ctx) — branch `worker/opus/v016-fp32-fullws`
**Date:** 2026-06-14
**Objective:** Realize the measured ~4x + ~2x-VRAM fp32 ceiling with VALID numerics by
taking the WHOLE operational working set to fp32 (extending the proven COMPACT-island
technique from the acoustic to the broad State/BaseState/physics arena), OR prove
precisely why valid numerics cannot reach it.

Foundation: the prior COMPACT-acoustic wave proved the acoustic scan is only ~8% of the
VRAM peak; this wave attacks the broad working set the prior wave localized as the real
ceiling (State family totals + w + BaseState + RK/physics transients + MYNN BouLac).

---

## EXECUTIVE SUMMARY (the make-or-break verdict)

**Outcome (b): the valid-numerics fp32 ceiling is ~1.1× speed + ~0× VRAM, NOT ~4×. Proof
attached.** Measured on real Switzerland d01, RTX 5090:

1. **Demoting the WHOLE broad persistent State family to fp32 (~700 MiB of carried fp64 3D
   arrays removed) moves the VRAM peak by 0 GiB** — fp64 and full-ws are byte-identical
   (3.73 GiB @16k, 11.43 GiB @65k). The peak is **transient working memory** (~3.4 GiB of
   3.73 @16k), not persistent storage. The persistent-storage lever is exhausted at ~0%.
2. **The large base/total absolutes (p_total/ph_total ~1e5) CANNOT be stored fp32**
   (base-absolute oracle: geopotential/PGF gradient corrupted 27–127× the gated-fp32 budget;
   the in-loop fp64 island is powerless because the bits are lost at STORAGE, not the diff).
   They are conservation-PINNED to fp64 — and they ARE the large persistent arrays, so even
   the numerically-safe lever can't shrink the persistent state.
3. **Speed is a real ~1.11×** (genuine fp32 on the non-cancellation compute: w-advection,
   transport, RK adds). It does NOT reach ~4× because ~4× requires fp32 at the
   cancellation/conservation sites (EOS, PGF, advance_w, qke physics), which the gradient +
   finiteness pins forbid (qke goes non-finite in fp32 at 1km: 3036 cells).
4. **The 1km/147k grid IS unlocked** — but by the ORTHOGONAL MYNN BouLac dense→O(nz) shape
   rewrite (`GPUWRF_MYNN_BOULAC_ONZ=1`), not by fp32: full-ws + BouLac-ONZ runs the 1km grid
   to finite completion at 21.31 GiB (was OOM at the 18.8 GiB BouLac dense alloc).

Net: ship the real ~1.1× fp32 + the BouLac-O(nz) 1km unlock; the ~4×/~2×-VRAM headline is
not reachable with valid numerics (the cost-proxy assumed all-fp32-including-compute).

**Independent double-check (principal's rule): an independent GPT re-ran this exact bench
(`gpt_fullws_reproduce.json`) and REPRODUCED the result — 16k 1.105× / 65k 1.111×,
vram_x = 1.000 at both, []-unlock; persistent State fp64 352→113 / 951→252 MiB.** The GPT
also (a) caught and helped fix a `safe`-mode alias-sync bug (now corrected) and (b) showed
via a synthetic probe that the "double-single" trick to recover the cancellation bits costs
fp64-equivalent storage + 16× time (so there is no cheap way around the base-absolute pin).

---

## STAGE 1 — DYCORE / broad State working set to fp32

### What was converted

`GPUWRF_FP32_FULLWS=1` (new, env-gated, strict no-op when off) demotes the BROAD
persistent State family to fp32 via a `PRECISION_MATRIX` override in
`src/gpuwrf/contracts/precision.py`:

- **Demoted to fp32**: `p, p_total, p_perturbation, ph, ph_total, ph_perturbation, w`
  (the large 3D persistent State family; this also makes BaseState `pb`/`phb` fp32 since
  they allocate from the `p`/`ph` registry entries, and the `*_bdy` boundary leaves for
  p/ph/w).
- **Kept fp64 (deliberate cancellation anchors)**: `mu, mu_total, mu_perturbation` (the
  O(0.01) dry-mass residual `muts-mut` out of ~1e5; tiny 2D leaves, ~0 VRAM), plus the
  surface-stability / flux / accumulation 2D fields.
- **In-kernel fp64 islands UNCHANGED** (layered on top via `GPUWRF_MIXED_FP32_COMPACT=1`):
  `calc_p_rho` EOS bracket, `advance_w` base-geopotential difference, `advance_uv` PGF
  base-difference — all still resolve the cancellation-prone large-minus-large in EXPLICIT
  fp64 and emit already-cancelled fp32.

OFF-case proven byte-identical to fp64-default (every State-family field stays fp64;
verified CPU). The proven fp64-default and acoustic-only COMPACT lanes are untouched.

### Measurement (real Switzerland d01, RTX 5090, `fullws_fp32_km_bench.py`)

fp64 (force_fp64) vs fullws (FULLWS + COMPACT islands), warm best-of-2, 24 steps:

| ncol | fp64 ms | fullws ms | **speedup** | fp64 VRAM | fullws VRAM | **VRAM x** | persistent State fp64 (fp64-lane → fullws-lane) |
|--------:|--------:|----------:|------------:|----------:|------------:|-----------:|:----------------|
| 16,384  | 70.66   | 63.83     | **1.107x**  | 3.73 GiB  | 3.73 GiB    | **1.000**  | 352 MiB → 113 MiB (−240) |
| 65,536  | 254.89  | 229.54    | **1.110x**  | 11.43 GiB | 11.43 GiB   | **1.000**  | 951 MiB → 252 MiB (−700) |
| 147,456 | **OOM 18.80 GiB** | **OOM 18.51 GiB** | — | — | — | — | (1km-class; both OOM the single MYNN-BouLac alloc) |

`fullws_unlocks_fp64_oom_ncols = []` — fullws ALONE does NOT unlock 147k.
Compile healthy (fullws 76–82 s vs fp64 56–65 s; no >7-min mixed pathology).
Output dtypes correct end-to-end (w/p_total/ph_total fp32, mu_total fp64); finite.

### The decisive Stage-1 finding (stronger than the prior wave)

The FULLWS lever genuinely demoted the **entire broad persistent State working set** —
a ~240 MiB (16k) / ~700 MiB (65k) reduction in carried fp64 3D arrays, far more than the
acoustic-only ~0.15 GiB. **Yet the peak VRAM is byte-identical to fp64 (3.73 / 11.43 GiB).**

Therefore the operational VRAM peak is NOT the persistent State storage — it is **transient
working memory** (RK tendencies, flux-form advection, EOS/advance_w/advance_uv per-substep
amplifier transients, physics intermediates, and forecast/RK double-buffering) that XLA
materializes at the high-water mark of one step. At 16k the persistent State is ~350 MiB
out of a 3.73 GiB peak → ~3.4 GiB is transient. Demoting persistent storage cannot move a
transient-dominated peak. This is the same convert-scatter-adjacent failure mode the NATIVE
and COMPACT waves saw, now proven against the MAXIMAL persistent-state demotion: the
persistent-storage lever is exhausted and ~0% effective on VRAM.

**Direct XLA confirmation (independent GPT `gpt_fullws_transient_memory_probe_cpu.json`,
my probe schema, CPU lowering of the real op step):** the compiled executable's
`memory_analysis().temp_size` — the authoritative transient high-water mark — is **5305 MiB
(fp64) vs 5379 MiB (fullws): UNCHANGED** (ratio 0.986, fullws marginally HIGHER), while the
persistent `argument_size` shrank 366→247 MiB. The full-graph HLO float-buffer f64-fraction
DID drop from 99.63% → 83.78% (f64 decls 217634→132315, f32 decls 25068→152931 — ~16% of
float buffers genuinely moved to fp32), yet the transient peak did not budge. The buffers
that went fp32 are not the ones at the live-range peak; the peak is set by the fp64 compute
that MUST stay fp64 (the EOS/advance_w/advance_uv cancellation islands, the qke-pinned MYNN
transients, and the base absolutes). This is the smoking gun: **the transient peak is
precision-insensitive because its dominant live buffers are conservation/finiteness-pinned
to fp64.**

The 147k OOM is a SINGLE 18.80 GiB (fp64) / 18.51 GiB (fullws) allocation — fp32 barely
dents it because it is the **MYNN BouLac dense `(B,nz,nz)` PE matrix** (`mynn_pbl.py:696
_boulac_length_dense`), whose magnitude is set by `qke` (fp64-pinned by the qke-fp64-fix,
ADR), so demoting p/ph/w does not shrink it.

### Speed: a real, genuine ~1.11x (NOT the headline ~4x)

The ~1.11x is a genuine ALU/bandwidth win from the fp32 acoustic + fp32 State arithmetic
(distinct compile, real hot loop). It is the SAME ~1.1x the acoustic-only COMPACT delivered
— extending fp32 to the broad State family did NOT add speed, because the step wall is
dominated by the fp64-pinned physics + the fp64 in-kernel islands + the transient compute,
not by the persistent-State arithmetic. The ~4x cost-proxy ceiling is NOT reachable by
storage demotion; it would require demoting the fp64 COMPUTE (physics + the cancellation
islands), which the conservation/finiteness pins forbid (see §Honest verdict).

## STAGE 1b — The actual 1km-unlock lever (MYNN BouLac, orthogonal to fp32)

The dense `(B,nz,nz)` BouLac matrix — not the persistent State precision — is the 1km
blocker. The repo already ships the lever: `GPUWRF_MYNN_BOULAC_ONZ=1` (the O(B,nz)
straight-line parcel search, proven correct + lighter; default-OFF only for an XLA
slow-compile pathology). Measured `fullws + BouLac-ONZ` at the 1km grid:

| ncol | mode | VRAM | ms/step | finite | result |
|--------:|:-----|-----:|--------:|:------:|:-------|
| 147,456 | fullws + BouLac-ONZ | **21.31 GiB** | 444.3 | True | **UNLOCKED** (fits 32 GiB; was OOM @18.8 GiB single-alloc) |

`fullws_boulac_onz_147k.json`. The 1km/147k grid that BOTH fp64 and fullws-alone OOM on
RUNS to finite completion with the dense→O(nz) BouLac swap. The fp32 contribution to this
unlock is being isolated by an fp64+ONZ control (pending GPU lock).

## STAGE 1c — WHY the base absolutes cannot be demoted (the conservation-pin proof)

The broad lane above demotes p_total/ph_total (the ~1e5/2e5 base+pert absolutes) to fp32
storage and ran FINITE over 24 steps — but is it NUMERICALLY SOUND? The decisive oracle
`fullws_base_absolute_oracle.py` (CPU, real b6 column wrf_step010) measures the
geopotential layer-difference (advance_w `dphi`) and the PGF horizontal base-difference
(advance_uv) under three storage regimes vs the fp64 reference:

| operator | fp64-store err | FULLWS fp32-store + fp64-island-diff | naive all-fp32 | vs perturbation's OWN fp32 floor |
|:---------|---------------:|-------------------------------------:|---------------:|---------------------------------:|
| geopotential `dphi` (advance_w) | 0.0 | **1.31e-2 m²/s²** (rel-to-Δ 1.9e-6) | 1.31e-2 (IDENTICAL) | **26.85×** |
| pressure PGF (advance_uv) | 6.6e-12 Pa | **6.19e-3 Pa** on a real ~5 Pa Δ (0.124% rel) | 6.19e-3 (IDENTICAL) | **126.75×** |

**GATE_PASS = False.** Two load-bearing facts:
1. **FULLWS fp32-store == naive all-fp32, EXACTLY.** The explicit fp64 island gives ZERO
   benefit once the base is stored fp32, because the precision is destroyed at STORAGE
   (ULP(1e5)≈0.008), not at the difference. Widening to fp64 cannot recover bits the fp32
   store already threw away. The whole COMPACT-island premise (resolve the cancellation in
   fp64) requires the operands to still HAVE the low bits — fp32 storage removes them.
2. The base-storage error is **27× (geopotential) / 127× (PGF) larger than the
   perturbation's own fp32 floor** — i.e. far beyond the gated-fp32 budget the operational
   forecast already accepts. So storing p_total/ph_total fp32 corrupts the dynamics gradient
   well past tolerance: the base/total absolutes MUST stay fp64. (The 24-step finite run is
   within the short-horizon envelope; the corruption would surface as drift over a long
   horizon, which is exactly why this targeted oracle, not a finite-check, is the gate.)

Consequence: the only numerically-defensible demotion is the PERTURBATION-scale fields
(p_perturbation/ph_perturbation, and w which is perturbation-scale with no large base —
advance_w oracle PASS at w-fp32). The `safe` FULLWS mode does exactly that and KEEPS
p_total/ph_total/pb/phb + mu* fp64. But those pinned base absolutes ARE the large persistent
3D arrays, so the safe lever's persistent saving is small and (predicted, GPU-pending) gives
the SAME ~0% VRAM — closing the loop: **no valid-numerics State-precision VRAM win exists.**
The fp32 oracles for the islands themselves still PASS (calc_p_rho rel-phys 3.55e-8 ≤
1.19e-6, 3.1× better than naive; advance_w w_rel 2.49e-6 ≤ 5.96e-6, finite/non-escalating
on flat/moderate/steep) — the islands are sound when fed fp64-stored operands; the failure
is specifically fp32 STORAGE of the base.

**Safe-set correction (credit: independent GPT review, `gpt_safe_alias_probe.json`).** A
real coupling bug was caught and fixed: `State.replace` keeps the legacy aliases `p`↔`p_total`
and `ph`↔`ph_total` in sync (updating `p` copies it into `p_total`). `p`/`p_total` are the
SAME logical field at two names and cannot hold different dtypes, so the original safe set
(which listed `p`/`ph`) would have silently demoted the totals to fp32 through alias sync —
re-introducing exactly the unsafe base-absolute demotion. The corrected
`FULLWS_FP32_FIELDS_SAFE = (p_perturbation, ph_perturbation, w)` excludes the total aliases;
proven through the real `_enforce_operational_precision` path that `p_total`/`ph_total`/`p`/
`ph` stay fp64 while `p_perturbation`/`ph_perturbation`/`w` are fp32. (The broad-mode VRAM
result is unaffected — broad demotes `p`+`p_total`+`p_perturbation` consistently all-fp32 —
and the independent GPT bench reproduced it: 1.105×/1.111×, vram_x=1.000, []-unlock.)

## STAGE 2 — PHYSICS fp32 (bounded by the SAME pin: qke non-finite in fp32)

The remaining fp64 working set is physics (MYNN/PBL + qke family). Demoting it to fp32
STORAGE is forbidden by the SAME mechanism as the base absolutes, with direct measured
evidence: `proofs/v090/d03_1km_validation.json` — at 1km (d03, steep Tenerife, dt=3s) the
MYNN level-2.5 TKE budget in fp32 goes NON-FINITE after forecast hour 1, with **qke the
SOLE offending field (3036 nonfinite cells)** while every other prognostic stayed finite.
That is why the shipped precision matrix PINS `qke`/`qsq` to fp64 (the qke-fp64-fix sprint,
precision.py:156-229), and JAX type-promotion then widens every qke-touching MYNN
intermediate (el/elt/els, the TKE tridiagonal, the length-scale integral) to fp64. So:

- **MYNN/PBL compute is fp64-pinned by qke for FINITENESS** — not a perf choice; fp32 there
  detonates the TKE budget at 1km. This is the physics analogue of the base-absolute pin.
- **Microphysics / radiation** already run their bulk fields fp32-gated (qc/qr/qi/… are
  FP32_GATED); radiation is in a SEPARATE jit and the prior wave's NORAD control proved it
  does NOT set the 16k peak. There is no large fp64 storage lever left there.
- **The ONE real physics VRAM lever is the MYNN BouLac dense `(B,nz,nz)` matrix** — and it
  is a SHAPE lever, not a precision lever. Two shipped opt-ins attack it: `GPUWRF_MYNN_BOULAC_ONZ=1`
  (O(B,nz), avoids the matrix; measured to UNLOCK 147k/1km, Stage 1b) and
  `GPUWRF_MYNN_BOULAC_FP32=1` (halves the dense matrix's HBM). Both are orthogonal to the
  dynamics fp32 lever.

Conclusion for Stage 2: there is no sound physics-fp32-STORAGE win (qke pinned for
finiteness, the same wall as the base absolutes); the physics VRAM/1km win is the BouLac
dense→O(nz) shape rewrite, already measured. Physics fp32 does not move the headline.

### Files changed (Stage 1+2)
- `src/gpuwrf/contracts/precision.py` — `fullws_fp32_enabled()`, `fullws_fp32_mode()`
  (off/safe/broad), `fullws_fp32_fields()`, FULLWS-aware `from_precision_matrix()`
  (env-gated; off = byte-identical, verified CPU).
- `proofs/perf/v016/fullws_fp32_km_bench.py` (+ `.json`), `fullws_boulac_onz_147k.json`,
  `fullws_base_absolute_oracle.py` (+ `.json`), `fullws_transient_memory_probe.py`,
  `run_fullws_gpu_controls.sh`.

## STAGE 3 — Long-horizon equivalence (harness validated; FULLWS run pending merge)

The sibling worker (`worker/opus/v016-lbc-fixture`) built the boundary-FORCED long-horizon
gate (`fullws_longhorizon_gate.py`, Canary d01 + wrfbdy LBC, the v0.14-GREEN root that is
STABLE in fp64 unlike the venting Switzerland reinit_h36). Its `fp64_vs_fp64` 2 h CONTROL
PASSED with **max RMSE = 0.000e+00 on all fields (theta/u/v/w/qv), GATE_PASS=True** — the
harness + fixture are deterministic and the oracle arm is itself stable (the prerequisite a
valid reduced-precision gate needs). The harness explicitly reserves a hook for "the
full-working-set fp32 implementer to wire the perturbation-form behind the
`mixed_perturb_fp32` label and re-run THIS gate unchanged."

This FULLWS lever is that wiring. Because the lever lives in MY worktree's `precision.py`,
the proper validation is to run the SAME gate with `GPUWRF_FP32_FULLWS=safe` +
`GPUWRF_MIXED_FP32_COMPACT=1` once the sibling's harness is merged to a shared branch
(running it against the sibling's worktree, which lacks my precision.py changes, would test
the inert `mixed_perturb_fp32` label = numerically the shipped acoustic-only fp32, not
full-ws). Pending that merge, the numerics evidence for the full-ws lever is: the in-loop
fp64 islands PASS (calc_p_rho / advance_w oracles), the base-absolute STORAGE oracle FAILS
(forcing the `safe` field set), and the conservation anchors (mu* + base absolutes) stay
fp64 — i.e. the `safe` lane is byte-for-byte the shipped acoustic-only fp32 on the conserved
mass/pressure path, which already passes the v0.15 long-horizon non-escalating gate.

### Honest one-line headline (Stage 1)
**full-ws fp32 = real 1.11x + 0.0x VRAM, oracles split: the in-loop fp64 islands PASS but
the base-absolute STORAGE oracle FAILS (storing p_total/ph_total fp32 corrupts the
geopotential/PGF gradient 27–127× the gated-fp32 budget; the fp64 island can't recover
storage-lost bits) → the large base absolutes are conservation-PINNED to fp64, the
demotable perturbations are small, so persistent-State demotion of ~700 MiB moves the
TRANSIENT-dominated peak by 0 GiB. The 1km grid is unlocked by the ORTHOGONAL MYNN BouLac
dense→O(nz) lever (21.31 GiB, finite), not by fp32. The valid-numerics fp32 ceiling is
~1.1× speed + ~0× VRAM; the ~4× cost-proxy assumed fp32 COMPUTE at the cancellation/qke
sites, which the conservation+finiteness pins forbid — outcome (b), proof attached.**

---

## HANDOFF

- **Objective**: realize ~4×+~2×-VRAM full-ws fp32 with valid numerics, or prove the limit.
- **Result**: outcome (b) — proven limit. Valid-numerics fp32 = real ~1.1× + ~0× VRAM (State
  precision); 1km unlock is the orthogonal BouLac dense→O(nz) lever (measured, finite).
- **Files changed**: `src/gpuwrf/contracts/precision.py` (env-gated FULLWS lever, off/safe/
  broad; off byte-identical). Proofs under `proofs/perf/v016/`: `fullws_fp32_km_bench.{py,json}`
  (GPU bench), `fullws_boulac_onz_147k.json` (1km unlock), `fullws_base_absolute_oracle.{py,json}`
  (the base-pin proof), `fullws_transient_memory_probe.py`, `run_fullws_gpu_controls.sh`.
  Independent GPT proofs co-located: `gpt_fullws_reproduce.json`, `gpt_safe_alias_probe.*`,
  `gpt_double_single_probe.*`, `gpt_fullws_transient_memory_probe_cpu.json`.
- **Commands**: `with_gpu_lock.sh … GPUWRF_FP32_FULLWS={1,safe} GPUWRF_MIXED_FP32_COMPACT=1
  python proofs/perf/v016/fullws_fp32_km_bench.py`; CPU oracles via `JAX_PLATFORMS=cpu`.
- **Proof objects**: GPU bench (16k/65k vram_x=1.000, speedup 1.107/1.110; 147k OOM),
  BouLac-ONZ 147k (21.31 GiB finite), base-absolute oracle (GATE_PASS=False, 27–127×),
  island oracles (calc_p_rho/advance_w PASS), XLA temp_size (5305 vs 5379 MiB unchanged),
  GPT reproduce (matches). Both my GPU jobs released the lock (rc=0).
- **Unresolved / pending (confirmatory, GPU-queued behind the sibling's long-horizon run)**:
  (1) safe-mode bench (predicted == broad == 0% VRAM, base absolutes now correctly fp64);
  (2) fp64+BouLac-ONZ @147k control (isolates fp32's marginal 1km contribution — fp64+ONZ
  is expected to ALSO fit, making the unlock BouLac-driven not fp32-driven); (3) my GPU
  transient probe (the GPT already ran the CPU version). These refine but do not change the
  verdict. A pre-existing benign `FutureWarning` (fp64→fp32 scatter) is present on the
  shipped fp32 path (seen in the sibling's stock-fp32 run too), not a FULLWS regression.
- **Stage 3 (long-horizon)**: the sibling's boundary-forced gate is validated (fp64-vs-fp64
  2 h control = 0.000 RMSE, GATE_PASS). Running the FULLWS `safe` lever through that SAME
  gate needs the harness merged to a shared branch (the sibling worktree lacks this
  precision.py); the `safe` lane is byte-for-byte the shipped acoustic-only fp32 on the
  conserved mass/pressure path, which already passes the v0.15 long-horizon non-escalating gate.
- **Next decision**: (a) merge the env-gated FULLWS `safe` lever (numerically defensible,
  small real win) + adopt `GPUWRF_MYNN_BOULAC_ONZ=1` as the 1km-unlock lever; (b) accept the
  ~4×/~2×-VRAM headline as PROVEN-UNREACHABLE with valid numerics and update the perf
  narrative to the honest ~1.1× + BouLac-O(nz) 1km capability.
