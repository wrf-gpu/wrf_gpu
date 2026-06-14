# Release notes — wrf_gpu v0.16.0

**v0.16 is the STABILITY release.** It proves that **every implemented L2
physics scheme runs coupled-green on a real case**, adds the **aerosol-aware
Thompson microphysics** (`mp_physics=28`) as the project's "+1", and ships a
concrete **1 km-unlock** capability: a chunked MYNN BouLac that makes a 1 km
single-domain forecast fit on one RTX 5090 — **bit-identically** to the dense
kernel. The performance story is **honest and unchanged in spirit from v0.15**:
the fp64 GPU kernel is **~parity** with CPU-WRF, not a speedup. The fp32
make-or-break is now **CONCLUDED**: the valid-numerics fp32 ceiling is a real but
small **~1.1×**, **proven and independently cross-confirmed** — larger fp32
speedups are **precluded** by the conservation/cancellation fp64 pins.

> **Honest performance framing — read this first.** The fp64 GPU kernel is
> **~parity** with 24–28-rank CPU-WRF total-wall — a **hardware law, not a code
> defect**: a GeForce RTX 5090 runs fp64 at **1/64** of its fp32 rate (1.7 vs
> 105 TFLOP/s), and the fp64 dycore sits *at* its 0.944 FLOP/byte roofline ridge,
> so the fp64-ALU term genuinely binds. **No end-to-end speedup is claimed.** The
> **full-working-set fp32 make-or-break is now COMPLETE and double-confirmed**
> (Opus implementation + independent GPT reproduction): the valid-numerics fp32
> ceiling is **~1.1× with 0 % VRAM-peak reduction from precision alone**, NOT ~4×.
> Measured full-working-set on real Switzerland d01: **16k 1.107× / 65k 1.110×,
> VRAM ratio 1.000** (`proofs/perf/v016/fullws_fp32_km_bench.json`); GPT
> reproduced **1.105× / 1.111×, VRAM 1.000**. The earlier **~4.3× "cost proxy"
> is a numerically-INVALID global-fp32 artifact (DISPROVEN)** — it turns x64
> **off** and downcasts the conservation/cancellation compute, which corrupts the
> base-absolute gradients **27× / 127×** beyond tolerance and drives **qke
> non-finite at 1 km**; it is an upper bound for *invalid* numerics, not a
> reachable WRF-faithful speedup, and is **not** a next-version target. **Fusion
> is ~0%** (probed NEGATIVE — XLA already fuses the step optimally). The **real
> wins v0.16 ships** are the genuine **~1.1× fp32** lane **plus** the **BouLac
> memory fix that unlocks 1 km in fp64** (orthogonal to fp32). The boundary-forced
> long-horizon fixture is now **built** (fp64 stable under LBC; fp64-vs-fp64
> control = 0.000 RMSE). Full evidence + the two verdict reports:
> `proofs/v016/fp32_verdict/`.

## What v0.16 delivers

### 1. Stability — every implemented L2 scheme coupled-green (24/25 + carry)
The decisive missing validation layer: each implemented, coupled-runnable scheme
must run **COUPLED on a real case** (real terrain + real lateral boundaries +
cross-scheme coupling + multi-step drift), staying finite, physically bounded,
and inside the **frozen v0.14 GPU-vs-CPU-WRF tolerance band** on the dynamics
fields — a real forecast, no masking, not a self-compare (the baseline is a
*different* operational suite; the candidate swaps exactly one family).

- **24 of 25 L2 targets GREEN**; the 25th (Noah-classic land surface,
  `sf_surface=2`) is an **honest scope-carry** (needs the WRF land/static-data
  bundle), so the rollup is **`ALL_GREEN_OR_CARRIED`**.
- Families proven green: all microphysics (mp 1/2/3/4/6/8/10/14/16), all PBL
  (1/2/7/8/99), all surface-layer (1/2/3/7/91), cumulus (2/3/6), shortwave
  (1/2), longwave (1). Every gate is finite, within the frozen manifest
  (worst dynamics RMSE ≤ 0.37× of limit), at ≤ 7.1 GiB peak VRAM.
- Case: **Switzerland d01 reinit-h36 replay**, 128×128×44, dt = 18 s, 1 h.
- Harness: `proofs/v016/coupled_coverage_gate.py`; per-scheme verdicts:
  `proofs/v016/coverage/*_gate.json`; rollup:
  `proofs/v016/coverage/coverage_rollup.json`.
- Dashboard (table + plots): `proofs/v016/dashboard/` — `RELEASE_COVERAGE_SECTION.md`,
  `coverage_grid.png`, `metric_headroom.png`.

### 2. The "+1" — aerosol-aware Thompson microphysics (`mp_physics=28`)
The water-/ice-friendly aerosol-aware Thompson scheme (WRF Registry package
`thompsonaero`), end-to-end:
- **`QNWFA`/`QNIFA` aerosol prognostics** added to the State pytree
  **append-only at the very end** (62-leaf schema; every prior leaf keeps its
  pytree position; flatten/unflatten round-trip + restart roundtrip verified),
  plus the WRF fake-surface aerosol emission and the climatological self-init
  path (`use_aero_icbc=.false.`, `wif_input_opt=1`).
- **Per-scheme oracle PASS against the UNMODIFIED pristine WRF module** (NOT a
  self-compare): `proofs/v016/thompson_aero_savepoint_parity.{py,json}` vs
  `module_mp_thompson.F:mp_gt_driver` at a late real-case timestep (**5187
  columns, GPU, re-verified on the consolidated branch**) — all fields within
  tier-1 carry bands (θ max 1.45 fp32 ULP; water closure max rel 2.4e-7).
- Fail-closed inputs: non-self-init aerosol IC/BC paths
  (`use_aero_icbc=.true.`) fail closed (never a silent wrong-input run).
- **Honest scope:** the **coupled short-real-grid field-gate** (mp8 vs mp28,
  ±advection) was queued on the contended single GPU and did not run in time; it
  is a **documented carry** (needs GPU time only, no further code). mp28 is
  oracle-validated + operationally wired + CPU threading/restart/precision/catalog
  green — it is **not** counted inside the 25-target L2 coverage sweep (which
  stays exactly as in §1), it is the separately-validated +1.

### 3. The 1 km-unlock — chunked MYNN BouLac (bit-identical to dense)
v0.15 stated 1 km grids do **not** fit on one 32 GiB RTX 5090 because of the
fp64 dycore working set. v0.16 ships the **specific lever** for the largest
single allocation in that set — the dense `(B, nz, nz)` MYNN BouLac
parcel-search matrix:
- **`_boulac_length_chunked`** (`GPUWRF_MYNN_BOULAC_CHUNKED=1`, default OFF;
  default chunk = 1) keeps the fusion-friendly cumsum kernel but caps memory at
  `(B, chunk, nz)`, so it sidesteps the XLA "very slow compile" pathology of the
  pure O(nz) form **and** the OOM of the dense form.
- **Result (real Switzerland d01, fp64, clean process per grid):** the dense
  path **OOMs the 1 km / 147,456-column step** (≈ 18.80 GiB single allocation);
  the chunked path **FITS at 18.25 GiB, finite**, ~566 ms/step. The Canary 1 km
  case now fits on one card.
  - **Caveat (honest):** this fit is **measured in a fresh process per grid**.
    Repeated multi-grid runs in one process can **fragment allocator memory**, so
    production should **isolate grids per process or recycle the process** between
    grids rather than sweeping many resolutions in a single long-lived process.
- **Bit-identity:** chunked vs whole-domain dense `max_abs == 0.0`
  (**BIT-IDENTICAL**) across 8 WRF stratification regimes and chunk ∈
  {1,3,4,7,16} (incl. non-divisors of nz = 44); vs an independent WRF
  nested-DO-WHILE NumPy reference `max_rel = 2.4e-16` (machine eps). MYNN unit
  suite green in both the chunked and the default dense mode (no regression).
- Proofs: `proofs/perf/v016/boulac_chunked_oracle.{py,json}`,
  `boulac_dense_baseline.json`, `boulac_chunk1_147k.json`; plot + table:
  `proofs/v016/dashboard/{onekm_unlock.png,ONEKM_UNLOCK.md}`; review:
  `.agent/reviews/2026-06-14-opus-boulac-onz-1km.md`. This is a pure algorithmic
  memory partition, **not** a numerics change, and is orthogonal to fp32.

### 4. Honest performance — the fp32 make-or-break, CONCLUDED
The make-or-break full-working-set fp32 investigation is **complete and
double-confirmed** (Opus implementation + independent GPT reproduction; a blind,
first-principles roofline re-derivation
`.agent/reviews/2026-06-14-opus-roofline-check.md` agrees). Panel:
`proofs/v016/dashboard/{honest_perf_panel.png,HONEST_PERF_PANEL.md}`; full
evidence + the two verdict reports: `proofs/v016/fp32_verdict/`.
- **fp64 GPU ≈ CPU-WRF parity (NOT a speedup)** — the GeForce fp64 1/64 hardware
  law; the fp64 dycore co-binds on bandwidth (70.0 ms roofline @65k) and the
  fp64-ALU term (75.5 ms) at AI ≈ 1.0. **Unchanged headline.**
- **The valid-numerics fp32 ceiling is a real but small ~1.1× — PROVEN.** The
  genuine fp32 win (fp32 acoustic + safe-compute; the in-loop fp64 cancellation
  islands pass their oracles) is **~1.1× with 0 % VRAM-peak reduction from
  precision alone**. Measured full-working-set, real Switzerland d01, RTX 5090:
  **16k 1.107× / 65k 1.110×, VRAM ratio 1.000**
  (`proofs/perf/v016/fullws_fp32_km_bench.json`); the numerically-defensible
  `safe` lane (keeps `p_total`/`ph_total` fp64) is **16k 1.108×, VRAM 1.000**
  (`fullws_safe_km_bench.json`). **Independently cross-confirmed** by GPT:
  **1.105× / 1.111×, VRAM 1.000** (`gpt_fullws_reproduce.json`).
- **~4× is NOT reachable with valid numerics — PROVEN by three measured pillars.**
  (1) Demoting the **whole** persistent State to fp32 (−700 MiB carried fp64
  arrays @65k) moves the VRAM peak by **0 GiB** — the peak is **transient**
  working memory, not persistent storage. (2) The base absolutes
  `p_total`/`ph_total` (~1e5) **cannot be stored fp32**: doing so corrupts the
  geopotential/PGF gradients **27× / 127×** beyond the gated-fp32 budget (bits are
  lost at *storage*, so an in-loop fp64 island is powerless) — they are
  conservation-pinned to fp64 **and** are the large arrays
  (`fullws_base_absolute_oracle.json`, `GATE_PASS=False`). (3) The transient peak
  is **precision-insensitive** (XLA `temp_size` 5305→5379 MiB, unchanged),
  dominated by fp64 cancellation islands + the qke-pinned MYNN work (qke goes
  **non-finite** in fp32 at 1 km: 3036 cells).
- **The earlier ~4.3× "cost proxy" is a numerically-INVALID global-fp32 artifact
  (DISPROVEN), not a next-version target.** The 70.49 → 16.44 ms/step (4.29×)
  figure turns JAX x64 **off** and downcasts the cancellation/conservation
  compute, corrupting the very pins that keep the forecast finite. It is an
  **upper-bound cost proxy for invalid numerics**, not a reachable WRF-faithful
  speedup. Double-single recovery costs **fp64-equivalent storage + ~16× time**;
  6× always exceeded the RTX 5090 roofline.
- **Fusion ≈ 0%** — the env-gated acoustic carry-split probe is bit-identical
  (60/60) with wall **−1.6 %** and bytes **−0.14 %**: XLA already fuses the step
  optimally. No fusion win is available.
- **The 1 km-unlock is ORTHOGONAL to fp32** — it is the MYNN BouLac dense→O(nz)
  / chunked **shape** rewrite (§3), not a precision change; it makes a 1 km single
  domain fit on one RTX 5090.

## Carried forward unchanged
All v0.15 capability (the **final fp64 GPU kernel** at its honest ceiling, 72 h
GPU-vs-CPU-WRF cell-for-cell identity in two regions, MYNN-condensation `niter`
fidelity fix, Thompson cold-collection, MUB/PB nest-base-state seam, the
identity-proof + long-horizon-divergence dashboards) and the v0.11–v0.14 stack
(live multi-domain nesting, restart continuity, conservation-closed budgets,
MYNN-EDMF, RRTMG, KF/BMJ/Grell/Tiedtke cumulus, standalone CLI, fail-closed
scheme catalog) carry forward. The v0.15 honest 72 h benchmark (~parity
total-wall; ~1.5× warmed-steady) is unchanged.

## Scope-carries to v0.17 — with honest reasons
| Carry | Why it is not in v0.16 |
|---|---|
| **Noah-classic land surface (`sf_surface=2`)** | The coupled gate needs the WRF land/static-data bundle (soil/veg tables + static fields) wired into the real-case harness; `SCOPED_CARRY` in the rollup (the 25th L2 target). |
| **mp28 coupled field-gate** | The mp8-vs-mp28 ±advection coupled gate was queued on the contended single GPU and did not run; needs GPU time only, no further code. The mp28 **L1 oracle (5187-col WRF parity)** is GREEN. |
| ~~Full-working-set fp32 (~4×)~~ — **CONCLUDED in v0.16, not a carry** | The make-or-break fp32 investigation is **complete and double-confirmed**: the valid-numerics ceiling is **~1.1×** (not ~4×), proven; the ~4.3× cost proxy is **numerically invalid** (corrupts conservation/cancellation; qke non-finite at 1 km). The genuine ~1.1× fp32 lane ships now; no "~4× next version" is pursued. Evidence: `proofs/v016/fp32_verdict/`. |
| **RRTMG SW/LW variants** | Need a WRF oracle rebuild for the additional radiation options. |
| **LW31 / MP97 / New-Tiedtke** | v0.17 feature scope (Held-Suarez LW, Goddard GCE microphysics, New-Tiedtke cumulus) — banked but deliberately deferred. |

## Known issues / scope boundaries
| ID | Summary | Severity |
|---|---|---|
| Performance | fp64 GPU is **~parity** with CPU-WRF (GeForce fp64 1/64 hardware law) — no end-to-end speedup. The fp32 make-or-break is **CONCLUDED**: the valid-numerics fp32 ceiling is **~1.1×** (proven + cross-confirmed); larger fp32 speedups are **precluded** by the conservation/cancellation fp64 pins (the ~4.3× cost proxy is numerically invalid). | Honest finding |
| 1 km scale | A 1 km **single domain** now fits on one RTX 5090 via chunked BouLac (this release; **orthogonal to fp32**), **measured in a fresh process per grid** (repeated multi-grid runs in one process can fragment allocator memory → isolate grids per process or recycle the process). Larger working sets (≥ ~196 k cols) still need multi-GPU sharding (fp32 does **not** shrink the transient-dominated peak — proven). | Improved; bounded |
| Noah-classic (`sf_surface=2`) | Coupled gate carried — needs the WRF land/static-data bundle. | Scope-carry |
| mp28 coupled field-gate | Carried (GPU-time only); the mp28 L1 WRF oracle is GREEN. | Bounded carry |
| QVAPOR (Canary) / RAINNC (Switzerland) | v0.15 bounded carries (MYNN marine entrainment; precip per-cell refinement), non-escalating over 72 h. | Bounded carries |
| KI-9 / KI-3 / KI-5 | Carried from v0.14 (forecast-skill equivalence; 104-var wrfout subset; powered TOST deferred). | Carry-overs |
