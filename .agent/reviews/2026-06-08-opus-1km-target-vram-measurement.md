# 1 km Target-Geometry Peak-VRAM Measurement (does it fit the 32 GB RTX 5090?)

**Date:** 2026-06-08
**Worker:** Opus (CPU-only probe, GPU untouched — TOST marathon protected)
**Branch:** `worker/opus/v0120-1km-vram-measure` (based off `worker/opus/v0120-integration`)
**Probe:** `proofs/v013/target_1km_vram_probe.py` → `proofs/v013/target_1km_vram_probe.json`
**Method:** Path (a) DIRECT CPU peak-RSS of the resident carry **and** Path (b) ANALYTIC per-step transient, anchored to **GPU-measured** proof artifacts.

---

## Bottom line

**No.** The principal's target d03 (641×321×50 ≈ 205,761 columns) does **NOT** fit the 32 GB RTX 5090 — neither as a live 3-domain nest nor as a d03-replay. The binding constraint is the **per-step RRTMG longwave radiation g-point transient**, which scales **linearly in `ncol = ny·nx`** and is **not column-tiled** in the current code. At 641×321×50 that single per-step transient is **~86 GiB** (best case, LW chunk=1) — ~3× the whole card.

| | 641×321×50 target |
|---|---|
| d03-replay peak (chunk=1) | **89.3 GiB** — does NOT fit |
| live 3-dom-nest peak (chunk=1) | **89.9 GiB** — does NOT fit |
| (chunk=14 / unchunked) | 156 GiB |

**Largest 1 km grid that fits 32 GB (29 GiB usable, chunk=1):**
- **d03-replay:** ~**59,500 columns** → **~345×172** (2:1) or **~243×243** (square)
- **live 3-dom nest** (d01 9km + d02 3km resident too): ~**58,400 columns** → **~342×171** / **~241×241**

The target is ~3.5× too many columns. The prod 1 km nest we ship (94×76 ≈ 7,144 cols) sits at ~3.1 GiB and fits comfortably (consistent with the GREEN v0.13 24h nested-1km+GWD gate).

---

## Per-shape table (50 levels, fp64, full physics)

`ncol = ny·nx`. Peak = resident carry + dominant per-step transient (rad peak = max(SW,LW) since WRF runs the two drivers sequentially and XLA frees one before the other — confirmed by the v0.12.0 OOM post-mortem: the OOM was a *single* recurring transient, not SW+LW summed). `+dyn` = ~20 mass-grid fp64 working arrays for RK3/acoustic/advection (small vs radiation).

| 1km shape | ncol | resident carry | rad transient chunk=1 (SW/LW→peak) | rad transient chunk=14 →peak | **d03-replay peak** chunk1 / chunk14 | **live-nest peak** chunk1 / chunk14 | fits 32 GB? |
|---|---|---|---|---|---|---|---|
| 400×250×50 | 100,000 | 1.00 GiB | 7.9 / 41.7 → **41.7** | → 73.9 | 43.4 / 75.6 | 44.0 / 76.2 | **No** |
| 560×280×50 | 156,800 | 1.56 GiB | 12.4 / 65.3 → **65.3** | → 115.9 | 68.1 / 118.6 | 68.6 / 119.1 | **No** |
| **641×321×50** (target) | 205,761 | 2.04 GiB | 16.2 / 85.8 → **85.8** | → 152.1 | **89.3** / 155.6 | **89.9** / 156.2 | **No** |
| 720×360×50 | 259,200 | 2.56 GiB | 20.5 / 108.0 → **108.0** | → 191.6 | 112.5 / 196.0 | 113.1 / 196.6 | **No** |

(For reference, the *prod* d03 94×76×45 ≈ 7,144 cols → replay peak ~3.1 GiB — fits, matches the shipped GREEN gate.)

---

## chunk=1 vs chunk=14 (g-point band tiling) delta

The RRTMG g-point/optics chunking knob tiles only the **band axis** (14 SW / 16 LW bands), not columns. GPU-measured at the anchor (ncol=24576, nlev=48, RTX 5090, fp64; `proofs/v013/optics_taumol_chunk.json`):

| | upfront (chunk=14/16) | chunked (chunk=1) | reduction |
|---|---|---|---|
| SW peak | 16,730 MiB | 1,906 MiB | **−88.6 %** |
| LW peak | 17,854 MiB | 10,068 MiB | **−43.6 %** |

So chunk=1 is a large win on SW (~9×) but only ~1.8× on LW, and **LW is the binding term at every 1 km size** (LW chunked floor = several concurrent `(ncol, nlay, 16, 16)` fp64 taumol/rtrnmc buffers). At the target, chunk=1 drops the radiation peak from ~152 GiB → ~86 GiB — a 1.8× cut, still ~3× over the card. **Band-chunking alone cannot make 641×321 fit.**

---

## Dominant memory consumers (ranked)

1. **RRTMG LW g-point radiation transient — DOMINANT, the binding constraint.**
   Per-step XLA temporary on the full `ncol = ny·nx` column batch. The chunked floor is several concurrent `(ncol, nlay, 16, 16)` fp64 taumol/rtrnmc working arrays. A *single* such array at 641×321×50 is **19.6 GiB**; the measured chunked peak holds ~4–5× that. Scales **linearly in ncol**, ~linearly in nlev. **Not column-tiled** anywhere in `physics/rrtmg_lw.py` / `coupling/physics_couplers.py` (radiation reshapes `(nz,ny,nx)→(ncol,nz)` and processes all columns at once).
2. **RRTMG SW g-point transient — second** (chunk=1 cuts it to ~16 GiB at target; chunk=14 ~142 GiB). Same `ncol`-linear scaling, but band-chunking is far more effective on SW.
3. **Resident timestep carry (`State` + `OperationalCarry`)** — persistent but small: 2.0 GiB at the target (53 SoA `State` fields, mixed fp64/fp32-gated/int32, + 14 fp64 3D scratch saves). Trivial vs radiation; even d01+d02 companions add only ~0.55 GiB. **NOT the constraint.**
4. RK3/acoustic/advection flux scratch — ~0.4–1 GiB at the target; negligible.

---

## Validation of the method (why these numbers are trustworthy)

- **Path (a)** measured the resident carry on the CPU JAX backend (fp64, taskset -c 8-15). Measured-array bytes **match the analytic field-shape/precision-matrix sum exactly** at every shape (e.g. 641×321 → 2.0397 GiB both ways), and the `/proc` RSS delta tracks it (lazy-zeros under-count RSS, expected).
- **Path (b)** scales the **GPU-measured** radiation peaks (`proofs/v013/optics_taumol_chunk.json`, RTX 5090) by `ncol·nlev`. **Independent cross-check against a real run:** scaling the *unchunked* anchor down to the real d02 nested grid (160×67×45) predicts a **6.7/7.1 GiB** radiation transient; the v0.12.0 nested-OOM post-mortem (`proofs/v0120/nested_oom_fix.json`) **measured 8.21 GiB** largest single alloc / 9.11 GiB peak on that exact grid. The model is ~14 % **conservative** (under-predicts), so the real 641×321 transient is likely *somewhat larger* than the 86 GiB reported — the "does not fit" verdict is robust.

**Confidence: HIGH.** The verdict rests on a measured GPU anchor, a real-run cross-validation, and a linear scaling law backed by the actual array shapes in the source. The exact max-fitting `ncol` (~59 k) is ±~15 % depending on allocator fragmentation headroom and the precise LW concurrent-buffer count, but the order of magnitude (target is ~3.5× too big) is not in doubt.

---

## What would make 641×321 fit (for the manager, not implemented here)

The only lever that helps is **column-tiling the RRTMG drivers** (a `lax.scan`/`lax.map` over column tiles, the same pattern already used for band-tiling). A column tile of ~60 k columns would cap the radiation transient at the current ~29 GiB ceiling and let the d03 columns be processed in ~3.5 sequential tiles — making 641×321 fit at the cost of radiation-step wall-clock (the rest of the step is already `ncol`-streaming-friendly). fp32-gating the radiation working set (currently fp64) would roughly halve it but is a precision/ADR decision. Multi-GPU sharding of the column axis is the other route. None of these exist today; the radiation path is single-pass over all columns.
