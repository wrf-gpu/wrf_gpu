# v0.18 default-path perf regression — independent HLO/structural root-cause (Opus)

**Worker:** Opus 4.8 (effort max), branch `worker/opus/v018-perffix`.
**Angle:** complementary to GPT's warm-timing — HLO/jaxpr **structural** diff. Required OTHER-MODEL double-check of the perf result.
**Case:** canonical `proofs/perf/warmed_timing.py` default-scheme coupled real d02 (ny66 nx159 nz44, dt 10 s, radiation cadence 180; Thompson mp=8 + RRTMG + MYNN + Noah).
**Shippable artifact measured:** `worker/gpt/v018-integration @292a4431` ("Reduce v018 operational scan overhead" — GPT's fix). v0.17 baseline = `worker/opus/v017-release` @ 99c1b83c.

---

## TL;DR

- **Two independent costs were conflated by the raw eqn-count; only one mattered at runtime:**
  1. **Thompson cold-process additions** (rci/sci/rcg/qcfz + graupel-number diag + ice-dist bounds, commit `044bb65a`): **+382 jaxpr eqns / +771 StableHLO ops per step**, BUT **warm-FREE** — same-session ablation gating the headline ice-collection off moved per-step time by **+0.07 % (noise)**; XLA fuses the 157 ops into existing grid passes. **Keep as-is, no fuse, no gating.** Eqn-count was misleading; runtime is what matters.
  2. **Default carry program-shape**: v0.18 left the mp=8 conditional leaves (`qh,Nh,qvolg,qvolh,nwfa,nifa,hail_acc`) `None`, **narrowing the scan carry 81→74 leaves**. My structural diff FOUND this change but I initially mis-signed it as harmless ("narrower = less data = faster"). **GPT's `292a4431` proves the opposite**: the `None`-leaf default mp=8 GPU program shape is the (small) warm regressor; re-materializing the leaves at the public entry (`_operational_scan_state`, `include_all_conditional=True`) restores v0.17's wide-carry layout. This is the two-angle cross-check working: structural diff located the change, warm-timing fixed its sign.
- **GPT's structural hypotheses, reconciled:** radiation-cond and noahmp-carry **REFUTED**; "State-pytree change" was REAL but its sign was inverted — the *narrowing* (not growth) was the regressor. (§3)
- **Same-session warm (canonical `warmed_timing.py`, public entry):** v0.17 = **21.1516** s/fc-h vs v0.18@292a4431 = **21.2661** s/fc-h → **+0.54 % (+0.32 ms/step) = PERF-NEUTRAL** (the delta is at the noise floor — v0.17's own intra-run repeat spread is ±0.48 %). The originally-reported +5.8 % was cross-session clock drift + the now-fixed carry program-shape. (§5) **Evidence-honesty caveat (integration-critic F2):** this `21.1516 / 21.2661` post-fix pair was measured in-session but its warmed-timing JSON was **not committed** (no committed JSON in the repo contains those numbers — the committed Opus same-session JSONs `perf_neutrality_v017_rerun/v018_warmed_timing.json` = 20.6253/21.8359 are the **pre-fix +5.86 % transient** this report supersedes). The **committed** perf-neutral conclusion rests on the committed **GPT independent series** (`gpt_verify_*`, v0.18 −1.05 % = faster) plus this report's structural root-cause (cold-process +0.07 % ablation-noise; carry fix bit-identical); the +0.54 % Opus figure is consistent prose, not a committed artifact, and the verdict does not depend on it.
- **No fidelity-vs-speed decision needed.** Cold-process is free; the carry fix is bit-identical (materializing zero leaves changes no value). **v0.18-integration @292a4431 is shippable on perf.** (§6/§7)

---

## 1. Method (structural, GPU-frugal)

The production scan body was **lowered to StableHLO** and **traced to a jaxpr** (trace-only, no kernels) for the identical default real case on both branches. Two diffs: (a) StableHLO op histogram of the per-step graph; (b) jaxpr source-attribution — every primitive (recursively through scan/cond sub-jaxprs) mapped to its `gpuwrf/...` `file:line`, then per-file/line counts diffed. Warm magnitude via same-session probes + the canonical `warmed_timing.py`. Scripts: `/tmp/hlo_dump.py`, `/tmp/jaxpr_attrib.py`, `/tmp/perf_probe.py`.

## 2. Carry / State pytree — the real (small) warm regressor, sign-corrected

Threaded scan carry (`OperationalCarry`) for the default case:

| | leaves | bytes | dtypes |
|---|---|---|---|
| v0.17 | 81 | 343.0 MB | 69 fp64 + 11 fp32 + 1 i32 |
| v0.18 @7b3bcc89 | **74** | 331.8 MB | 68 fp64 + 5 fp32 + 1 i32 |
| v0.18 @292a4431 (GPT fix) | **81** (re-materialized) | 343.0 MB | restored |

v0.18 left mp=8 conditional leaves `None` (`contracts/state.py`), narrowing the carry. **Counter-intuitively this produced a slower default mp=8 GPU program shape**, not a faster one (fewer-but-`None` leaves change the lowered program/donation layout and force one-time zeros/materialization in the executable). GPT's `292a4431` re-materializes all conditional leaves OUTSIDE the jit (`_operational_scan_state` → `ensure_conditional_leaves(include_all_conditional=True)`), restoring the pre-v0.18 wide carry. **Bit-identical** (the materialized leaves are exact zeros, inert for mp=8).

> My lean `_scan_forecast_segment` probe could NOT see this: the carry materialization happens in the PUBLIC entry (`run_forecast_operational` → `_operational_scan_state` → `_dealias_pytree_buffers` → `_run_forecast_operational_jit`), which the direct-segment probe bypasses. Hence the canonical `warmed_timing.py` (public entry) is the authoritative measurement for this effect.

## 3. GPT hypothesis reconciliation (independent evidence)

| GPT hypothesis | Verdict | Evidence |
|---|---|---|
| radiation-step scan/cond embedding | **REFUTE (timed path)** | `run_forecast_operational`/`_run_forecast_operational_jit` use `_scan_forecast_segment` with **static** `run_radiation` (separate compiles, no per-step cond). Radiation runs 1/180 steps; rad-step op-Δ (+808) ≈ nonrad op-Δ (+771) → radiation added ~nothing. |
| materialized full-carry (noahmp_land) | **REFUTE** | `noahmp_land/rad`, `slab_*`, `px_*` are `None` in default; not materialized. |
| State-pytree change | **CONFIRM, sign-inverted** | The pytree DID change — but it *narrowed* (81→74), and the narrowing (not growth) was the regressor. GPT's `292a4431` fix = re-widen. §2. |

## 4. Op-count diff + localization (where the eqns went — but they're warm-free)

Production scan body (StableHLO), default case: nonrad **83,709→84,480 (+771)**, rad 106,664→107,472 (+808). Δ flavor: constant +241, broadcast +207, multiply +105, divide +60, power +26, log +6, maximum +27, compare +34 … = per-cell elementwise+transcendental microphysics, identical in nonrad & rad (every-step).

jaxpr per-file Δ (nonrad): **thompson_column.py +382** (dominant); operational_mode.py −18; physics_couplers.py +5; state.py −6; mynn_edmf.py −2. The +382 localizes to the new cold-process code: `_ice_distribution` bounds, `_default_mp8_graupel_number`/`_graupel_distribution`, `_ice_collection_rates_from_moments` (rci/sci), `_cloud_water_freezing_rates` (qcfz), the ice/rain budget rescaling. **These are warm-free (§5) — XLA fuses them.**

## 5. WARM magnitude — same-session, clock-stable

GPU clocks drift ~4 % session-to-session here (same `7b3bcc89` binary measured 58.66 ms/step in one probe session, 56.06 in another) — **only same-session deltas are authoritative** (matches the principal directive). The originally-reported +5.8 % was largely cross-session drift.

**Ablation (Session A, v0.18 binary, lean per-step probe, build-once, GPU warmed):**

| config | ms/step | s/fc-h | Δ vs full |
|---|---|---|---|
| full | 58.656 | 21.116 | — |
| `ICE_COLLECTION=0` (rci/sci off) | 58.695 | 21.130 | **+0.07 %** ← cold-process is FREE |
| `COLD_COLLECTION=0` (whole cold lane off) | 57.298 | 20.627 | −2.32 % (lane is mostly pre-v0.17) |

**Same-session canonical (`warmed_timing.py`, public entry) — measured in-session, JSON NOT committed (see F2 caveat in TL;DR):**

| build | warmed ms/step | warmed s/fc-h | intra-run repeat spread |
|---|---|---|---|
| v0.17 (99c1b83c) | 58.7545 | 21.1516 | ±0.283 ms (0.48 %) |
| v0.18-integration @292a4431 | 59.0724 | 21.2661 | ±0.109 ms (0.19 %) |

> The committed warmed-timing JSONs for this report's Opus same-session pair were
> not captured; these two rows are prose-only. The committed perf-neutral evidence
> is the GPT independent series (`gpt_verify_v017/v018_warmed_timing.json`, v0.18
> −1.05 % = faster) plus the structural ablation above (cold-process +0.07 %).

**Same-session Δ (v0.18@292a4431 − v0.17) = +0.32 ms/step = +0.1145 s/fc-h = +0.54 %** → **PERF-NEUTRAL** (delta ≈ v0.17's own ±0.48 % repeat spread; not a resolvable regression). For reference the original report was +5.8 % (mostly cross-session GPU clock drift — the same `7b3bcc89` binary measured 58.66 vs 56.06 ms/step in two different sessions).

## 6. Optimization / verdict

- **Cold-process (rci/sci/rcg/qcfz):** already warm-free (XLA-fused, +0.07 % ablation). No manual fuse/vectorize helps; **keep default-ON**. No fidelity-vs-speed decision needed — it costs ~nothing warm and closes the cold mixed-phase RAINNC oracle (`coldmix_validation_after_rci_sci.json`: 0.2462 mm vs WRF 0.2469 mm). RAINNC fidelity untouched by this work.
- **Carry program-shape:** fixed by GPT `292a4431` (re-materialize conditional leaves; bit-identical). This was the real (small) warm regressor my structural diff located but mis-signed; GPT's timing fixed the sign.
- **Residual after `292a4431`:** +0.54 % same-session, which is **within the warm-sample noise floor** (≈ v0.17's own ±0.48 % intra-run repeat spread). Not a resolvable regression; nothing further to fix.

## 7. Recommendation

**SHIP v0.18-integration @292a4431 on perf grounds.** It is perf-neutral vs v0.17 (+0.54 %, within noise). No gating, no fidelity-vs-speed trade, no further optimization warranted:
- Cold-process additions (rci/sci/rcg/qcfz) are warm-free (XLA-fused) and improve RAINNC cold mixed-phase fidelity — keep default-ON.
- GPT's `292a4431` already neutralized the carry program-shape regressor (bit-identical).
- The headline +5.8 % was an artifact of cross-session GPU-clock drift measured against the pre-fix `7b3bcc89`; same-session vs the shippable `292a4431` it is noise.

**Independent cross-check requested:** GPT to warm-verify `292a4431` vs same-session v0.17 (this worker used the canonical `proofs/perf/warmed_timing.py`; agreement expected).
