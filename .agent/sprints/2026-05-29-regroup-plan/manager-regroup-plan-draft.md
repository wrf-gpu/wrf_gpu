# Manager (Opus 4.8) regroup plan — DRAFT (to synthesize with GPT-5.5 blind plan)

## State delta vs reset plan
Reset plan (M8-M23) assumed dycore done; it wasn't (self-compare lie). F-series (A-N) = the real dycore rebuild. NOW: dry dycore ~95% (warm bubble PASS 6/6, Straka matches WRF to 3% til touchdown, F7N closing the touchdown residual), validated vs WRF ground truth. New reusable assets: pristine WRF v4.7.1 + per-operator savepoint methodology (PROVEN); parallel multi-agent workflow; honest gates; diagnostic harness.

## The single biggest efficiency unlock
**Apply the WRF-savepoint ground-truth methodology to EVERY physics scheme from sprint 1.** The dycore was slow because we debugged blind until ground truth arrived late. Each physics scheme (microphysics/PBL/radiation/land) gets its own pristine-WRF per-operator savepoint oracle from the start → no blind debugging. This is the lever that makes physics FASTER than the dycore.

## Parallelization (physics couplers are COLUMN schemes = independent)
Once the M11 physics↔dycore coupling interface is defined, run these as CONCURRENT Opus frontrunners, each with its own WRF-savepoint oracle:
- M17 Thompson microphysics
- M12 MYNN PBL + surface layer
- M13 RRTMG radiation + land-surface diurnal
- M16 Noah/Noah-MP land (the big lift — possibly deferred, see cut list)

## Critical path (NOT physics breadth — that's parallel)
1. **M11 physics→RK1-bundle cadence** = the coupling INTERFACE; gates all physics. DO FIRST (dycore-runtime).
2. **M9 operational-mode savepoint parity** (divergence map across SWDOWN/GLW/HFX/LH/PBLH/TSK/T2/U10/V10/PSFC) — defines the operational gate; partly built.
3. **M19 single-case L2/L3 skill recovery** — integration gate (needs all physics coupled+correct).
4. **M20 validation corpus (≥30 Canary L2/L3 seasonal cases, IC/BC on disk + CPU WRF baseline)** = THE LONG POLE; data/compute-bound, NOT AI. START NOW as a parallel track (Gen2 baseline + AIFS; ~1 month of solutions exist per canairy_meteo memory).
5. **M21 TOST equivalence** (needs corpus + skill).

## Perf sequencing (F7-perf)
Dycore grew many ops → 22× (on incomplete dycore) projects to ~10-15×. Do a perf pass AFTER dycore close to lock GPU-efficiency of the correctness-locked dycore (XLA fusion + fp32 in safe regions), then physics adds ops, then FINAL M22 recert. Keep correctness and perf separated (proof discipline).

## Cut/defer list for v0.1.0
- **Noah-MP (M16)**: the 8-14 wk big lift. ADR-030-conditional. DECISION at M19: if the skill gate passes with simpler Noah (not Noah-MP) or prescribed land, defer Noah-MP to v0.2.0 (saves 8-14 wk). v0.1.0 = Canary skill equivalence, land scheme = whatever meets the gate.
- Full 362-var WRF I/O — keep the 41-var minimum (done).

## Biggest risks + early de-risk
- Validation corpus (long pole) → start building NOW, verify ≥30 cases IC/BC + CPU WRF baseline on disk.
- Physics↔dycore coupling (M11) → do first, validate vs WRF savepoints.
- Perf collapse after correctness → perf pass early + M22 recert.
- Overfit to one case → M19 + INV-4 mini-ensemble (median RMSE non-increase).

## Next 3-5 sprints the moment the dycore closes (parallel where marked ‖)
1. M9 op-mode savepoint parity audit (extend savepoint methodology to operational forecast path).
2. M11 physics→RK1-bundle coupling cadence (the interface) — Opus.
3. ‖ M20-prep: validation-corpus kickoff — verify ≥30 Canary cases IC/BC + CPU WRF baseline on disk (data track).
4. ‖ (after M11 interface) parallel physics: M17 Thompson + M12 MYNN/surface, each Opus + per-scheme WRF savepoint oracle.
5. ‖ F7-perf: dycore GPU-efficiency pass (XLA fusion + fp32 safe-region) — lock the speedup baseline.

## Honest timeline note
Reset said 32-45 wk. Dycore is now CORRECT (not fake-done). Physics phase should be faster (parallel + ground-truth-from-start). Dominant remaining cost = validation corpus + (conditional) Noah-MP + TOST. If Noah-MP deferred, v0.1.0 (Canary skill equiv w/ Noah) is meaningfully sooner.

## To reconcile with GPT blind plan
Compare on: (a) critical-path ordering (is M11-coupling-first right?), (b) Noah-MP defer decision + its trigger, (c) perf sequencing, (d) corpus-as-long-pole + start-now, (e) any milestone GPT would cut/add, (f) the concrete next-sprint set.
