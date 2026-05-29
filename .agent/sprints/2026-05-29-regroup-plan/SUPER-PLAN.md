# Revised Roadmap (SUPER-PLAN) — fastest efficient path to v0.1.0

**Synthesis of the GPT-5.5 blind plan (`gpt-regroup-plan.md`) + manager draft (`manager-regroup-plan-draft.md`), 2026-05-29. Strong convergence; this supersedes the serial M8-M23 ordering in PROJECT-RESET-PLAN-FINAL for sequencing (the GOAL + invariants are unchanged).**

## Goal (unchanged)
Canary L2/L3 24-72 h RMSE on T2/U10/V10 statistically equivalent to CPU WRF v4 under TOST on ≥30 seasonal cases; ≥10× speedup vs 28-rank CPU WRF; no shortcuts; GPU-efficient.

## Honest wall-clock from dycore close
- **Without full Noah-MP: 12-16 weeks.** With full prognostic Noah-MP: 20-28 weeks. (Decided by the Noah-MP discriminator, not assumed.)

## Core strategy (the two efficiency unlocks)
1. **Pristine WRF v4.7.1 as a per-scheme ORACLE FACTORY.** Every physics scheme is validated against per-operator WRF savepoints from sprint 1 — no blind debugging (the thing that made the dycore slow). Build is done (`/home/enric/src/wrf_pristine`, env `wrfbuild`).
2. **Parallel lanes.** Physics couplers are column schemes (independent) → run concurrently as separate Opus frontrunners, each owning its code paths (file-ownership map), each gated by WRF-oracle parity. Recompose only after per-scheme deltas are explained.

## Sequence

### Gate 0 — F7 dycore close (2-5 days; IN PROGRESS via F7N)
Straka touchdown residual closed (front ~15 km, min θ′ −9..−10 K), warm bubble PASS 6/6, per-substep WRF touchdown diff shows no unexplained discrepancy, **GPT-5.5 pre-close critique done**, merge f7d chain. Allowed in parallel before close: WRF-oracle planning, corpus discovery, CPU-baseline inventory, perf-measurement scripts (NO Phase-B model-code merge before close).

### Gate 1 — Interface + Oracle Freeze (3-5 days; 1 frontrunner + GPT critique)
Freeze BEFORE dispatching parallel implementers: coupler/state interfaces (physics tendencies, surface/radiation/land/boundary diagnostics); savepoint schema (vars/units/stagger/precision/tolerance ladders/source-run/checksums); one oracle manifest per scheme; **file-ownership map** (no shared-core edits without manager merge); diagnostic-harness scope fix (physically-inactive ≠ missing operator). Proof: `phase_b_oracle_freeze.json`. Plus fold in the M9 operational divergence-map target (SWDOWN/GLW/HFX/LH/PBLH/TSK/T2/U10/V10/PSFC) as the operational gate emerging at recomposition.

### Phase B — Parallel Lanes (3-6 weeks wall-clock; concurrent after Gate 1)
Each lane: own code paths + proof objects; merge only after WRF-oracle parity + a coupled diagnostic-harness run.
- **B1 Thompson microphysics** (2-4 wk): WRF savepoints around `mp_gt_driver` + internal process boundaries (autoconv/accretion, ice nucleation, sat-adjust, evap, melt/freeze, **sedimentation — implement, don't assume irrelevant**). MUST validate on a MOIST/cloudy column pack + a Canary slab (NOT the cloud-free 20260521 start). Owns `physics/thompson_*`.
- **B2 Surface layer + MYNN** (2-4 wk; **likely the first T2/U10/V10 limiter**): `sfclay_pre/post` + `mym_level2/length/turbulence/predict` + `mynn_tendencies` savepoints. HFX/LH hour-1 RMSE must drop ≥70% vs the stale M12 failure; U10/V10/T2 diagnostics match WRF; column budgets close; no hidden guard fallback. Owns `surface_*`,`mynn_*`, bottom-BC adapter.
- **B3 Radiation + diurnal/land driver** (2-3 wk): `radiation_driver`+`RRTMG_SW/LWRAD` savepoints; THREAD model time (no fixed-time fallback); SWDOWN/GLW/heating-rate parity; diurnal T2 within 1 K on pinned case. Owns `rrtmg_*` + radiation/time diagnostics. → **Land discriminator** (below).
- **B4 Static fields + lateral boundaries** (1-3 wk; parallel, NOT after physics): LU_INDEX/HGT/LANDMASK/XLAND/IVGTYP/ISLTYP/VEGFRA/ALBEDO/EMISS/ZNT/SST/soil bitwise-or-explainable vs WRF; full U/V/W/T/QV/P/PB/PH/PHB/MU/MUB boundary + relax-zone width; boundary-strip not the dominant first-hour error. Owns IO/boundary/static loaders.
- **B5 Validation corpus + statistics** (SHORTER than feared; **STARTS IMMEDIATELY — on the M21 critical path, must not wait for M19**): **INVENTORY 2026-05-29: the CPU WRF baseline corpus is LARGELY PRE-BUILT — `/mnt/data/canairy_meteo/runs/wrf_l3` = 36 daily runs, `wrf_l2` = 30 daily runs (20260510→20260528, 18z), AIFS/WPS IC staging present (`Gen2/runs/wps_cases`), existing Tier-4 `data/fixtures/gen2_baseline/rmse_summary.csv`.** So B5 is NOT "build 30 cases" — the cases exist. Remaining B5 work: case_manifest + cpu_baseline_manifest (hashes) + station_join_manifest + tost_design (predeclared margins, power analysis, missing-data) + select 5-case mini-ensemble — all buildable from existing data NOW. **THE REAL GAP = SEASONAL DIVERSITY**: existing cases span only ~3 weeks of May 2026 (not seasonally stratified). Single-case M19 + mini-ensemble + initial TOST run on existing cases immediately; the final "seasonal equivalence" claim needs extended CPU-WRF coverage across the year (a wall-clock/data track, not AI — start the additional Gen2 runs in the background early). Proofs: `case_manifest.json`, `cpu_baseline_manifest.json`, `station_join_manifest.json`, `tost_design.json`.
- **B6 Perf smoke / residency audit** (2-3 days, then held): warmed Nsight `d2h_inter_kernel==0` on operational path; record kernel count/compile/mem/speedup as ESTIMATES; if speedup <7× after dycore close → blocker perf sprint before adding physics, else wait.

### Recomposition Gate — Coupled Phase B Close (1-2 wk)
After B1-B4 pass independently, merge through ONE composed operational path: WRF physics cadence (dycore RK bundle → microphysics → surface → MYNN → radiation → boundary order); tendency accounting synchronized; zero hidden guard clips on valid WRF-range states (all limiter activations logged); diagnostic harness no unexplained MISSING/NOISY_ZERO; conservation (mass/water/energy) proof objects. **First time coupled skill is judged.**

### M19 — Single-case skill recovery (1-2 wk; AFTER recomposition)
Pinned 20260521 L2/L3 d02, same scorer+station mask as CPU WRF: T2/U10/V10 RMSE & MAE within 20% of CPU WRF; 5-case mini-ensemble median not worse than predeclared margins. True blocker order: surface/MYNN flux → land/TSK/diurnal → boundary → moist/radiation → dycore residuals. If pinned passes but mini-ensemble fails → triage by regime, do NOT advance to M21.

### M19-economics perf gate (2-5 days, parallel with M19 scoring)
Warmed Nsight transfer audit + speedup estimate under coupled code. If <8× → targeted perf sprint (XLA fusion + fp32 safe-region) BEFORE spending 30-case GPU time.

### M20/M21 — Corpus + TOST (4-8 wk after corpus ready; M20 mostly done before M19)
Run GPU forecasts for all ≥30 predeclared cases; reuse B5 CPU baselines; score by domain/lead-hour/station/aggregate; TOST with predeclared margins, report CIs + effect sizes. Do NOT tune margins after seeing GPU output; do NOT drop hard cases unless exclusion predeclared.

### M22/M23 — Release (3-5 wk)
M22 final perf recert (warmed Nsight, `d2h_inter_kernel==0`, ≥10× on final code+CPU denominator). M23 v0.1.0 tag + arXiv (evidence first, polish last), ADRs for every architectural change, README aligned, no 156× anywhere.

## Noah-MP decision (the big timeline lever)
Full Noah-MP is NOT the first sprint, but a prognostic land lower boundary probably is (static/prescribed TSK likely inadequate for 24-72 h T2). **Discriminator immediately after B2/B3 first coupled integration**: if land-station T2 diurnal amplitude or HFX/LH parity stays outside thresholds (>1.5 K pinned / >1.0 K median mini-ensemble) → promote full Noah-MP to critical path. Else defer full canopy/snow/groundwater to v0.2.0 with a verified minimal prognostic lower-boundary contract for v0.1.0.

## Cut/defer for v0.1.0
Full general WRF replacement scope; non-Canary domains; precip/gust/RH as RELEASE gates (but precip accumulators must exist for Thompson/land water budgets); arXiv polish; full Noah-MP if discriminator says unneeded; cosmetic refactors/perf-hero work before correctness freeze.
**Do NOT cut:** boundary completeness; static-field parity; surface-flux/land-diurnal evidence; the 30-case corpus + TOST; the transfer audit.

## Top risks → early de-risk
Corpus not ready at M19 (B5 starts now) · static land can't recover T2 (Noah-MP discriminator early) · surface/MYNN dominates U10/V10/T2 (WRF savepoints before coupled debug) · Thompson looks inactive on dry IC (moist column pack) · perf collapse (3 perf gates) · overfit to 20260521 (5-case mini-ensemble + 30-case predeclared) · parallel workers collide (Gate-1 file-ownership map).

## Next sprints the moment F7N closes the dycore
1. **GPT-5.5 pre-close dycore critique** (firm rule) → merge f7d.
2. **Sprint 1: Gate-1 Interface + Oracle Freeze** (frontrunner + GPT critique).
3. Then PARALLEL: **2A Thompson · 2B Surface/MYNN · 2C Radiation+land-discriminator · 2D Static+LBC · 2E Corpus+TOST buildout** (2E can start during Gate 0/1 — corpus discovery + CPU baseline inventory now).
4. **Sprint 3 Recomposition** (after 2A-2D) → **Sprint 4 M19 + mini-ensemble gate** → **Sprint 5 M19-economics perf gate**.

(2E corpus discovery + CPU-baseline inventory + perf-measurement scripts may begin NOW, in parallel with F7N, since they touch no Phase-B model code.)
