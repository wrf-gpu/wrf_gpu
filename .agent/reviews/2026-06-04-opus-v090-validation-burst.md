# v0.9.0 CORE Validation + Benchmark Burst

**Lane:** worker/opus/v090-validation-burst (branched from worker/opus/v090-release-trunk @ 2162e04, the merged 7-branch trunk)
**Date:** 2026-06-04
**Mode:** WRF-faithful, ADR-007 gated-fp32 (theta/u/v/qv fp32; mu/p/ph/w + acoustic/pressure accumulators fp64) — the OPERATIONAL SHIP mode.
**Resource:** GPU lock claimed (preserving cpu_cores_4_31 backfill); orchestration pinned to cores 0-3; ONE GPU job at a time.
**Honesty policy:** report real skill numbers; do not inflate.

---

## Objective

Close the two genuinely-open 0.9.0 gates and fill the honest speedup:
- **(A)** d02 multi-hour coupled SKILL vs CPU-WRF (only finiteness was previously confirmed).
- **(B)** d03 1km validation with the new faithful physics.
- **(3)** Fill the real-user-time speedup benchmark (9/3km nested + 1km).

The d02-replay hour-1 blow-up is FIXED in the merged trunk (validated stability namelist
epssm=0.5/damp_opt=3/w_damping=1/diff_6th_opt=2/zdamp=5000/dampcoef=0.2/top_lid=True via
fix-B, plus the MYNN qke cold-start seed via qkefix-followup). d02 finite through 3h and d03
finite at 24h were already confirmed pre-burst.

---

## Precision-mode note (why a harness change was needed)

The merged-trunk d02/d03 replay scripts hardcoded `force_fp64=True`. The 0.9.0 SHIP mode is
ADR-007 gated-fp32. Added a `--gated-fp32` CLI flag to both `scripts/m7_l2_d02_replay.py` and
`scripts/d03_replay.py` (default stays full-fp64; flag flips a module `_FORCE_FP64` that the
case-builder reads, since `execute_daily_pipeline` fixes the case_builder signature). Verified
the gated matrix in `src/gpuwrf/contracts/precision.py` (theta/u/v/qv/q*/qke = FP32_GATED;
mu/p/ph/w/ustar/fluxes = FP64) matches ADR-007.

## Corpus / reference setup

- d02 CPU truth: backfilled 28-rank CPU-WRF v4.7.1 L2 d02 wrfout in
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/`. These dirs are wrfout-only (the
  matching `wrf_l2/<run_id>` dirs have `namelist.input` but the wrfout was purged — that is why
  the backfill regenerated them). Built a composite staging dir `/tmp/vburst_runs/<run_id>` that
  symlinks the backfill wrfout + the matching corpus `namelist.input`/`wrfinput` (the loader reads
  grid metadata from the namelist). Honest: real CPU-WRF wrfout + the real namelist that produced it.
- Selected d02 case: **20260507_18z_l2_72h_20260513T124307Z** — representative stable mid-season
  (MAM) case, d02 mass grid 66x159 (the canonical L2 d02 grid; some backfill runs are a smaller
  66x120 sub-domain and were excluded).
- d03 CPU truth: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z`
  (intact: namelist + 25 d02 + 25 d03 wrfout; d03 1km mass grid 75x93, the Tenerife domain).

---

## PART 1 — d02 coupled multi-hour SKILL vs CPU-WRF

### 1h gated-fp32 smoke (path confirmation)
- Verdict **L2_D02_GREEN**, all statuses PASS, finite throughout.
- Final-hour Tier-4 RMSE vs CPU-WRF: **T2 0.492 K (bias -0.315), U10 0.427 m/s (-0.155), V10 0.325 m/s (-0.008)** — well inside bars (T2<3.0, U10/V10<7.5).
- Wall: total 561.8 s; hour-1 (compile-inclusive) 486.7 s.

### Multi-hour (24h) gated-fp32 — coupled skill
<!-- FILLED FROM proofs/v090/d02_coupled_skill.json + speedup_d02 -->
- Finite throughout: TBD
- Per-lead T2/HFX/U10/V10/PBLH RMSE + bias: TBD
- Within operational margins (vs v0.1.0/v0.2.0 bars): TBD
- Wall clock (command-to-finish): TBD

---

## PART 2 — d03 1km validation (new faithful physics)
<!-- FILLED FROM proofs/v090/d03_1km_validation.json + d03_prognostic_pblh.json -->
- Every output timestep within margins (T2/U10/V10/PBLH/precip + prognostic levels): TBD
- Finite/stable throughout: TBD
- Worst field: TBD
- Wall clock: TBD

---

## PART 3 — honest real-user-time speedup
<!-- FILLED FROM proofs/v090/speedup_benchmark.json -->
- 9/3km nested (d02, compile-inclusive headline): TBD
- 1km (d03, INDICATIVE): TBD
- compile caveat: TBD

---

## Risks / honest gaps
- TBD (e.g. if throughput capped the horizon below 72h).

## Files changed / proofs
- scripts/m7_l2_d02_replay.py, scripts/d03_replay.py (--gated-fp32 flag)
- proofs/v090/d02_coupled_skill_analyze.py, proofs/v090/d03_prognostic_pblh_analyze.py
- proofs/v090/d02_coupled_skill.json, proofs/v090/d03_1km_validation.json,
  proofs/v090/d03_prognostic_pblh.json, proofs/v090/speedup_benchmark.json (filled)
