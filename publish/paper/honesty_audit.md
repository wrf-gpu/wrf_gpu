# Publication Honesty Audit — v0.1.0

Date: 2026-05-31 (refreshed for the v0.1.0 paper on branch `worker/opus/final-verdict`).
Scope: `publish/paper/paper.md`. This supersedes the M7-era audit that scoped
`publication/draft/paper.md`; the old M7 station-RMSE / 22.26× rows are retired (the
22.26×/50.20×/156.82× figures now appear in the paper ONLY as retracted self-correction
history, §3.6 / §6.5).

Every load-bearing quantitative claim in the paper maps to a CURRENT proof object or its
companion generated table. The 9 current evidence tables are:
`performance_current.md`, `v010_d02_validation.md`, `v010_d03_status.md`,
`idealized_gate_summary.md`, `optimization_refutations.md`, `wind_persistence_skill.md`,
`systems_invariants.md`, `v010_claim_boundary.md`, plus the process pair
`ai_process_ledger.md` / `effort_accounting.md`.

| Claim in v0.1.0 paper | Current proof object / table | Status |
|---|---|---|
| Idealized warm bubble PASS 6/6 (θ′ 1.920 K, max\|w\| 11.68, rise 1924 m, mass drift ≈0) | `proofs/f7n/skamarock_bubble_diagnostics.json`; `proofs/sprintU/close_gate/warm_bubble_verdict.json`; table `publish/tables/idealized_gate_summary.md` | backed |
| Idealized Straka density current PASS 6/6 (front 14 150 m, θ′min −9.971 K, max\|w\| 14.575, mass 2.25e-9) | `proofs/f7n/straka_density_current_diagnostics.json`; `proofs/sprintU/close_gate/density_current_verdict.json`; table `idealized_gate_summary.md` | backed |
| Operational entry bitwise-identical to idealized harness over 50 warm-bubble steps | `proofs/f7/DYCORE_STATUS.md` (Sprint U P0-1) | backed |
| d02 (3 km) runs stable/finite to 72 h; verdict D02_VALIDATED, all 3 cases | `proofs/v010_validation/v010_d02_result.json` (HEAD `5319b8d`); table `publish/tables/v010_d02_validation.md` | backed; release gate = re-run on final commit (VERIFICATION.md row 4) |
| d02 case1 surface RMSE (T2 1.06–2.10 K, U10 1.51–1.80, V10 1.70–2.38, PRECIP ≤1.56 mm) | `proofs/v010_validation/v010_d02_result.json`; table `v010_d02_validation.md` | backed |
| d02 U10/V10 positive mean persistence skill in every case/region (not every lead; case3 V10 −0.13→+0.17) | `proofs/m19/verdict_result.json`; `proofs/wind/coriolis_fix_verdict.md`; table `publish/tables/wind_persistence_skill.md` | backed; narrowed (mean, not every-lead) |
| d02 T2 skill mixed; PRECIP loses to persistence at every lead (diagnostic, not skill) | `v010_d02_result.json`; table `v010_d02_validation.md` | backed; explicitly not a skill claim |
| d03 (1 km) 24 h verdict D03_1KM_VALIDATED (PASS); final-lead T2 1.92 K (gate 3.0), U10 3.45, V10 4.24 (gate 7.5) | `proofs/v010_validation/d03_summary_run24h_hfxfix4.json`; `d03_validation_run24h_hfxfix4.json`; table `publish/tables/v010_d03_status.md` | backed |
| d03 field qualifiers: T2 beats persistence most leads (final +0.16); V10 most leads (+0.09); U10 short leads only, loses at long leads (final −0.16); no field beats persistence at every lead | `d03_validation_run24h_hfxfix4.json` per-lead block (`persistence_beat_all_leads` all false); table `v010_d03_status.md` | backed; reported with qualifiers |
| HFX repair: land over-flux 4.22×→2.30×, T2 land bias +3.6 K→+1.2 K (all-cell T2 RMSE 2.106→0.827 K) | `proofs/v010_validation/sfclay_hfx_oracle_parity.json` | backed; labelled EMPIRICAL PARTIAL (3 known MYNN formula mismatches, §6.3) — NOT a faithful `module_sf_mynn.F` port |
| Speedup ~5.29× clean / ~7.84× realistic / ~3.2× dt-matched floor (d02, fp64) | `publish/runtime_optimization_analysis.md`; `proofs/perf/{roofline_costonly.json,speedup_denominator.md,compute_cycle_analysis.md}`; table `publish/tables/performance_current.md` | backed |
| Dycore AI ≈0.40 FLOP/byte; ~18.7% HBM BW; ~8.2% fp64 peak; ~11k ops/step; memory/launch-bound | `proofs/perf/{roofline_costonly.json,phase_breakdown.json}`; table `performance_current.md` | backed |
| Four optimization levers measured-and-refuted (fp32 ~1.0×, graph-capture 0.83–0.87×, fp32-Thompson ~1.0×, implicit sed 2.25–2.44× kernel but REJECTED +47% over-precip) | `proofs/thompson_perf/{PRECIP_ORACLE_AND_IMPLICIT_SED.md,kernel_lever_summary.json,coupled_timing_base_vs_opt.json}`; table `publish/tables/optimization_refutations.md` | backed |
| Pipeline end-to-end 9.09× vs derived CPU d02 denominator (1794 s vs 16 305 s) | `proofs/v010_validation/speedup_vs_cpu_24h.json`; table `publish/tables/systems_invariants.md` | backed; supporting pipeline evidence, not the like-for-like headline |
| 24 h coupled d02 all-finite guards-OFF at fp64; peak device memory length-independent (10 211 MB after 24 h ≈ 9 048 MB after one segment) | `proofs/perf/coriolis_segscan_24h.json`; table `systems_invariants.md` | backed |
| Counted in-loop D2H transfer audit; repeatability; restart-continuity | audit script `proofs/perf/fusion_transfer_audit.py` exists; `repeatability.json` / `restart_in_pipeline.json` are `NOT_RUN` | NOT YET RUN — paper carries explicit release-gate wording (VERIFICATION.md rows 8, 11); historical M7 audit reported 0 copies / 0 bytes |
| Statistical equivalence (TOST) | harness + scorer self-tested CPU-vs-CPU to 0.00 paired delta: `proofs/m20/{tost_design.json,selftest_verify_release.json,tost_campaign_plan.md}`; table `tost_readiness.md` | NOT a v0.1.0 claim; corpus = 3 distinct MAM days → underpowered single-season descriptive only |
| Process metrics: ≈12.6 calendar days, 884 commits, 249 sprint dirs, ≈500–700 agent-runs, order-10⁸ tokens, ≤€300/mo subscriptions | `publish/tables/ai_process_ledger.md`; `publish/tables/effort_accounting.md`; git history | backed (structural proxies; no token meter — stated plainly) |
| Claim boundary (what v0.1.0 IS / IS NOT) | table `publish/tables/v010_claim_boundary.md`; `publish/GPU_PORT_GAPS_TODO.md`; `.agent/decisions/POST-0.1.0-ROADMAP.md` | backed |
| Comparator context table (6 bibkeys resolve in references.bib) | `publish/tables/comparators.md` | backed; context only, not a normalized benchmark |
| Public repo URL / `v0.1.0` tag / exact release commit / environment manifest | not yet cut | release-time placeholders (`[MISSING RELEASE ITEM]` markers in §9) |

Citation integrity: all 29 unique inline `\cite{...}` keys in `paper.md` resolve in the
40-entry `references.bib`; no orphans. The 6 bibkeys used in `publish/tables/comparators.md`
also resolve.

Narrowed / softened in this refresh (vs the M7-era audit):

- The HFX/MYNN fix is described as an **empirical partial MYNN-inspired land thermal-roughness
  repair (full MYNN parity pending)**, NOT a faithful `module_sf_mynn.F` port; the three known
  formula mismatches (zol solved with momentum znt before the z_t block; restar from a
  blended/look-ahead ustar; psih2/psih10 on the thermal not the momentum baseline) are cited.
- The abstract no longer asserts a counted "zero-in-loop device-transfer audit"; it says a
  device-residency audit exists with the counted in-loop transfer count pending.
- d03 is now `D03_1KM_VALIDATED` (passes the bounded gate) but is reported with per-field
  persistence qualifiers and kept secondary to d02 because the unblocking surface-layer repair
  is empirical.
- The M7-era 22.26×/50.20×/156.82× speedups and pre-Coriolis station-RMSE deltas are retired
  to self-correction history; the current headline is roofline-grounded ~5.29×/~7.84×/~3.2×.
- The five M7-era tables (`performance_evolution`, `skill_evolution`, `m7_gates`,
  `sprint_ledger`, `test_coverage`) carry a STALE/RETIRED banner and are not cited by the paper.
