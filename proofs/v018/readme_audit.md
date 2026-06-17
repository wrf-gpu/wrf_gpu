# v0.18 README Audit — per-section currency checklist + proof

**Artifact:** `README.md` (public, user-facing, pushed to the public org).
**Branch:** `worker/gpt/v018-integration` (worktree `.wt-v018-integration`).
**Base HEAD audited:** `aedbd6fa` (post-triage clean state; integration
feature-complete, critic-accepted, perf-neutral, suite 0-failed / 38 documented
xfail).
**Auditor:** Opus README worker. **Date (UTC):** 2026-06-17.

The previous worktree README was a **v0.15-titled** file (the worktree base
branched before the v0.16/v0.17 README polish landed on `main`). The v0.18 README
was rebuilt from the **v0.17 polished README** (`main` commit `fc23be70`, the most
current framing: capability-not-single-card-speed, 1 km MEASURED, PROJECTED labels,
"use the manager") and updated to v0.18. Every section below was reviewed.

## Per-section checklist

| Section | Reviewed? | Up-to-date? | What changed for v0.18 |
|---|---|---|---|
| Title / intro (`# wrf_gpu`) | yes | yes | Kept the capability framing; made the fp64-dycore explicit in the intro (precision regime stated up front). |
| What it is good for | yes | yes | Unchanged framing; 1 km MEASURED + cluster/weak-scaling PROJECTED labels kept precise. |
| What it is NOT | yes | yes | Kept; the "not single-card speedup" + "skill-equivalence open gate" points retained; replaced the trailing "ongoing v0.18 work" sentence with the **feature-completeness 50/23/33 statement**. |
| Feature-completeness statement | yes | yes (NEW) | Added: every WRF v4 scheme classified — **50 operational / 23 reference-only-with-real-oracle / 33 documented-boundary+proven-irrelevant**; no scheme silently substituted. |
| First-run-slow box | yes | yes | Kept (~8–12 min cold compile, persistent cache, ~38 min opt-in fuse compile). |
| WRF-v4 identity (cell-for-cell) | yes | yes | Switzerland dashboard regenerated from **v0.18** run data; Canary dashboard regenerated from retained v0.15 GPU finalgate; added the **strictly-more-WRF-faithful Thompson** paragraph (cold-process + warm-process melt/cold-gate fix, qv bit-exact); explicit plot-provenance note (no fabrication). |
| Thompson fidelity note | yes | yes (NEW) | Added: v0.18 Thompson is strictly more faithful than v0.17 (WRF `N0_melt` override + rci/sci cold-gate; cell qv → 1.2e-13 bit-exact). |
| Quickstart (full clone) | yes | yes | Kept the 3-step clone→install→run flow. |
| Quickstart — source-only (NEW, req E) | yes | yes (NEW) | Added a **VERIFIED** cone-sparse-checkout path (`src` + `data/fixtures`) to run without the full repo; actually run fresh end-to-end (clone→sparse→pip install→import→CLI), evidence at `proofs/v018/quickstart_minimal_source_verified.txt`. |
| Use the manager | yes | yes | Unchanged (skill `.agent/skills/managing-sprints`). |
| Performance | yes | yes | Added the **perf-neutral-vs-v0.17** bullet (dual-confirmed; cold-process warm-free; carry-shape reverted bit-identically); kept ~parity default, opt-in fuse ~1.27–1.30×, the launch/occupancy ceiling, fp32-can't-move-it; MEASURED vs PROJECTED kept precise. |
| Apples-to-apples vs AceCAST (NEW, req D) | yes | yes (NEW) | Added a clearly-labeled **EXPECTATION / PROJECTED** note (no head-to-head run; like-for-like ballpark expectation, NOT a competitive claim), citing `proofs/v018/acecast_reconciliation.md`. |
| Whole-Earth-1km-fits-a-rack | yes | yes | Kept PROJECTED (exact memory arithmetic; multi-GPU throughput not shipped). |
| Opt-in env flags | yes | yes | Unchanged. |
| System requirements & resource profile | yes | yes | Added the **runtime-data** row (data/fixtures ~147 MiB required at import); added perf-neutral-vs-v0.17 to the throughput row; VRAM/compile rows kept. |
| Version history | yes | yes | Added the **v0.18.0** headline row (feature-completeness + scheme triage + Thompson fidelity + perf-neutral + experimental K2); repointed v0.16/v0.17 links to proofs that exist in this worktree (release-notes files + hostgap_fix/fp32_verdict proofs are not in the v018-trunk base). |
| Scope at a glance | yes | yes | Updated MP/PBL/CU/RA/Land/Multi-GPU rows for v0.18: aerosol Thompson, Shin-Hong/GBM PBL, CLM4/CTSM fail-closed, RUC/Pleim-Xiu land, experimental K2; the reference-only/documented-boundary tail named in each fail-closed cell. |
| GPU-operational physics menu | yes | yes | Updated to the final v0.18 operational integer sets (mp/cu/bl/sfclay/sf_surface/ra_lw/ra_sw from `scheme_count_no_clobber.json`). |
| Scheme triage (NEW, req B) | yes | yes (NEW) | **Replaced** the obsolete "Roadmap — delta to a complete WRF v4 port" framing with the current triage table: 50 / 23 / 33 (State = 67 leaves), citing `scheme_count_no_clobber.json` + the integration critic. |
| Boundaries — what is NOT claimed | yes | yes | Refreshed to v0.18 carries: RAINNC 5.22 mm class-c, K2 experimental specified-BC lab-only, Shin-Hong TKE-diagnostic; kept KI-9 credibility gate, no-TOST-claim, multi-GPU-unmeasured, fp64-only standalone, free-running edge; added AceCAST-is-an-expectation. |
| Roadmap — remaining work (rewritten, req B) | yes | yes | **Rewritten** from "delta / missing schemes" to "remaining fidelity / robustness / statistical / perf-scale work" (schemes are no longer missing); RAINNC, Shin-Hong, CLM4/CTSM, K2, reference-only-tail-wiring added as explicit items. |
| Core goals (immutable) | yes | yes | Unchanged (immutable). |
| Where to look first | yes | yes | Added rows for the source-only quickstart, the scheme triage, the AceCAST note, the suite_triage; updated identity-proof + performance rows to v0.18 paths. |
| Known issues (v0.18.0) (req C) | yes | yes | **Replaced** the v0.15/v0.17 table with the true v0.18 carries ONLY: RAINNC 5.22 mm class-c, CLM4/CTSM boundary, K2 experimental, Shin-Hong PBL11 TKE, **38 documented-xfail → `suite_triage.md`**; kept the still-open KI-9/KI-3/KI-4/KI-5/KI-6/KI-7/KI-10/KI-11; dropped everything resolved. |
| Layout | yes | yes | Added the `data/fixtures/` (vendored runtime tables) entry. |

**All sections reviewed; no section left stale.** The most-important
user-facing information is present: what it is / is not, how to install + run
(full and source-only), the cell-identity proof with refreshed plots, performance
(perf-neutral vs v0.17, capability-not-speed, AceCAST expectation), the
50/23/33 scheme triage, the v0.18 known-issue carries, and the remaining roadmap.

## Requirement-by-requirement proof

- **(A) Audit every section + checklist.** This file. 30 README sections, each
  reviewed (table above).
- **(B) Obsolete "Roadmap — delta to a complete WRF v4 port" removed/rewritten.**
  Replaced by (1) the **Scheme triage** subsection with the FINAL counts **50
  operational / 23 reference-only-with-oracle / 33 documented-boundary+proven-
  irrelevant (State = 67 leaves)**, and (2) a **rewritten Roadmap** framed as
  remaining fidelity/robustness/stat/perf work, not missing schemes. Counts from
  `proofs/v018/scheme_count_no_clobber.json` + `integration_honesty_critic_opus.md`
  (FINAL SCHEME-CLASS SUMMARY).
- **(C) Known issues updated to true v0.18 carries.** RAINNC 5.22 mm class-c
  (`rainnc_qvapor_status.json`); CLM4/CTSM v1.0 boundary fail-closed
  (`lsm_family_status.json`); K2 experimental specified-BC lab-only
  (`k2_multigpu_report.md`, `k2_flag_off_graph.json`); Shin-Hong PBL11
  TKE-diagnostic (`schemes_critic_opus.md`); the **38 documented-xfail →
  `proofs/v018/suite_triage.md`**. Resolved items dropped.
- **(D) AceCAST note — EXPECTATION/PROJECTED.** Added under Performance, clearly
  labeled "EXPECTATION / PROJECTED — not measured", no established-competitive
  claim, citing `proofs/v018/acecast_reconciliation.md`.
- **(E) VERIFIED source-only quickstart.** Cone sparse-checkout of `src` +
  `data/fixtures`, `pip install -e .`, `import gpuwrf`, `python -m gpuwrf.cli run
  --help` — all run fresh end-to-end; evidence
  `proofs/v018/quickstart_minimal_source_verified.txt`.
- **(F) Identity plots refreshed + included.** Switzerland from the **retained
  v0.18 72 h GPU run** (genuine v0.18 data, 9/10, RAINNC the miss); Canary from the
  **retained v0.15 GPU finalgate** (dynamics/thermo core, unchanged in v0.18, 9/10,
  QVAPOR the miss), each labeled with its provenance. **No v0.18 Canary 72 h run
  exists; that is stated plainly, not fabricated.** 5 plots per region committed:
  - `docs/assets/v018/identity_proof/switzerland_d01/{identity_dashboard,identity_scatter_1to1,identity_scoreboard,identity_spatial_diff_maps,identity_timeseries_rmse_bias}.png`
  - `docs/assets/v018/identity_proof/canary_l2_d02/{identity_dashboard,identity_scatter_1to1,identity_scoreboard,identity_spatial_diff_maps,identity_timeseries_rmse_bias}.png`
  - manifests under `proofs/v018/identity_proof/{switzerland_d01,canary_l2_d02}/identity_proof_manifest.json`.
- **(G) "honest/honestly/honesty" removed from README prose.** Zero prose
  occurrences remain; the single surviving match is the on-disk proof filename
  `integration_honesty_critic_opus.md` inside a link path (the file is not
  renamed). Verified by grep.

## Cross-cutting framing checks (all PASS)

- **More-WRF-faithful Thompson fix** — stated in the identity section (cold-process
  + warm-process melt/cold-gate; qv bit-exact).
- **Perf-neutral vs v0.17** — stated in Performance + the resource table + Known
  issues, citing `perf_neutrality_FINAL.md`.
- **fp64 dycore** — stated in the intro, Performance, and the fp64-only-standalone
  boundary.
- **1 km capability + cluster weak-scaling (not single-GPU speed)** — the headline
  framing throughout (What-it-is-good-for, Performance HEADLINE, version history).
- **MEASURED vs PROJECTED** — labels kept precise (1 km fits one card MEASURED;
  whole-Earth/multi-GPU-throughput/energy PROJECTED; AceCAST EXPECTATION).

## Link / anchor integrity

All relative file/dir links in `README.md` resolve in this worktree (audited:
every `](path)` target exists on disk). All 6 internal `#anchor` links resolve to a
header slug (GitHub slug algorithm). Two referenced families that are NOT in the
v018-trunk base (`RELEASE_NOTES_v0.16/17.0.md`, `proofs/v017/hostgap_fix_opus.md`,
`proofs/v016/fp32_verdict/`) were **repointed to proofs that exist here**
(`proofs/v017/analyze_hostgap_arm.py` + `run_all7_hostgap_arm.sh`,
`proofs/v016/coverage/` + `coverage_map.json`) so the public README has no dead
links.

## Note for the release double-check

- `docs/KNOWN_ISSUES.md` in this worktree says "**36** xfailed" while the
  authoritative `proofs/v018/suite_triage.md` says **38**. The README uses **38**
  (per the suite_triage authority and the sprint requirement). KNOWN_ISSUES.md is
  outside this README sprint's file ownership; flag for the sanitize/release step
  to reconcile 36→38.
- `pyproject.toml` and `src/gpuwrf/__init__.py` were bumped **0.15.0/0.14.0 →
  0.18.0** so the README quickstart's `gpuwrf.__version__` reports correctly.
