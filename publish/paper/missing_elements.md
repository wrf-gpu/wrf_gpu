# Missing Elements — v0.1.0 Paper Feedback List

Status: drafted 2026-05-31 alongside the rewritten `publish/paper/paper.md`. This is the manager's cross-reference list: every placeholder, [CITE], [VERIFY], and unfinished verification in the paper, plus exactly what is needed to fill it. Grouped by effort: **(i)** have-the-proof, just needs rendering; **(ii)** needs new work for 0.1.0; **(iii)** v0.2.0 / later.

Conventions: each item gives the paper location, what is needed, the proof object / table / figure / citation that fills it, and the owner type (table-worker, validation-run, citation-check, release-hygiene, human).

---

## (i) Have-the-proof — just needs rendering / re-grounding

These are blocked only on a render or a re-point at current proof objects; the underlying evidence exists on disk.

1. **§3.2 / §3.5 — model-role timeline figure.** Need `publish/figures/model_role_timeline.png` (stage on x-axis, role band per model: GPT-5.5-built foundations → Opus-4.7/4.8 manager handoff → GPT-5.5→Opus-4.8 implementer handoff; verifier every-sprint→every-milestone). Source: git history + human author notes §2. The existing `publish/figures/timeline.md` is an M7-era spec and must be regenerated. Owner: table/figure-worker.

2. **§5 — validation-pyramid figure.** Regenerate `publish/figures/validation_pyramid.png` from the existing `publish/figures/validation_pyramid.md` spec, updated to v0.1.0 status (Tier 1–3 green; Tier 4 d02 green, d03 marginal, TOST = future). Owner: figure-worker.

3. **§6.1 — idealized figure set + table.** Render `publish/figures/warm_bubble_panel.png` and `publish/figures/straka_density_current_panel.png` from existing PPM/plot outputs under `proofs/f7n/`, `proofs/sprintU/close_gate/`, `proofs/wind/idealized_postfix/plots`. Render `publish/tables/idealized_gate_summary.md` (columns: case, reference target, GPU metric, pass/fail, proof path). Proofs exist: `proofs/sprintU/close_gate/{warm_bubble,density_current}_verdict.json`, `proofs/f7/DYCORE_STATUS.md`. Owner: table/figure-worker.

4. **§6.2 — d02 validation table.** Render `publish/tables/v010_d02_validation.md` via `proofs/v010_validation/render_table.py --result proofs/v010_validation/v010_d02_result.json`. Must show full-domain AND Tenerife-box RMSE at 6/12/24/48/72 h for T2/U10/V10/PRECIP, with persistence-skill columns for T2/U10/V10, for all three cases. NOTE: the inline case1 numbers in the paper were read directly from the proof JSON and must match the rendered table; reconcile. A parallel worker is generating this — confirm path. Owner: table-worker.

5. **§6.3 — d03 status + before/after table.** Render `publish/tables/v010_d03_status.md` from `proofs/v010_validation/d03_summary_run24h_v5fix.json` (final-lead RMSE/threshold/within/beats-persistence) plus a self-correction row from the pre-fix d03 summary (`d03_summary_run24h_v2.json` or the earliest BOUNDED_FAIL with ~10.8 K T2) to show the boundary-pump bias collapse. Owner: table-worker.

6. **§6.4 — roofline figure + performance/refutation tables.** Render `publish/figures/roofline_dycore.png` from `proofs/perf/roofline_costonly.json` + `phase_breakdown.json` (dycore AI 0.40, fp64/fp32 ridges, 5.3x-over-floor). Render `publish/tables/performance_current.md` (replaces stale `performance_evolution.md`) and `publish/tables/optimization_refutations.md` (rows: fp32 dynamics, command-buffer, fp32 Thompson, implicit sed, sed-unroll, acoustic-unroll; columns: measured effect, fidelity verdict, proof path). All numbers exist in `publish/runtime_optimization_analysis.md` and `proofs/perf/`, `proofs/thompson_perf/`. Owner: table/figure-worker.

7. **§6.5 — self-correction timeline figure.** Optional but author-requested elsewhere: render `publish/figures/self_correction_timeline.{md,png}` with events: v0.0.1 over-claim → self-compare retraction → dycore F7 close → speedup-denominator correction → persistence-baseline wind gap → missing-Coriolis fix → d03 boundary-pump fix → current v0.1.0 status. Sources are all in the §3.6 error-catch ledger. Owner: figure-worker.

8. **§2.2 — comparator table re-grounding.** `publish/tables/comparators.md` exists but is sourced from the old `publication/research_brief`; re-confirm each row's citation key resolves in `publish/paper/references.bib` and that the wording stays "context, not normalized benchmark." Owner: citation-check.

9. **§9.3 — workflow visualization reconciliation.** The mermaid + ASCII loop diagrams are drafted inline in §3.6; render a clean `publish/figures/workflow_loop.{md,png}` and reconcile role names with the final process-metrics ledger. Owner: figure-worker.

10. **§9.4 — publication audit script + manifest.** Update `scripts/m7_publication_audit.sh` to target `publish/paper/` (currently targets `publication/draft`) and the current v0.1.0 proof objects; run it and paste output to `publish/manifest/publication_audit_v1.json`. Owner: release-hygiene.

---

## (ii) Needs new work for 0.1.0

These require a fresh run, a new artifact, a decision, or a citation that does not yet exist — and they gate an honest v0.1.0 release.

11. **§3.5 — AI process-metrics ledger.** `publish/tables/ai_process_ledger.md` does not exist (the M7-era `publish/tables/sprint_ledger.md` is a seed). Need a regenerated table from git history: per-stage sprint count, role, model, objective, proof objects produced, verdict, major claim affected. GPT idea-doc P10 gives a starter script over `.agent/sprints/`. **Mandatory** because the AI-methodology section is the headline. Owner: process-ledger-worker.

12. **§3.5 — effort accounting.** `publish/tables/effort_accounting.md` does not exist. Need: agent-runs/sprints per stage (the honest unit per human author notes §3 — nightly free-token runs, not 24/7), an approximate total-token count, wall-clock span from nothing→v0.0.1→v0.1.0→publication EXCLUDING the dead earlier attempt, and the cost envelope (€200/mo Claude Max + €100/mo GPT Pro, plausibly ~€100, no funding). Owner: process-ledger-worker + human (cost numbers).

13. **§6.4 / §9.3 — fresh v0.1.0 D2H transfer audit.** The 0-byte in-loop D2H proof cited is the historical M7 `d2h_audit_v2.json`. Re-run the transfer audit on the current v0.1.0 operational path and emit a v0.1.0 proof object. Owner: validation-run (GPU; manager — NOT this drafting worker).

14. **§6.4 / §9.3 — repeatability + restart proofs.** Current `proofs/v010_validation/repeatability.json` and `restart_in_pipeline.json` are `status: NOT_RUN` (flags not requested). Re-run the d02 pipeline with `--repeat` and `--restart-at-hour` to produce real proof objects, or remove those systems claims. Owner: validation-run (GPU; manager).

15. **§6.3 / §8 — d03 strict-gate resolution OR downscope.** Current `d03_summary_run24h_v5fix.json` is `D03_1KM_BOUNDED_FAIL` (T2 RMSE 3.01 K vs 3.0 K threshold). Per GPT release gate #2 and the paper's claim boundary: either land a passing d03 proof (close the daytime surface-flux/HFX warm bias, P0-6/P1-4) or keep 1 km out of the positive claim (the paper currently does the latter — confirm this is the manager's release decision). Owner: validation-run / manager decision.

16. **§9.1 — public repo URL, release tag, exact commit.** Placeholders. Need the public GitHub URL, the `v0.1.0` tag, and the exact release commit hash (current validated HEAD `5319b8d` + d02/d03 fixes through `234265a`; confirm the final commit). Owner: release-hygiene + human.

17. **§9.2 — pinned environment manifest.** Need exact Python / JAX / jaxlib / CUDA / driver / XLA flags / OS at the release commit (package currently declares only `python>=3.10`, `jax>=0.4`). Owner: release-hygiene.

18. **§9.4 — data/fixture availability statement.** Need a concrete statement: which Gen2/CPU-WRF corpus fixtures and AEMET observations ship with the release vs are too large/licensed and are described for regeneration. Owner: human + release-hygiene.

19. **§1 — [VERIFY] AEMET/HARMONIE-AROME adequacy claim.** The Canary-microclimate-adequacy claim (human author notes §4) needs a citable AEMET/HARMONIE-AROME product-resolution reference or must be softened to "in the author's operational experience." **NEW bibkey likely needed** (e.g. `aemet_harmonie_arome`). Owner: citation-check + human.

20. **§1 / §2.1 — [VERIFY] prior abandoned open-source GPU-WRF attempt + commercial-variant completeness.** The "to our knowledge, first" framing and the "never open-sourced" reflection need either a citation to the prior abandoned attempt or an explicit "to the best of our knowledge" hedge. **NEW bibkey possible.** Owner: citation-check + human.

21. **§9.5 — independent human numerical-methods review.** Disclosed as a limitation for arXiv (acceptable); REQUIRED before any journal submission per GPT release gate #9. Owner: human.

22. **Stale-text purge (cross-cutting).** Before release, confirm no surviving stale claims anywhere the paper or release notes are assembled: the old 22.26x in README "Core goals" line, the M7-era `publish/tables/{performance_evolution,skill_evolution,m7_gates}.md`, the old `publication/draft/` paper, and `publish/paper/honesty_audit.md` (M7-era — must be refreshed so every quantitative claim in the new paper has a current proof path). The new `paper.md` treats 22.26x/50.20x/156.82x ONLY as retracted self-correction history. Owner: manager + honesty-audit-refresh-worker.

---

## (iii) v0.2.0 / later (referenced as future work, not a 0.1.0 blocker)

These appear in the paper only as roadmap/future-work and do not gate the v0.1.0 release.

23. **§5 / §7 — seasonal-ensemble TOST equivalence.** The formal ≥15-case seasonal TOST equivalence test is named as future work, not claimed. Backfill the corpus and run it for a v0.2.0 / journal version. Sources: `proofs/m20/` (case manifest, tost_design, seasonal_gap_assessment). Optional table `publish/tables/tost_readiness.md`. Owner: v0.2.0.

24. **§7 — differentiability / ML-hybrid / DA demonstration.** Stated as a structural JAX property only; any actual gradient/DA/ML-hybrid result is v0.2.0+. Do NOT let this creep into a 0.1.0 claim. Owner: v0.2.0.

25. **§8 — P0/P1 roadmap items as evidence.** P0-1 (live nesting), P0-3 (prognostic Noah-MP), P0-4 (d01 cumulus), P0-2 (native init), S1 (multi-GPU), and the P1 fidelity debts are roadmap, each to close with its own proof object per `.agent/decisions/POST-0.1.0-ROADMAP.md`. They belong in a v0.2.0 model-description / second paper. Owner: v0.2.0.

26. **§6.2 — precipitation verification.** Precipitation is reported as a diagnostic/limitation (RMSE grows with lead, persistence skill poor). A dedicated precip verification (FSS/SAL/event corpus) is future work before any precip skill claim. Sources: bibkeys `roberts2008scale`, `wernli2008sal` already in bib. Owner: v0.2.0.

27. **§8 — Hopper/H200 re-measurement.** Portability is argued (recompiles, expected faster), but speedup-vs-CPU on Hopper must be measured on that hardware (none available). Owner: v0.2.0 (hardware-gated).

---

## Citation status summary

All 29 [CITE: ...] keys used in the draft already resolve in `publish/paper/references.bib` (40 keys total). No new bibkeys are required for the existing citations. The two **[VERIFY]** flags (items 19, 20) may each require ONE new bibkey:

- `aemet_harmonie_arome` — Canary regional-product resolution/adequacy (item 19), OR soften the claim.
- `prior_gpu_wrf_attempt` — the abandoned open-source GPU-WRF attempt (item 20), OR keep the "to the best of our knowledge" hedge.

If these references cannot be sourced, both claims must be softened in-text rather than cited; the paper is already written to degrade gracefully (the hedges are present).
