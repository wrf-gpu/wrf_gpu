# Missing Elements — v0.1.0 Paper Cross-Reference List

Status: refreshed 2026-06-01 after the **groundwork paper rewrite** (branch
`worker/opus/final-verdict`), the completed GPU verification campaign
(`proofs/PROOF_TABLE.md`: 9 PASS / 1 comparator-harness FAIL / 1 device-residency
INCONCLUSIVE), and the figure renders (all 7 PNGs wired in as Figures 1–7). Every inline
`[Enric:]`/`[GPT suggestion]` comment in `paper.md` has been resolved or flagged
`<!-- NEEDS-PRINCIPAL -->`. The 9 evidence tables under `publish/tables/` + the
effort/process ledgers are landed and wired into the paper prose (every quantitative claim
in the text traces to a generated table or a committed proof object). What remains is
grouped into:

- **(i) DONE / filled** — text + table evidence complete; only a figure *render* or a
  release-hygiene paste is outstanding (no new science).
- **(ii) IN-PROGRESS (GPU campaign)** — needs a live GPU/CPU run on the final release commit;
  the paper carries explicit release-gate wording for each.
- **(iii) v0.2.0 / later** — referenced only as future work / planned proof targets; not a
  v0.1.0 blocker.

Conventions: each item gives the paper location, what is needed, the proof/table/figure that
fills it, and the owner type.

---

## (i) DONE / filled — evidence landed; only a figure-render or hygiene paste remains

These are no longer science gaps; the content is in the paper text and the generated tables.
The only outstanding work is rendering PNGs and one script-path update.

1. **§3.5 — AI process-metrics table + effort accounting — DONE.** `publish/tables/ai_process_ledger.md`
   and `publish/tables/effort_accounting.md` are generated and the §3.5 prose now carries the
   per-stage table, the ≈12.6-day calendar span (nothing→v0.0.1 ≈9.2 d→v0.1.0), 884 commits,
   249 sprint dirs, ≈500–700 agent-runs, order-10⁸ tokens (band 1–6×10⁸, no meter), the
   ≤€300/mo (Claude Max + GPT Pro) cost envelope plausibly reproducible for ≤€100, and the
   "no git tag → v0.1.0 is first" fact. Owner: DONE.

2. **§6.2 — d02 validation table — DONE.** `publish/tables/v010_d02_validation.md` generated
   (full-domain n=10494 + Tenerife-box n=955, 6/12/24/48/72 h, T2/U10/V10/PRECIP + persistence
   skill, all 3 cases). Inline case1 numbers reconciled to match the proof JSON. Owner: DONE
   (figures only, see #11).

3. **§6.3 — d03 status + before/after table — DONE.** `publish/tables/v010_d03_status.md`
   generated (v4→v5fix bias collapse, v5fix scores vs persistence, per-lead win/loss). The
   HFX/MYNN column-oracle fix is wired into §6.3 + §6.5(d) + the §3.6 error-catch ledger.
   Owner: DONE (the 24 h re-run is item #13, ii).

4. **§6.1 — idealized gate table — DONE.** `publish/tables/idealized_gate_summary.md` generated;
   §6.1 numbers (warm bubble θ′ 1.920 K / w 11.68 / rise 1924 m; Straka front 14150 m / θ′
   −9.971 / w 14.575 / mass 2.25e-9) match. Owner: DONE (figures only, see #11).

5. **§6.4 — performance + optimization-refutation tables — DONE.** `publish/tables/performance_current.md`
   (replaces stale `performance_evolution.md`) and `publish/tables/optimization_refutations.md`
   generated; the 5.29×/7.84×/3.2× headline, AI 0.40, roofline %s, ~11k ops/step, and the 5
   measured-and-refuted levers are wired in. Owner: DONE (roofline figure only, see #11).

6. **§6.5 — wind-skill (Coriolis) evidence — DONE.** `publish/tables/wind_persistence_skill.md`
   generated; §6.2/§6.5(c) carry the before/after (case3 V10 −0.13→+0.17, case2 winds ~50%),
   and the now-narrowed claim: U10/V10 have positive mean persistence skill in every case/region
   (not every lead; V10 has isolated case2 losses/ties). Owner: DONE.

7. **§3.6 / §9.3 — workflow visualization — DONE (text).** mermaid + ASCII loop reconciled to
   the §3.5 ledger role names; placeholder removed. Owner: DONE (clean PNG render optional, #11).

8. **§8 — claim boundary + roadmap — DONE.** `publish/tables/v010_claim_boundary.md` +
   `publish/GPU_PORT_GAPS_TODO.md` + `.agent/decisions/POST-0.1.0-ROADMAP.md` wired into §8.
   Seasonal TOST is now framed as future evidence / planned proof target, not a v0.1.0 claim
   and not a timing promise (see iii). Owner: DONE.

9. **§2.2 — comparator table re-grounding.** `publish/tables/comparators.md` exists; re-confirm
   each row's bibkey resolves in `references.bib` and the wording stays "context, not normalized
   benchmark." Current `paper.md` uses `\cite{...}` syntax; all 29 unique inline citation keys
   resolve in the 40-entry `references.bib`. Owner: citation-check.

10. **§9.4 — publication-audit script + manifest.** Update `scripts/m7_publication_audit.sh` to
    target `publish/paper/` (currently `publication/draft`) + current v0.1.0 proofs; paste output
    to `publish/manifest/publication_audit_v1.json`. Note: `scripts/verify_all.sh` +
    `publish/VERIFICATION.md` now exist as the binding 11-row contract — the audit script should
    cross-check against them. Owner: release-hygiene.

11. **Figure renders — DONE.** All 7 PNGs are rendered and wired into the paper as Figures 1–7
    (the `[MISSING FIGURE ...]` markers are removed): `model_role_timeline.png` (§3.2, Fig 1),
    `workflow_loop.png` (§3.6, Fig 2), `validation_pyramid.png` (§5, Fig 3 — re-rendered to show
    **d03 PASS (secondary)** and TOST underpowered), `warm_bubble_panel.png` (§6.1, Fig 4),
    `straka_density_current_panel.png` (§6.1, Fig 5), `roofline_dycore.png` (§6.4, Fig 6),
    `self_correction_timeline.png` (§6.5, Fig 7 — re-rendered to show d03 24 h validated).
    Re-render with `taskset -c 0-3 python3 publish/figures/render_paper_figures.py`. Owner: DONE.

---

## (ii) GPU campaign — EXECUTED. Only the final tag (PENDING-TAG) remains.

The GPU verification campaign is complete; the outcomes are recorded in
`proofs/PROOF_TABLE.md` (HFX-fix HEAD `d1c373b`; 9 PASS / 1 comparator-harness FAIL — not a
production defect / 1 device-residency INCONCLUSIVE — architecturally fine). The only remaining
step is release hygiene: cut and tag the final commit so the published numbers are tied to a
tagged hash (PENDING-TAG). `publish/VERIFICATION.md` has been updated to the executed outcomes.

12. **§6.2 — d02 re-validate on final commit — DONE (PENDING-TAG hash).** The 3-case d02
    validation was re-run post-HFX-fix: **D02_VALIDATED**, T2 RMSE unchanged vs pre-fix (no
    regression), winds beat persistence at every lead, finite/stable to 72 h (proof table row 4).
    The numbers are tied to the final tagged commit once cut. Owner: PENDING-TAG (manager).

13. **§6.3 / §8 — d03 24 h HFX-fix confirmation — DONE.** The 24 h d03 re-run now passes:
    **D03_1KM_VALIDATED** (proof table row 5), T2 RMSE 1.92 K ≤ 3.0 gate (beats persistence,
    skill +0.16), U10 3.45 / V10 4.24 ≤ 7.5 gate (V10 beats persistence). The earlier OOM-BLOCKED
    state is resolved. d03 enters the positive claim **as a secondary result** with field
    qualifiers (T2 beats persistence; V10 mostly; U10 short-leads only, loses at long leads but
    within gate) because the unblocking HFX repair is an empirical partial fix. Owner: DONE.

14. **§6.4 / §9.3 — counted D2H audit + repeatability + restart — DONE / INCONCLUSIVE.**
    Repeatability (`--repeat`) and restart-continuity (`--restart-at-hour 1`) both **PASS** now
    (proof table row 8). The *byte-counted* in-loop D2H audit is **INCONCLUSIVE** (proof table row
    11): the classifier finds in-loop events but cannot extract per-event byte sizes, so a
    `bytes_accounted` guard yields INCONCLUSIVE rather than a fabricated zero. Residency stays
    architecturally guaranteed by construction. The paper now states this precisely (no false
    zero-in-loop claim). The byte-counted audit is a tracked v0.2.0 follow-up. Owner: DONE
    (v0.2.0 for the byte-size extraction).

15. **§6.1 / §6.4 / §9 — PASS rows confirmed via the campaign.** Rows 1, 2 (idealized), 7
    (conservation), 9 (performance), 10 (precip) are PASS on the HFX-fix HEAD (proof table). Row 3
    (savepoint-parity *comparator harness*) is **FAIL — comparator-harness gap, not a
    production-dycore defect** (the validation-only core path is fed a state missing ~30
    `small_step_prep` leaves; the production dycore is validated by rows 1/2/7 + d02/d03). The
    paper and VERIFICATION.md report this honestly; the comparator fix is a v0.2.0 follow-up.
    *PENDING-TAG:* a final `scripts/verify_all.sh` (`VERIFY_RUN_GPU=1`) re-run on the tagged commit
    regenerates `proofs/PROOF_TABLE.md` against that exact hash. Owner: PENDING-TAG (manager).

---

## (iii) v0.2.0 / later — future work / planned proof targets; not a v0.1.0 blocker

16. **§5 / §8 — statistical equivalence (TOST) as future evidence.** The paper states plainly:
    the reuse-only corpus = **3 distinct usable MAM (spring) days** → any TOST on it is underpowered
    + single-season and is **never called "seasonal."** The harness, station-paired scorer, and
    predeclared ADR-029 margins are **built and self-tested** (CPU-vs-CPU reproduces the benchmarks
    to 0.00 delta — `proofs/m20/selftest_verify_release.json`, `tost_campaign_plan.md`). The
    May-only n≈15 backfill is technically feasible from preserved AIFS forcing (≈13 CPU-WRF runs
    ≈68 CPU-h + ≈5.5 GPU-h), but it would prove only spring/MAM equivalence. The paper no longer
    promises "within days"; it frames TOST as a v0.2.0 / near-term planned proof target. A truly
    multi-season TOST needs going-forward retained-output capture or cross-season backfill
    (calendar-bound). Owner: v0.2.0.

17. **§7 — differentiability / ML-hybrid / DA.** Structural JAX property only; no gradient/DA/ML
    result is claimed for v0.1.0. Owner: v0.2.0.

18. **§8 — P0/P1 roadmap items.** P0-1 nesting, P0-3 prognostic Noah-MP (the residual daytime
    HFX is partly its surface-energy-balance coupling), P0-4 d01 cumulus, P0-2 native init, S1
    multi-GPU, and P1 fidelity debts each close with their own proof object. v0.2.0 / second paper.
    Owner: v0.2.0.

19. **§6.2 — precipitation verification.** Reported as diagnostic/limitation only; FSS/SAL event
    corpus is future work before any precip skill claim. Owner: v0.2.0.

20. **§8 — Hopper/H200 re-measurement.** Portability argued; speedup-vs-CPU must be measured on
    that hardware (none available). Owner: v0.2.0 (hardware-gated).

---

## Release-hygiene + human decisions (gate the tag; not science)

21. **§9.1 — public repo URL, `v0.1.0` tag, exact release commit — PENDING-TAG.** Paper now uses
    `_PENDING-TAG_` markers (not `[MISSING RELEASE ITEM]`). Validation ran on the HFX-fix HEAD
    `d1c373b` (proofs on `worker/opus/final-verdict`); the tag is cut from the final commit after
    the docs refresh. v0.1.0 is the FIRST tag (no v0.0.1 git tag ever existed). The same hash must
    be back-filled into `CITATION.cff`, `zenodo_metadata.json`, and §9.1. Owner: release-hygiene + human.

22. **§9.2 — pinned environment manifest — PENDING-TAG.** Pin Python/JAX/jaxlib/CUDA/driver/
    XLA-flags/OS at the release commit (drafting env: Python 3.13.11 / JAX 0.10.0 / CUDA 13.1 /
    driver 595.71.05 / RTX 5090, per `publish/manifest/environment.json`). The package still
    declares only `python>=3.10`, `jax>=0.4`; the release replaces these floors with pins. Owner:
    release-hygiene.

23. **§9.4 — data/fixture availability statement — PENDING-TAG.** Paper now carries the
    `_PENDING-TAG_` statement (small references/proofs ship; full Gen2/CPU-WRF corpus + AEMET obs
    are too large/licensed → described for regeneration). Finalize at release. Owner: human + release-hygiene.

24. **§1 — `[GPT suggestion]` AEMET/HARMONIE-AROME adequacy — flagged NEEDS-PRINCIPAL.** No citation
    produced; wording kept at the softened "in the author's operational experience." A
    `<!-- NEEDS-PRINCIPAL -->` note in §1 records this. Provide bibkey `aemet_harmonie_arome` to
    strengthen, or leave as-is. Owner: citation-check + human.

25. **§1 / §2.1 — `[GPT suggestion]` prior open GPU-WRF / commercial completeness audit — flagged
    NEEDS-PRINCIPAL.** No prior-art audit table produced; "to the best of our current knowledge" is
    retained and no bare "first" appears in the title/abstract. A `<!-- NEEDS-PRINCIPAL -->` note in
    §1 records this for journal submission. Owner: citation-check + human.

26. **§9.5 — independent human numerical-methods review — OPEN (human).** Disclosed as an arXiv
    limitation (acceptable); REQUIRED before any journal submission. Owner: human.

27. **Stale-text purge (cross-cutting) — DONE in paper-owned scope; manager owns the rest.** The
    stale earlier-draft `publish/paper/paper.pdf` (2026-05-28) was REMOVED (rebuild via the
    `publish/paper/README.md` instructions). `publish/VERIFICATION.md` was refreshed to the executed
    proof-table outcomes. The new `paper.md` treats 22.26×/50.20×/156.82× ONLY as retracted
    self-correction history. STILL OWNED BY MANAGER (outside this doc pass's file ownership):
    the README 22.26× "Core goals" line, the M7-era `publish/tables/{performance_evolution,
    skill_evolution,m7_gates,sprint_ledger,test_coverage}.md` STALE banners, and any old
    `publication/draft/` paper. `publish/paper/honesty_audit.md` is already refreshed to v0.1.0.

28. **§7 — `[GPT suggestion]` author-reflection placement — RESOLVED + NEEDS-PRINCIPAL.** The
    reflection was converted from first person to third-person professional "human author" voice
    (per Enric's note) and explicitly marked non-load-bearing. A `<!-- NEEDS-PRINCIPAL -->` note
    flags the journal-vs-arXiv placement decision. Owner: human + editor.

---

## TOP remaining items for the manager BEFORE tagging (post-groundwork)

The GPU campaign is done; the science is settled in `proofs/PROOF_TABLE.md`. What remains is
release hygiene and a few human decisions:

1. **Cut + tag the final commit (PENDING-TAG)** and back-fill the hash into §9.1, `CITATION.cff`,
   and `zenodo_metadata.json`; optionally re-run `scripts/verify_all.sh VERIFY_RUN_GPU=1` on the
   tagged commit to regenerate `proofs/PROOF_TABLE.md` against that exact hash (#21, #15).
2. **Pin the environment manifest** (#22) and finalize the data-availability statement (#23).
3. **Build the PDF** on a machine with pandoc+LaTeX (instructions in `publish/paper/README.md`);
   figures (Fig 1–7) are rendered and wired.
4. **Human decisions:** AEMET citation (#24), prior-art "first" audit (#25), independent
   numerical-methods review before any journal submission (#26), author-reflection placement
   (#28), and the v0.2.0 "within hours/days" conclusion wording (RESOLVED to honesty-preserving
   text; principal may override — flagged in §10).
5. **Manager-owned stale-text purge** of files outside `publish/paper/` (#27): README 22.26× line,
   M7-era table banners.

Citation status: current `paper.md` uses inline `\cite{...}` syntax; the 16 inline `\cite{}` calls
cover all unique cited keys, and every key resolves in `references.bib` (40 entries). The only
possibly new bibkeys are optional evidence for #24 and #25; both claims degrade to a hedge if
uncitable.
