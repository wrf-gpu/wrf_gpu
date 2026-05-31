# Missing Elements — v0.1.0 Paper Cross-Reference List

Status: refreshed 2026-05-31 alongside the filled `publish/paper/paper.md` (branch
`worker/opus/final-verdict`). This is the manager's pre-tag cross-reference list. The 9
evidence tables under `publish/tables/` + the effort/process ledgers are now **landed and
wired into the paper prose** (every quantitative claim in the text traces to a generated
table or a committed proof object). What remains is grouped into:

- **(i) DONE / filled** — text + table evidence complete; only a figure *render* or a
  release-hygiene paste is outstanding (no new science).
- **(ii) IN-PROGRESS (GPU campaign)** — needs a live GPU/CPU run on the final release commit;
  the paper carries an explicit `[PLACEHOLDER]` + "in-progress" note for each.
- **(iii) v0.2.0 / later** — referenced only as future work / proof-promise; not a v0.1.0 blocker.

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
   Owner: DONE (the 24 h re-run is item #14, ii).

4. **§6.1 — idealized gate table — DONE.** `publish/tables/idealized_gate_summary.md` generated;
   §6.1 numbers (warm bubble θ′ 1.920 K / w 11.68 / rise 1924 m; Straka front 14150 m / θ′
   −9.971 / w 14.575 / mass 2.25e-9) match. Owner: DONE (figures only, see #11).

5. **§6.4 — performance + optimization-refutation tables — DONE.** `publish/tables/performance_current.md`
   (replaces stale `performance_evolution.md`) and `publish/tables/optimization_refutations.md`
   generated; the 5.29×/7.84×/3.2× headline, AI 0.40, roofline %s, ~11k ops/step, and the 5
   measured-and-refuted levers are wired in. Owner: DONE (roofline figure only, see #11).

6. **§6.5 — wind-skill (Coriolis) evidence — DONE.** `publish/tables/wind_persistence_skill.md`
   generated; §6.2/§6.5(c) carry the before/after (case3 V10 −0.13→+0.17, case2 winds ~50%),
   and "U10/V10 beat persistence in every case/region." Owner: DONE.

7. **§3.6 / §9.3 — workflow visualization — DONE (text).** mermaid + ASCII loop reconciled to
   the §3.5 ledger role names; placeholder removed. Owner: DONE (clean PNG render optional, #11).

8. **§8 — claim boundary + roadmap — DONE.** `publish/tables/v010_claim_boundary.md` +
   `publish/GPU_PORT_GAPS_TODO.md` + `.agent/decisions/POST-0.1.0-ROADMAP.md` wired into §8.
   The seasonal-TOST v0.2.0 proof-promise is added to §5 + §8 (see iii). Owner: DONE.

9. **§2.2 — comparator table re-grounding.** `publish/tables/comparators.md` exists; re-confirm
   each row's bibkey resolves in `references.bib` and the wording stays "context, not normalized
   benchmark." (All 29 [CITE] keys already resolve; 40 bib entries.) Owner: citation-check.

10. **§9.4 — publication-audit script + manifest.** Update `scripts/m7_publication_audit.sh` to
    target `publish/paper/` (currently `publication/draft`) + current v0.1.0 proofs; paste output
    to `publish/manifest/publication_audit_v1.json`. Note: `scripts/verify_all.sh` +
    `publish/VERIFICATION.md` now exist as the binding 11-row contract — the audit script should
    cross-check against them. Owner: release-hygiene.

11. **Figure renders (all (i)).** Render the PNGs the prose references (each has a clear
    `[PLACEHOLDER ... Figure pending; the table/text carries the content]`):
    `model_role_timeline.png` (§3.2), `validation_pyramid.png` (§5, updated status),
    `warm_bubble_panel.png` + `straka_density_current_panel.png` (§6.1; PPMs already in
    `publish/figures/idealized/`), `roofline_dycore.png` (§6.4), `workflow_loop.png` (§3.6),
    optional `self_correction_timeline.png` (§6.5). No new science; pure rendering.
    Owner: figure-worker.

---

## (ii) IN-PROGRESS (GPU campaign) — live run on the final release commit required

Each of these has a clear `[PLACEHOLDER]` + "in-progress" note in the paper. Per
`publish/VERIFICATION.md`, v0.1.0 tags only when the 11-row proof table is all-PASS on the
release commit, and the d02/d03 validations MUST be re-run on the final post-HFX-fix code.

12. **§6.2 — d02 re-validate on final commit.** The published d02 table is on `5319b8d`
    (Coriolis); VERIFICATION.md row 4 requires a re-run on the final post-HFX commit so no number
    rests on a pre-fix proof. Owner: validation-run (GPU; manager).

13. **§6.3 / §8 — d03 24 h HFX-fix confirmation.** The MYNN land-thermal-roughness fix is
    VALIDATED at the 1 h midday column oracle (`sfclay_hfx_oracle_parity.json`: HFX land 4.22×→2.30×,
    T2 land bias +3.6 K→+1.2 K). The 24 h d03 re-run with the fix is currently
    `D03_1KM_BLOCKED` (a GPU OOM under shared-GPU contention — NOT a forecast failure;
    `d03_summary_run24h_hfxfix.json`). Re-run on an idle GPU to either pass the bounded gate
    (VERIFICATION.md row 5, currently FAIL/BLOCKED) or keep 1 km out of the positive claim
    (the paper currently does the latter — confirm the manager's decision). Owner: validation-run.

14. **§6.4 / §9.3 — counted D2H audit + repeatability + restart.** `systems_invariants.md`:
    speedup (9.09×), all-finite-guards-off, and length-independent device residency are PASS;
    but the *counted* in-loop D2H proof (script exists, no committed count),
    `repeatability.json`, and `restart_in_pipeline.json` are NOT_RUN (switches not enabled).
    Re-run the d02 pipeline with `--repeat`/`--restart-at-hour` + emit a counted transfer audit,
    or drop those specific systems claims (VERIFICATION.md rows 8, 11). Owner: validation-run.

15. **§6.1 / §6.4 / §9 — re-confirm PASS rows on the release commit.** VERIFICATION.md rows
    1–3 (idealized + savepoint parity), 7 (conservation), 9 (performance) are PASS but must be
    re-confirmed on the final commit via `scripts/verify_all.sh` (`VERIFY_RUN_GPU=1`). Owner:
    validation-run (GPU; manager).

---

## (iii) v0.2.0 / later — future work / proof-promise; not a v0.1.0 blocker

16. **§5 / §8 — seasonal TOST equivalence (explicit v0.2.0 proof-promise).** The paper states
    plainly: the reuse-only corpus = **3 distinct usable MAM (spring) days** → any TOST on it is
    underpowered + single-season and is **never called "seasonal."** The harness, station-paired
    scorer, and predeclared ADR-029 margins are **built and self-tested** (CPU-vs-CPU reproduces
    the benchmarks to 0.00 delta — `proofs/m20/selftest_verify_release.json`,
    `tost_campaign_plan.md`). The paper makes the ONE allowed timing commitment: the full
    single-season ≥15-case TOST (≈13 CPU-WRF backfill runs ≈68 CPU-h + ≈5.5 GPU-h) ships at the
    v0.2.0 release within days of the backfill, executed by the same AI process — phrased as a
    proof-promise, not a generic cadence. A truly multi-season TOST needs going-forward nightly
    capture (calendar-bound). Owner: v0.2.0.

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

21. **§9.1 — public repo URL, `v0.1.0` tag, exact release commit.** Placeholders. Current
    validated HEAD `5319b8d` (Coriolis) + fixes through `234265a`; confirm the final commit after
    the (ii) re-runs land. v0.1.0 is the FIRST tag (no v0.0.1 git tag ever existed). Owner:
    release-hygiene + human.

22. **§9.2 — pinned environment manifest.** Pin Python/JAX/jaxlib/CUDA/driver/XLA-flags/OS at the
    release commit (package declares only `python>=3.10`, `jax>=0.4`; drafting env reports
    Python 3.13 / JAX 0.10 / CUDA 13 / RTX 5090). Owner: release-hygiene.

23. **§9.4 — data/fixture availability statement.** State which Gen2/CPU-WRF corpus + AEMET obs
    ship vs are described-for-regeneration. Owner: human + release-hygiene.

24. **§1 — [VERIFY] AEMET/HARMONIE-AROME adequacy.** Cite an AEMET/HARMONIE-AROME product-resolution
    reference (new bibkey `aemet_harmonie_arome`) OR soften to "in the author's operational
    experience." The paper already degrades gracefully (hedge present). Owner: citation-check + human.

25. **§1 / §2.1 — [VERIFY] prior abandoned open-source GPU-WRF + commercial completeness.** Cite the
    prior attempt (new bibkey) OR keep the "to the best of our knowledge" hedge (present). Owner:
    citation-check + human.

26. **§9.5 — independent human numerical-methods review.** Disclosed as an arXiv limitation
    (acceptable); REQUIRED before any journal submission. Owner: human.

27. **Stale-text purge (cross-cutting).** Confirm no surviving stale claims at assembly: the old
    22.26× in the README "Core goals" line; M7-era `publish/tables/{performance_evolution,
    skill_evolution,m7_gates,sprint_ledger,test_coverage}.md`; the old `publication/draft/` paper;
    and `publish/paper/honesty_audit.md` (M7-era — refresh so every quantitative claim in the new
    paper has a current proof path). The new `paper.md` treats 22.26×/50.20×/156.82× ONLY as
    retracted self-correction history. Owner: manager + honesty-audit-refresh-worker.

---

## TOP remaining items for the manager to cross-reference BEFORE tagging

In priority order — these are the gates between the now-filled draft and an honest tag:

1. **Re-run d02 + d03 on the FINAL post-HFX commit** (ii #12, #13) and tie every published number
   to the release commit. The d03 24 h HFX confirmation is the single open scientific question
   (currently OOM-BLOCKED, not failed); decide pass-the-bounded-gate vs keep-1km-out-of-claim.
2. **Run the counted D2H audit + repeatability + restart** (ii #14) or drop those systems claims;
   they are currently NOT_RUN.
3. **Execute `scripts/verify_all.sh` with `VERIFY_RUN_GPU=1`** on the release commit (ii #15) so the
   11-row `publish/VERIFICATION.md` table is all-PASS — the binding tag condition.
4. **Render the 5–6 figures** (i #11) — no new science, but the paper references them.
5. **Pin the environment manifest + set repo URL / tag / commit** (hygiene #21, #22).
6. **Resolve or hedge the two [VERIFY] claims** (#24, #25) and complete the stale-text purge (#27).
7. **Confirm the seasonal-TOST proof-promise wording** (iii #16) is acceptable as the ONE timing
   commitment in the paper (machinery built + self-tested; only corpus backfill outstanding).

Citation status: all 29 [CITE] keys resolve in `references.bib` (40 entries). The only possibly
new bibkeys are the two optional [VERIFY] ones (#24, #25); both claims are written to degrade to a
hedge if uncitable.
