# Paper groundwork rewrite — handoff (2026-06-01)

Agent: Opus 4.8 (xhigh) documentation pass. Branch `worker/opus/final-verdict`.
Commit: `333226a [paper] v0.1.0 groundwork rewrite + Zenodo-ready package`.
Scope honored: touched ONLY `publish/paper/**`, `publish/VERIFICATION.md`, and
`publish/figures/**`. Did NOT touch `scripts/**`, `src/**`, `proofs/PROOF_TABLE.md`,
`state.py`, `dynamics/**`, or the GPT-owned `m6b6_coupled_step_compare.py` /
`savepoint_parity.sh`. No GPU job run; pure doc + CPU matplotlib figure re-render.

## Objective

Groundwork first-pass rewrite of the v0.1.0 paper + a Zenodo-ready folder so the principal
can read a clean draft and decide publish-on-0.1.x vs hold-for-0.2.0. Honor every inline
comment, remove obsolete hedging, preserve every narrowed claim, follow Zenodo guidelines.

## Files changed

- `publish/paper/paper.md` — the rewrite (all inline comments resolved; figures wired; claims aligned to the final proof table).
- `publish/paper/CITATION.cff` — NEW. CFF 1.2.0, valid YAML.
- `publish/paper/zenodo_metadata.json` — NEW. Zenodo deposition `metadata` block, valid JSON.
- `publish/paper/README.md` — NEW. PDF build instructions + folder contents + authorship/license summary.
- `publish/paper/paper.pdf` — REMOVED (stale 2026-05-28 earlier draft; rebuild via README).
- `publish/paper/missing_elements.md` — reconciled against the rewrite + completed campaign.
- `publish/VERIFICATION.md` — refreshed Status column + release rule to the executed PROOF_TABLE outcomes (was stale: pre-campaign BLOCKED/NOT_RUN rows).
- `publish/figures/render_paper_figures.py` — updated 2 figure functions (d03 now PASS).
- `publish/figures/{validation_pyramid,self_correction_timeline}.png` — re-rendered (CPU-only).

## Commands run

- `git status/log/show`, `ls`, read-only proof/table/figure inspection.
- `python3 publish/figures/render_paper_figures.py` (2 functions only; matplotlib Agg, no GPU import).
- Citation-integrity check (29 unique `\cite{}` keys → all resolve in the 40-entry `references.bib`; 6 comparator keys also resolve).
- JSON/YAML validation of the two new metadata files (both parse clean).
- `git add` + `git commit` on `worker/opus/final-verdict`.

## Proof objects produced

None new (doc work). Numbers sourced exclusively from `proofs/PROOF_TABLE.md` (read-only).

## Every inline comment — resolved vs flagged

| Location | Comment | Disposition |
|---|---|---|
| §1 | GPT #24 — AEMET/HARMONIE-AROME adequacy citation | **NEEDS-PRINCIPAL.** No citation produced; kept softened "in the author's operational experience"; `<!-- NEEDS-PRINCIPAL -->` note in §1. |
| §1 | GPT #25 — prior-art "first" audit | **NEEDS-PRINCIPAL.** No audit table produced; kept "to the best of our current knowledge"; no bare "first" in title/abstract; note in §1. |
| §1 | Enric — add the plain-language "anyone with a GPU / massively faster / AGPLv3" framing | **RESOLVED.** Incorporated, kept inside the knowledge-hedge, no new unqualified "first". |
| §7 | Enric — self-improving 100%-AI-written memory/skill files note | **RESOLVED.** Added as a "Self-improving governance assets" paragraph in third-person voice. |
| §7 | Enric — move reflection from first person to professional "human author" voice; GPT #28 — placement for journal | **RESOLVED + NEEDS-PRINCIPAL.** Converted to third person, marked non-load-bearing; `<!-- NEEDS-PRINCIPAL -->` flags the arXiv-vs-journal placement. |
| §10 | Enric — v0.2.0 "likely within hours or days" | **RESOLVED + NEEDS-PRINCIPAL (recommend keep as written).** Added a roadmap pointer WITHOUT the timing promise (conflicts with GPT #16 + §8 honesty rule). Flagged for principal override. |
| §10 (general) | Enric — per-stage agent-runs/tokens table + cost message | **RESOLVED.** Already implemented in §3.5 (per-stage table, ≈12.6 d span, 884 commits, ≈500–700 agent-runs, order-10^8 token estimate, ≤€300/mo envelope). Cost takeaway also surfaced in §7 ("The cost message"). Git-history accounting was read-only as requested; no per-agent run IDs were logged, so the token figure is an explicit estimate, stated as such. |

All `[MISSING FIGURE]` / `[MISSING RELEASE ITEM]` markers were removed (figures wired; release
items converted to PENDING-TAG). Zero `[Enric:]`/`[GPT suggestion]`/`[MISSING]` markers remain.

## Obsolete hedging removed; narrowed claims preserved

Removed (now-validated): abstract/§5/§6.3/§8 "1 km nest pending / bounded-fail / 24 h
re-confirmation pending"; §6.4/§9.3 "counted D2H + repeatability + restart are still release
gates / NOT_RUN"; §6.2 "must be re-run on the final commit" (now done, no regression); §8
"this positive statement still depends on closing the final release-commit rows".

Preserved exactly (honesty-critical):
- HFX fix = **empirical partial MYNN-inspired land thermal-roughness repair, NOT a faithful
  `module_sf_mynn.F` port** (the three formula mismatches kept verbatim in §6.3).
- d03 **field-qualified**: T2 beats persistence (most leads, final +0.16); V10 mostly; U10
  short leads only / loses at long leads but within 7.5 gate; no field beats persistence at
  every lead. d03 kept **secondary** to d02.
- Speedup ~5.3–7.8x **faithful ceiling** (never ≥10x); ~8–11x stated as a fidelity-bounded ceiling.
- Device-transfer audit row 11 = **INCONCLUSIVE** (byte-counted classifier could not extract
  per-event sizes; residency architecturally guaranteed) — never a false zero-in-loop PASS.
- TOST row 6 = **underpowered single-season MAM descriptive paired-delta check**, U10 equivalent
  within margin / V10 borderline / T2 not — **never "equivalence PASS," never "seasonal."**
- Row 3 savepoint comparator = **comparator-harness gap, NOT a production-dycore defect**
  (added honest Tier-1 + abstract + §8 qualifiers; production dycore validated by rows 1/2/7 + d02/d03).

## Zenodo authorship arrangement implemented + policy basis

- **Creator / author of record (sole, accountable, in the citation):** Enric Guenther.
- **Contributors (NOT in the citation):** Anthropic (Claude Opus 4.7/4.8), OpenAI (GPT-5.5
  Codex), Google (Gemini 3.5), each as DataCite `contributorType: "Other"`.
- **Policy basis:** Zenodo `creators` appear in the citation and are the accountable authors;
  `contributors` use the DataCite contributor controlled vocabulary (ContactPerson, …,
  ResearchGroup, **Other**). Zenodo permits *organizations* as creators/contributors but has
  no field making an AI tool an author, and the broad authorship policy (arXiv 2026 / Nature
  2024, already cited in §9.5) is that AI cannot be an author because it cannot hold
  accountability. Therefore: human = sole creator; AI orgs = contributors with role `Other`;
  plus an AI-use disclosure (~99.9% AI implementation under author direction) in the byline,
  §9.5, `CITATION.cff` notes, and the `zenodo_metadata.json` description. CFF 1.2.0 has no
  first-class contributors-vs-authors split in the citation, so AI systems are recorded in CFF
  `references` (entity `name` = the org) rather than `authors`. Sources:
  help.zenodo.org/docs/deposit/describe-records/{creators,contributors}; DataCite contributor
  vocabulary; citation-file-format schema-guide (person vs entity objects).
- **License:** AGPL-3.0-or-later (honors Enric's note + `publish/LICENSE_RECOMMENDATION.md`).

## What remains PENDING-TAG (do NOT fabricate — proof table stable, final commit not yet tagged)

1. Public repo URL, `v0.1.0` tag, exact release commit hash (validation ran on HFX-fix HEAD
   `d1c373b`; §9.1, CITATION.cff, zenodo_metadata.json all carry PENDING-TAG placeholders).
2. Pinned environment manifest at the release commit (drafting env Python 3.13.11 / JAX 0.10 /
   CUDA 13.1 / driver 595.71.05 / RTX 5090 in `publish/manifest/environment.json`; §9.2).
3. Data-availability statement finalization (§9.4 carries the PENDING-TAG draft).
4. Zenodo DOI (minted on first publish; reserve concept DOI, back-fill into CITATION.cff + paper).
5. Optional: re-run `scripts/verify_all.sh VERIFY_RUN_GPU=1` on the tagged commit to regenerate
   PROOF_TABLE.md against that exact hash (manager/`scripts/**`-owned).

## Places I was unsure (for the principal/manager)

- **§10 v0.2.0 timing:** I declined Enric's "within hours/days" phrasing because it directly
  conflicts with the paper's own honesty rule (§8) and GPT #16. I recommend keeping the
  honesty-preserving wording, but this is a principal call — flagged NEEDS-PRINCIPAL in §10.
- **VERIFICATION.md release rule:** the original said "tagged only when rows 1–11 are all PASS."
  Rows 3 (FAIL-harness) and 11 (INCONCLUSIVE) cannot reach literal PASS. I rewrote the rule to
  PASS the 8 forecast-correctness rows + qualified row 6, and explicitly treat rows 3/11 as
  honest non-defects / v0.2.0 follow-ups (matching PROOF_TABLE.md). If the manager wants the
  literal "all-11-PASS" contract instead, that requires the v0.2.0 comparator/byte-size fixes
  before tagging — a scope decision, not a doc decision.
- **`scripts/m7_publication_audit.sh` repoint (§9.4, ME #10):** still targets `publication/draft`;
  it is a `scripts/**` edit outside my ownership. Flagged with a `<!-- NEEDS-PRINCIPAL -->` note
  in §9.4 for the manager.
- **Manager-owned stale-text purge (ME #27):** the README 22.26× "Core goals" line and the
  M7-era `publish/tables/{performance_evolution,skill_evolution,m7_gates,sprint_ledger,
  test_coverage}.md` STALE banners are outside `publish/paper/`; left for the manager. (The new
  `paper.md` already treats 22.26×/50.20×/156.82× only as retracted history.)
- **PDF not built:** pandoc + LaTeX are not installed; per instructions I did not install system
  packages and instead documented the build in `publish/paper/README.md` (LaTeX/natbib route +
  citeproc alternative; note the one mermaid block has an ASCII fallback + the rendered Fig 2).

## Next decision needed

Principal reads the clean draft and decides: publish on a 0.1.x base now (tag + Zenodo) vs hold
for v0.2.0. If publishing: cut/tag the final commit, fill the PENDING-TAG fields, build the PDF,
and (manager) repoint the audit script + purge the two stale-file groups outside `publish/paper/`.
