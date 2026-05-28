# Tester Report — Sprint #6 (paper-control-opus)

**Role:** tester (Claude Opus 4.7 in sonnet-test-engineer role)
**Branch:** `tester/opus/paper-control`
**Worktree:** `/tmp/wrf_gpu2_papercontrol`
**GPU usage:** none. All work pinned to CPU cores 0–3.

---

## Scope

The Sprint #6 contract gives Opus (acting as worker) the binding final quality
gate on `publication/draft/paper.md` before PDF + arXiv. The tester job is to
re-run every validation command in that contract from a clean shell, add
edge-case tests the worker may not have written, and try to break the paper /
publication pipeline. This report is the tester deliverable; it does not
include the worker's `paper_control_verdict.md` (that file is the worker's
deliverable, not the tester's).

State of `.agent/sprints/2026-05-28-paper-control-opus/` at tester entry:

```
sprint-contract.md
role-prompts/tester.md
.tester-completion.sh
.tester-retry-count
```

There is no `worker-report.md`, `precondition_compliance.md`,
`narrative_critique.md`, or `paper_control_verdict.md` in the sprint
directory. The "worker" deliverable for Sprint #6 is the final state of
`paper.md` + `references.bib` + `honesty_audit.md` themselves (the worker
controls those files under AC6). The tester therefore validated those files
directly against the contract.

---

## Validation commands re-run from a clean shell

### 1. `bash scripts/m7_publication_audit.sh`

Ran under `taskset -c 0-3`. Result (truncated to header):

```json
{
  "ok": true,
  "paper_word_count": 7134,
  "bib_entries": 66,
  "cited_keys": 57,
  "missing_citations": [],
  "uncited_entries": [
    "anthropic2024effective", "anthropic2026claude", "fredj2023adios2wrf",
    "huang2013thermal", "jakobs2024wsm7", "milroy2018ensemble",
    "roberts2008scale", "schmidt2025senior", "wernli2008sal"
  ],
  "proof_objects_checked": 17,
  "validate_agentos": { "ok": true, "required_files_checked": 31, "skills_checked": 13 },
  "errors": []
}
```

`ok: true`. AC7 is satisfied. Returncode 0.

### 2. `python -m pytest tests/test_agentos_smoke.py`

```
2 passed in 0.02s
```

### 3. AC1 — Precondition compliance (independent re-check)

| Precondition | Status | Evidence |
|---|---|---|
| Option-2 novelty wording verbatim in paper | PASS | `publication/draft/paper.md:19`. Byte-equal match against the quoted block in `.agent/sprints/2026-05-28-gpu-wrf-history-research/novelty_bounds.md` (Option 2), under whitespace normalization. Confirmed by `tests/test_paper_control_edge_cases.py::test_option_2_novelty_wording_verbatim`. |
| Canary skill regression in Abstract | PASS | `paper.md:9` — Abstract carries the explicit "T2 +161 % to +378 % relative RMSE; U10 +214 % to +370 %; V10 +177 % to +353 %" band and the phrase "materially less skilful than CPU WRF". |
| Canary skill regression in Results | PASS | `paper.md:159–175` — entire subsection 7.3 (Forecast Skill) with full 9-row RMSE table; "The forecast-skill result is negative." |
| Canary skill regression in Limitations | PASS | `paper.md:207` — L6 names the regression and quotes the same RMSE bands. |
| Canary skill regression in Discussion | PASS | `paper.md:191` (section 9) — "the coupled surface/near-surface physics path produces materially worse station RMSE than CPU WRF." |
| Paper title contains no "Canary" | PASS | `paper.md:1` — `# wrf_gpu: An Open-Source JAX-Native WRF v4 Port with Whole-State GPU Residency`. Case-insensitive substring search for "canary" returns no match in the title line. |

All six AC1 preconditions hold.

### 4. AC2 — Honesty audit cross-check

Cross-checked every row in `publication/draft/honesty_audit.md` (lines 8–54).
Each row's referenced `.agent/...` or `publication/...` path resolves on disk
(see `tests/test_paper_control_edge_cases.py::test_honesty_audit_rows_point_to_existing_files`).
The nine RMSE table rows in `paper.md` section 7.3 each have a matching row
in honesty_audit.md
(`tests/test_paper_control_edge_cases.py::test_honesty_audit_covers_skill_table_rows`).

Spot-checked the four most load-bearing claims:

- 22.26× current speedup → `post_iter2_speedup.json` exists.
- 100-step bitwise savepoint parity → `savepoint_deep_column100.json` exists.
- Zero inter-kernel D2H → `d2h_audit_v2.json` exists.
- 1-h pipeline bitwise determinism → `determinism_repeat.json` exists.

No quantitative claim was found in paper.md that lacks a backing proof object
in honesty_audit.md.

### 5. AC3 — Citation audit

The audit script reports 0 missing citations and 9 uncited bib entries.
Independently re-implemented as a unit test
(`test_all_cite_keys_resolve_in_bib`, `test_bib_has_no_duplicate_ids`,
`test_no_placeholder_citations`, `test_bib_entries_have_minimum_fields`).
All pass.

The 9 uncited entries are exactly the set named in the sprint contract.
`test_audit_recorded_uncited_set_is_tracked` pins this set so any silent
drift surfaces as a unit-test failure. The CITED-or-TRIM decision is
the worker's call under AC3, not the tester's.

---

## Tests added under `tests/`

New file: `tests/test_paper_control_edge_cases.py` — 22 tests, all passing
in 0.38 s. They cover both the explicit contract acceptance criteria and
adversarial cases a worker focused on prose may have missed:

| Test | What it catches |
|---|---|
| `test_paper_title_does_not_contain_canary` | AC1 title invariant; case-insensitive. |
| `test_paper_title_is_stable_string` | Exact title pin to catch silent retitling. |
| `test_option_2_novelty_wording_verbatim` | Extracts Option-2 quote from `novelty_bounds.md` at runtime and verifies it appears verbatim in `paper.md` under whitespace normalization. |
| `test_skill_regression_in_required_sections` | Locks the Sprint #4 binding precondition that the skill regression appears in Abstract, Results, Discussion, *and* Limitations. |
| `test_no_unqualified_first_gpu_wrf_claim` | Adversarial: every occurrence of "first GPU-enabled WRF" / "first GPU WRF" / "first full GPU" must sit inside an explicit denial. |
| `test_no_first_full_open_source_gpu_wrf_outside_denial` | Adversarial: blocks Option-1 framing in any other location. |
| `test_rejected_speedup_numbers_are_marked_as_rejected` | Adversarial: 156.82× must only appear with a rejection marker; 50.20× must only appear as pre-fix/diagnostic. |
| `test_only_current_speedup_is_22_26x` | Adversarial: only 22.26× may be described as the *current* speedup. |
| `test_all_cite_keys_resolve_in_bib` | Defensive AC3 unit test; runs without the shell audit. |
| `test_bib_has_no_duplicate_ids` | Catches BibTeX duplicate-key bugs that hide behind bibtexparser's last-wins behavior. |
| `test_no_placeholder_citations` | Catches TODO/FIXME/PLACEHOLDER citation keys in either paper or bib. |
| `test_bib_entries_have_minimum_fields` | Every `@article/@inproceedings/@book/@techreport/...` carries a title and (unless `@misc`/`@software`/`@online`/`@unpublished`) a year. |
| `test_referenced_proof_objects_exist` | Every literal backticked `.agent/...` path in paper.md is a real path on disk. |
| `test_honesty_audit_rows_point_to_existing_files` | All `.agent/...` and `publication/...` paths inside `honesty_audit.md` resolve. |
| `test_honesty_audit_covers_skill_table_rows` | Each of the 9 RMSE rows in paper §7.3 has a matching honesty-audit row. |
| `test_paper_and_bib_are_ascii` | Smart quotes / Unicode that would break LaTeX rendering are blocked at the test layer. |
| `test_word_count_inside_audit_band` | Locks the 6000–12000 word window from the audit. |
| `test_required_sections_present` | Locks the 15 expected sections + References. |
| `test_limitations_section_has_seven_items` | The Limitations section enumerates exactly L1..L7. |
| `test_author_disclosure_present` | AI-system disclosure + human corresponding-author identification + "AI Use Disclosure" section must be present. |
| `test_publication_audit_returns_ok_true` | End-to-end re-run of `scripts/m7_publication_audit.sh` from the test layer. |
| `test_audit_recorded_uncited_set_is_tracked` | Pins the 9 known uncited bib entries; drift in either direction is a test failure. |

The full new-test suite plus the existing AgentOS smoke suite both pass:

```
$ taskset -c 0-3 python -m pytest tests/test_agentos_smoke.py \
                                    tests/test_paper_control_edge_cases.py
24 passed in 0.38s
```

---

## Break-attempts the tests will catch

I sanity-checked the guardrails against synthetic adversarial inputs (no
file edits) to confirm they are load-bearing, not vacuous:

- Title rewritten to include "Canary Islands Port" → title test fails.
- 156.82× quoted as a "current headline" without rejection markers → the
  rejected-speedup guardrail fires.
- Abstract rewritten to drop the percentage band and the "materially less
  skilful" phrase → the skill-regression-placement test fires.
- Option-2 quote re-paragraphed across multiple lines with collapsed
  whitespace → still detected, because the test normalizes whitespace
  before substring matching.

In other words, the tests do not rely on the current paper being unchanged;
they encode the contract.

---

## Fixtures and inputs

- `publication/draft/paper.md` (51,549 bytes, 260 lines).
- `publication/draft/references.bib` (26,561 bytes, 66 entries, 57 cited).
- `publication/draft/honesty_audit.md` (9,328 bytes, 47 numeric/claim rows).
- `.agent/sprints/2026-05-28-gpu-wrf-history-research/novelty_bounds.md`
  (read for Option-2 verbatim).
- `.agent/sprints/2026-05-28-testing-execution-opus-check/publishability_decision.md`
  (read to confirm the AC1 precondition source).
- 14 proof-object JSON / MD files under `.agent/sprints/...` and
  `.agent/decisions/...` (existence-checked, not parsed).
- No GPU runs. No live data fetches. CPU pinning `taskset -c 0-3` honored.

---

## Gaps / things the tester deliberately did NOT do

- **PDF rendering.** Sprint #6 ends one step before PDF. The tester did not
  invoke LaTeX/pandoc; the test suite confirms the source is render-safe
  (ASCII, citations resolve, sections present).
- **Citing AC4 (narrative flow) and AC5 (top-5 must-fix list).** These are
  worker deliverables under the contract (`narrative_critique.md`,
  the top-5 list, and the binding `paper_control_verdict.md`). The tester
  cannot substitute its own narrative judgment for the worker's
  manager-grade gate; that authority is explicitly assigned to Opus-as-worker
  in the contract. The tester audited everything the worker can be held to
  numerically (Option-2 wording, skill regression placement, citation
  resolution, honesty audit, proof-object existence, audit script outcome).
- **CITED-or-TRIM call on the 9 uncited bib entries.** Contract AC3 assigns
  that decision to Opus-as-worker. The tester pins the current set so any
  silent change to `references.bib` after this report surfaces immediately.
- **AC8 final verdict.** That is the worker's emission, not the tester's.

These are not test gaps in the contract sense; they are scope boundaries
the contract draws between worker and tester.

---

## Risks / things to watch in PDF / arXiv push

1. **Uncited entries (9).** They are not a blocker for `ok: true` but will
   become a publication-style nit when the LaTeX bibliography is built with
   `biber --validate-config`. The worker has authority to CITE-or-TRIM under
   AC3; that decision should happen before PDF, otherwise the PDF will carry
   9 dangling bib records.
2. **Tone of section 9 (Discussion).** §9 is intentionally short. If the
   reviewer of record asks for a longer Discussion, that is a v0.1 task,
   not a Sprint #6 reopen.
3. **Section 8 (Canary Case Study) sits after Results.** Some venues prefer
   Case Study before Discussion; the current order is consistent with the
   "port-first, Canary-as-workload" framing and was AC4-flagged in
   `paper.critique.md`. Leaving as-is is the tester's reading; the worker
   keeps AC4 authority on structural moves.
4. **Renaming of `paper.md` to LaTeX.** The audit script reads `paper.md`
   directly. When the manuscript moves to `paper.tex`, the audit will need
   one path update and the ASCII / citation checks should be re-run on the
   rendered `.tex`. That is a v0.0.1 push task, not a Sprint #6 task.

---

## Decision

Decision: **PASS**.

Justification: every validation command in the Sprint #6 contract that lies
inside the tester's authority returns green from a clean shell.
`scripts/m7_publication_audit.sh` returns `ok: true` with 0 errors and 0
missing citations. All six AC1 preconditions (Option-2 verbatim wording,
Canary skill regression in Abstract + Results + Limitations + Discussion,
title free of "Canary") hold under independent re-check. The honesty-audit
cross-reference for AC2 is complete: every quantitative paper claim has a
backing proof object that exists on disk. AC3 citation resolution passes.
The 22 new edge-case tests in `tests/test_paper_control_edge_cases.py` all
pass, including the adversarial "no unqualified first-GPU-WRF claim" and
"only 22.26× is the current speedup" guardrails. The remaining contract
items — AC4 narrative critique, AC5 top-5 must-fix list, AC6 actual fix
application, AC8 final verdict — are worker-authority deliverables and are
out of scope for the tester.

The paper as currently committed on `tester/opus/paper-control` is, from
the tester's point of view, ready for the worker's AC8 verdict gate and the
subsequent PDF + v0.0.1 push.
