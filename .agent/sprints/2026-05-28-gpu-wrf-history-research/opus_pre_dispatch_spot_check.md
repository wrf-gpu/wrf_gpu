# Tester Report — Sprint 2026-05-28-gpu-wrf-history-research

**Role**: tester (Claude Opus 4.7, acting as sonnet-test-engineer)
**Sprint**: `2026-05-28-gpu-wrf-history-research`
**Branch**: `tester/opus/gpu-wrf-history-research`
**Worktree**: `/tmp/wrf_gpu2_history`
**Attempt**: 1 of 5
**Generated**: 2026-05-28

## 1. State of worker deliverables

The sprint contract defines worker outputs AC1–AC6 (history narrative,
catalogue, novelty bounds, why-it-is-hard, multi-agent framing memo, BibTeX
stubs). At the time this tester pass ran, **none of those files exist on disk**
in `.agent/sprints/2026-05-28-gpu-wrf-history-research/` and **no
`worker-report.md` has been produced**. The directory contains only the
contract, role prompt, completion script, and retry-count file. There is no
worker branch matching `*gpu-wrf-history-research*` and the latest commit on
all branches is `929b27e` (the dispatch commit itself).

Conclusion: the worker has not yet been dispatched, or was dispatched in
parallel with this tester and has not landed output. Either way the contract's
AC7 verdict `RESEARCH_READY` cannot honestly be issued from this pass.

## 2. What this pass produced instead

Per the role prompt's mandate to "try to break the implementation" and add
tests under `tests/`, I built the validation surface a future tester pass will
need the moment worker output lands. Both artefacts live inside the sprint
directory so they are within both the role-prompt scope (`tests/`) and the
contract scope (`.agent/sprints/<sprint>/**`).

- `tests/validate_deliverables.sh` — mechanical AC1–AC7 gate. Checks each
  deliverable for presence, minimum byte size, mandatory content tokens,
  catalogue row count (≥8 per AC2), fabricated-citation keys (any
  `\cite{...}` key that resolves neither in `publication/draft/references.bib`
  nor in the worker's proposed `citations_to_add.md`), and one
  claim-overreach pattern ("first GPU port of WRF" without acknowledging
  AceCAST). Runs CPU-pinned on cores 0–3 per project rule. Exit codes 0/1/2/3
  signal pass / missing-file / content-failure / no-worker-report.
- `tests/adversarial_probes.md` — manual checklist of 17 probes the validator
  cannot fully automate: citation honesty (fabricated keys, dead URLs), claim
  overreach (AceCAST acknowledgement, the three-options gate, scope creep into
  non-WRF systems), inclusion completeness (mandatory rows: WRF-OpenACC,
  WRF-CUDA microphysics Mielikainen series, Michalakes & Vachharajani 2008,
  AceCAST), license-column population, why-it-is-hard section coverage,
  multi-agent framing requirements, and negative-result reproducibility
  (any "no full open-source GPU port" claim must be backed by an explicit
  GitHub/search log, not by absence of memory).

Ran `validate_deliverables.sh` against the current empty state to confirm the
harness fails closed: exits 3 ("worker has not delivered") as expected.

## 3. Independent spot-check of the central claim

The sprint contract quotes the principal author: *"there is no full GPU port
even commercially available let alone open source and not for the lack of
trying."* This is the load-bearing motivating sentence for the paper rewrite.

I spot-checked it against the two inputs the worker will be drawing from:
`publication/research_brief/english_brief.txt` (1638 lines) and
`publication/draft/paper.md` §2.2 (the current published framing).

Findings the worker must address:

- The user's verbal claim "no full GPU port even commercially available" is
  **strictly stronger** than the existing briefs support. TempoQuest's AceCAST
  is a commercial WRF-acceleration product (english_brief.txt:249, 1283–1288;
  `tempoquest2025acecast` in references.bib). The word "commercially" in the
  user claim is therefore not defensible as written; the defensible version is
  something like "no commercially-available *full clean-slate* GPU port" or
  "no commercial port that removes host-resident physics".
- The english brief at lines 1340–1344 already states the bounded version the
  worker should be converging on: *"first clean-slate rewrite of the WRF v4
  dynamics and operational physics suite in Python + JAX/XLA, achieving
  complete, zero-in-loop-transfer device residency on a single, consumer-grade
  GPU workstation."* This is defensible against AceCAST (directive-based,
  host-physics), WRF-OpenACC (partial scope), Pace (different model, FV3 not
  WRF), SCREAM (different model, exascale not workstation), and NIM
  (icosahedral, abandoned).
- `paper.md` §2.2 already enforces this bound — it explicitly disclaims "first
  GPU regional NWP system" and pivots to "workstation-scale JAX/XLA execution
  with the high-frequency d02 state resident on one consumer GPU"
  (paper.md:42). The worker's AC3 deliverable should reconcile the contract's
  aggressive verbal framing with this conservative on-page framing, and the
  paper-rewrite sprint should adopt whichever of the worker's three claim
  options the principal author signs off on.

Gaps in the existing inputs that the worker must close, not duplicate:

- The english brief contains literal blanks where speedup numbers should be
  (e.g. line 246: *"approximately to speedup"* — two missing values). AC2
  cannot inherit these blanks; the worker must source them.
- AceCAST license is not tagged anywhere in the briefs as "closed-source
  commercial" — that fact is necessary for the catalogue.
- The Michalakes & Vachharajani 2008 WRF-CUDA dynamics paper and the
  Mielikainen 2012–2015 microphysics series are the canonical "5-7× ceiling"
  citations and are **not** in the english brief at all. The worker must
  introduce them.
- WRF-OpenACC (Govett et al., NOAA ESRL) is present as a generic OpenACC row
  in the brief's table (line 1277) but without organisation, scope, or
  abandonment year. The worker must back-fill those columns.

## 4. Tests added or run

- Added `tests/validate_deliverables.sh` (executable, mechanical AC1–AC7
  gate).
- Added `tests/adversarial_probes.md` (17-item manual probe list keyed to ACs).
- Ran `validate_deliverables.sh` once against the current empty state. Exit
  code 3, output as expected: every AC1–AC6 file reported MISS, worker-report
  reported MISS, harness terminated before the citation-honesty and
  catalogue-row checks (correctly — those depend on files that do not exist).

Fixtures used: the harness reads `publication/draft/references.bib` directly
to build the known-citation-keys set. No binary fixtures were committed; the
references.bib file is already in git.

## 5. Gaps and risks

- **Cannot validate AC1–AC6 in this pass** — no worker output exists. The
  retry-count file shows attempt 1 of 5; if the dispatch system intended worker
  and tester to run in parallel, the next attempt should be deferred until the
  worker branch lands.
- **Role-prompt vs. contract mismatch**: the role prompt is the generic
  template ("re-run validation commands", "edit only `tests/`") and assumes a
  code sprint. This is a research sprint with no validation commands and no
  `src/`. I resolved the conflict by putting the validation harness *inside*
  `.agent/sprints/<sprint>/tests/` so both scopes are honoured, but a future
  dispatcher pass should consider templating the role prompt by sprint type.
- **Honesty-gate calibration**: the validator's `CLAIM-OVERREACH` rule fires
  on one specific phrase pattern. Sophisticated overreach phrasings ("first
  practical open-source port", "first viable workstation port") will pass the
  automated check and require the manual probe-list pass. The validator is a
  floor, not a ceiling.

## 6. Five-line executive summary (per AC7, for the paper-rewrite sprint)

Worker deliverables for sprint `2026-05-28-gpu-wrf-history-research` were not
produced; the contract's `RESEARCH_READY` verdict therefore cannot be issued
from this pass. Independent spot-check of the inputs confirms the principal
author's verbal claim ("no full GPU port even commercially available") is
strictly stronger than the english brief supports — AceCAST is a commercial
product — so the defensible ceiling for the paper introduction is the brief's
own line 1340 wording: *first clean-slate rewrite of WRF v4 dynamics and
operational physics in Python + JAX/XLA with zero-in-loop-transfer device
residency on a consumer-grade workstation GPU*. A mechanical validation
harness (`tests/validate_deliverables.sh`) and a 17-item adversarial probe
checklist (`tests/adversarial_probes.md`) are now in the sprint directory
ready to gate the worker output on the next pass.

## 7. Decision

Decision: **BLOCKED — NO_WORKER_DELIVERABLES.** Verdict
`RESEARCH_READY` is not achievable until the worker produces AC1–AC6. The
validation harness and adversarial-probe checklist are committed inside the
sprint `tests/` directory and ready to gate the next pass. Recommended manager
action: dispatch the research worker, and on the worker's exit re-run
`bash .agent/sprints/2026-05-28-gpu-wrf-history-research/tests/validate_deliverables.sh`
before issuing this tester role again.
