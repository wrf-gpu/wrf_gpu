# Adversarial Probes for Worker Deliverables

Tests the validator runs against each AC, with the failure mode each probe is
designed to catch. Any future tester pass should walk this list manually after
`validate_deliverables.sh` returns 0.

## A. Citation honesty probes

1. **Fabricated key probe**: pick any 3 `\cite{...}` keys at random from each
   deliverable. Confirm each appears in `publication/draft/references.bib` *or*
   in `citations_to_add.md`. Any unknown key → fabrication or typo. The
   validator does this automatically and exits 2 on miss; this manual pass is
   the second line.
2. **Verifiable-URL probe**: for any BibTeX entry the worker proposes in
   `citations_to_add.md`, dereference the `url`/`doi` field. arXiv, GMD, BAMS,
   AMS Journals, and IEEE pages must resolve. If the worker proposes a
   `tempoquest.com/...` URL, treat it as marketing material and tag the
   citation as `[verify]` in the catalogue.
3. **Year-vs-author cross-check**: the worker should not cite Michalakes &
   Vachharajani 2008 for any post-2015 claim, nor cite Mielikainen 2012 for the
   "5-7× ceiling" claim without also citing a 2017+ retrospective.

## B. Claim-overreach probes

4. **AceCAST acknowledgement**: `novelty_bounds.md` must explicitly name
   AceCAST/TempoQuest and explain why the proposed claim survives that prior
   art. The validator's `CLAIM-OVERREACH` rule catches the most obvious cases;
   re-read the document for hedged-but-still-overreaching phrasings such as
   "first practical open-source port" or "first viable port".
5. **Three-options gate**: AC3 requires three claim sentences ranked by
   aggressiveness. Confirm the *most aggressive* option is one the validator
   would reject in any other context — that is the test of the ranking. If the
   most aggressive option is also defensible, the worker has under-stretched.
6. **Scope-creep test**: the catalogue should not be padded with non-WRF
   regional GPU models (ICON, COSMO, SCREAM, NIM, ERF, Pace). Those belong in
   the history narrative as context, not in the catalogue. The catalogue must
   be **WRF-only**.

## C. Inclusion-completeness probes

7. **Mandatory rows**: the catalogue must include, at minimum:
   - WRF-OpenACC (Govett et al., NOAA ESRL line of work)
   - WRF-CUDA microphysics (Mielikainen 2012-2015 series)
   - WRF-CUDA Fortran dynamics (Michalakes & Vachharajani 2008)
   - AceCAST (TempoQuest, commercial)
   - WRF physics-on-GPU efforts at IBM Almaden / Stony Brook (if any survive)
   - One row marked "no peer-reviewed citation found" if the worker can prove
     a vendor-only or abandoned attempt exists.
   Missing any of the first four → catalogue fails AC2 even if row count ≥ 8.
8. **License column populated**: every catalogue row must have an explicit
   license entry. "Closed-source commercial", "Apache-2.0", "BSD-3-Clause",
   "abandoned/unknown" are all acceptable; blank or "TBD" is not.
9. **Status column with date**: each "status" cell must include either
   `active YYYY` (with the year the worker verified activity) or `abandoned
   ~YYYY` (with the last known activity year).

## D. Why-it-is-hard probes

10. **Math section must name**: split-explicit RK3 + acoustic substeps,
    vertically implicit solves, terrain-following mass coordinate. If any one
    is missing the section fails AC4.
11. **Physics section must name**: scheme-specific physics interfaces (Thompson,
    MYNN, RRTMG, Noah-MP) and the host-resident control flow they assume.
12. **Coding section must name**: legacy F77 control flow, the WRF
    registry-generated state, the I_DM macro chain. These are the actual
    blockers, not abstract "Fortran is old".
13. **Organisational section** is the hardest to write without sounding
    dismissive. The probe: the section must *praise* at least one prior
    attempt (cite by name) for what it got right before naming what stopped it.

## E. Multi-agent framing probes

14. The framing must acknowledge that the user's *first attempt* failed —
    single-model GPT-5.4 alone and Opus 4.6 alone both failed. The framing must
    not retrofit this as inevitable; it should be presented as evidence that
    the dialog mechanic carries weight.
15. The framing must cite at least one peer-reviewed actor-critic paper for the
    methodology, not only LLM-vendor marketing material.

## F. Reproducibility probes

16. Every numerical claim in the history narrative ("5-7× ceiling", "AceCAST
    reports 6× on Ampere", etc.) must point to a specific figure/table/page in
    a cited source. `[verify]` tags are acceptable for values the worker could
    not pin to a page.

## G. Negative-result probes

17. If the worker reports that *no* open-source full-physics device-resident
    WRF port exists pre-this-work, the tester must construct a single
    counter-evidence search: a `gh search code --owner ncar` or equivalent
    GitHub query. The negative result is itself the load-bearing claim and
    must be backed by an explicit search log, not by absence of memory.
