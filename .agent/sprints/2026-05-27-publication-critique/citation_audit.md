# Citation Audit

Scope: `publication/draft/references.bib` and citations in `publication/draft/paper.md`. This audit used local BibTeX metadata and the research brief; it did not add new citations.

## Citation Key Integrity

- Missing cited key: `TODO_Mollick_AI_authorship` is cited in the paper but absent from `references.bib`.
- Uncited bibliography entries: `fredj2023adios2wrf`, `paredes2023gt4py`.
- All other cited keys have matching BibTeX entries.

## Strong Or Mostly Usable Entries

- WRF/core/physics: `skamarock2019description`, `powers2017weather`, `thompson2008explicit`, `iacono2008radiative`, `niu2011noah`.
- NWP/GPU peers with DOI or stable IDs: `bertagna2024scream`, `dahm2023pace`, `govett2017parallelization`, `milroy2018ensemble`, `roberts2008scale`, `wernli2008sal`.
- ML weather: `lam2023graphcast` is strong; arXiv-only ML entries are usable as preprints if labeled that way.

## Entries Needing Metadata Verification

- `nakanishi2006numerical`: the research brief and current BibTeX disagree on title/DOI details. Verify exact MYNN reference before submission.
- `fuhrer2026icon` and `lapillonne2026benchmarking`: both have DOI-form metadata, but they are 2026 entries and should be verified against publisher pages before relying on page ranges/titles.
- `tempoquest2025acecast`: generic manual URL is weak support for a numeric `5-14x` speedup claim. Use exact TempoQuest/WeatherBell material or soften the claim.
- `whitaker2023gt4py`: verify authors/title against the actual GMD GT4Py paper. `paredes2023gt4py` is in the BibTeX but unused; choose the correct GT4Py citation set.
- `frostig2018tracing`: URL-only and old conference metadata. Acceptable for JAX/XLA tracing context, but not enough for runtime/version claims.
- `bennun2019dace`: has arXiv ID but no DOI; add final SC citation metadata if available.
- `lang2025update`: arXiv-only; keep as preprint and verify arXiv ID/title.
- `anthropic2026claude`: looks like an internal/marketing-style source, not a stable technical paper. Avoid using it as a hard scholarly support unless official and accessible.
- `arxiv2026policy`: risky. It is represented as arXiv administrative policy but points to a secondary news site. Replace with official arXiv policy if the claim remains.
- `pcmag2026arxiv`: usable only as news context, not authoritative policy.
- `nature2024editorial`: verify DOI/title/page metadata; use official publisher AI policy if the point is venue rules.
- `schmidt2025senior`: arXiv preprint; verify arXiv metadata.
- `nvidia2025geforce`: title says NVIDIA specs but URL is PNY. Either cite PNY accurately or use official NVIDIA specs.
- `aemet2026observations`: generic AEMET homepage. Need exact observation product/station-data source.

## Placement Issues

- The AI authorship paragraph combines arXiv policy, publisher policy, news, and senior-author analogy in one citation cluster. Split into: official venue policy, publisher policy, and conceptual authorship discussion.
- The Pace/GT4Py/DaCe paragraph cites `whitaker2023gt4py` but not the unused `paredes2023gt4py`; decide which paper supports which statement.
- `fredj2023adios2wrf` is unused. Either remove it or cite it when discussing I/O/restart/data-pipeline limitations.
- The RTX 5090 citation should not be used to support local measured memory use; local proof object supports that.

## Recommendation

Before public release, run a bibliography verification pass that resolves the missing Mollick key, removes uncited entries or cites them intentionally, replaces policy/news citations with official sources, and checks every DOI/arXiv ID against publisher metadata.
