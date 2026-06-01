# Paper source — v0.1.0 release package

This folder is the self-contained source for the v0.1.0 preprint and the Zenodo
deposit. It is intended to be uploaded as-is (paper source + figures + bibliography +
citation/metadata) to Zenodo and/or arXiv.

## Contents

| File | Purpose |
|---|---|
| `paper.md` | The manuscript source (Markdown with LaTeX `\cite{}` citations and one mermaid diagram that has an ASCII fallback below it). |
| `references.bib` | BibTeX bibliography (40 entries). All inline `\cite{}` keys resolve here. |
| `CITATION.cff` | Machine-readable citation metadata (CFF 1.2.0). Human author = sole creator; AI systems recorded as non-author contributors. |
| `zenodo_metadata.json` | Zenodo deposition `metadata` block (creators, contributors with DataCite roles, keywords, license, version, related identifiers). |
| `missing_elements.md` | Cross-reference / open-items tracker (reconciled against this draft). |
| `honesty_audit.md` | Maps every load-bearing quantitative claim to a current proof object/table. |
| `human_author_notes.md` | The human author's source reflections (raw material for §1/§7/§10). |
| `../figures/*.png` | The 7 rendered figures referenced by the paper (see below). |
| `../tables/*.md` | The 9 evidence tables the paper cites. |

## Figures (all referenced in `paper.md`)

| Figure | File | Section |
|---|---|---|
| 1 | `../figures/model_role_timeline.png` | §3.2 |
| 2 | `../figures/workflow_loop.png` | §3.6 |
| 3 | `../figures/validation_pyramid.png` | §5 |
| 4 | `../figures/warm_bubble_panel.png` | §6.1 |
| 5 | `../figures/straka_density_current_panel.png` | §6.1 |
| 6 | `../figures/roofline_dycore.png` | §6.4 |
| 7 | `../figures/self_correction_timeline.png` | §6.5 |

Figures are regenerated (CPU-only, no GPU) with:

```bash
taskset -c 0-3 python3 ../figures/render_paper_figures.py
```

## Building the PDF

`pandoc` and a LaTeX toolchain are **not installed** in the drafting environment, so the
PDF is built on a machine that has them. Do not install system packages here.

Recommended build (LaTeX route, preserves `\cite{}` via natbib + BibTeX):

```bash
# 1) Markdown -> LaTeX, keeping \cite{} for a BibTeX pass
pandoc paper.md -s --natbib -o paper.tex

# 2) LaTeX -> PDF with the bibliography
pdflatex paper.tex
bibtex   paper
pdflatex paper.tex
pdflatex paper.tex
```

Alternative single-shot route (pandoc resolves the bibliography itself; requires that
the `\cite{a,b}` calls be acceptable to pandoc-citeproc, otherwise convert them to
`[@a; @b]` first):

```bash
pandoc paper.md \
  --citeproc \
  --bibliography=references.bib \
  --resource-path=.:../figures \
  -o paper.pdf
```

Notes for the PDF build:
- The mermaid block in §3.6 will render as a fenced code block under plain pandoc; an
  equivalent ASCII diagram is included immediately below it, and Figure 2
  (`workflow_loop.png`) is the rendered version. A `mermaid-filter` is optional.
- Figure paths in `paper.md` are repo-relative (`publish/figures/...`). When building
  from inside `publish/paper/`, pass `--resource-path=.:..:../figures` or build from the
  repository root so the paths resolve.
- The stale `paper.pdf` from 2026-05-28 (an earlier draft) is NOT this manuscript; delete
  or regenerate it before upload.

## Authorship and license (summary; see §9.5 of the paper)

- **Author / creator of record:** Enric Guenther (sole accountable party).
- **AI contributors (non-author):** Anthropic (Claude Opus 4.7/4.8), OpenAI (GPT-5.5 Codex),
  Google (Gemini 3.5). AI agents performed ~99.9% of the implementation under the author's
  direction. AI cannot hold accountability and is therefore not an author/creator (consistent
  with arXiv/Nature AI-authorship policy and the Zenodo/DataCite creator vs contributor split).
- **License:** AGPL-3.0-or-later (see `../LICENSE_RECOMMENDATION.md` for the rationale).
