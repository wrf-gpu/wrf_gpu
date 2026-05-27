# Publish Folder - M8 Pre-Release Staging

This folder collects the arXiv-ready package built from the project. The user will create the public git repo and import this content when ready.

## Layout

```
publish/
|-- paper/        <- paper.md, references.bib, honesty_audit.md (copied from publication/draft/)
|-- tables/       <- consolidated benchmark + comparison tables for the paper
|-- figures/      <- any figure source / spec markdown (no large binaries here)
`-- manifest/     <- release manifest (commit hashes, env, proof object paths)
```

## What goes where

- `paper/paper.md` - the LaTeX-convertible markdown source
- `paper/references.bib` - bibliography (40 entries, all cited as of M8 staging)
- `paper/honesty_audit.md` - quantitative-claim audit
- `tables/performance_summary.md` - pre-fix / iter-1 / iter-2 GPU vs CPU
- `tables/skill_regression_summary.md` - pre-fix -> iter-1 -> iter-2 AEMET scoring
- `tables/comparators.md` - speedup vs Pace, ICON-exclaim, SCREAM, NIM, AceCAST (consolidation sprint output)
- `tables/m7_gates.md` - per-gate completion + proof-object pointer
- `figures/*.md` - figure specs (e.g. timeline schematic, manager/worker/critic role taxonomy)
- `manifest/environment.json` - Python, JAX, jaxlib, CUDA, driver, GPU, OS
- `manifest/proof_objects.json` - canonical list with commit hashes
- `manifest/git_state.json` - manager branch HEAD + key sprint commits

## Status

| Item | Status | Source |
|---|---|---|
| paper.md | staged from `publication/draft/paper.md` post iter-2 update | manager branch `05bded8` |
| references.bib | 40 entries, all cited | revision-pass `6f97743` |
| honesty_audit.md | iter-1 state - needs iter-2 amendment | revision-pass `6f97743` |
| tables/performance_summary.md | original split retained; superseded for publication by `tables/performance_evolution.md` | revision-pass `6f97743` + consolidation sprint |
| tables/skill_regression_summary.md | original split retained; superseded for publication by `tables/skill_evolution.md` | revision-pass `6f97743` + consolidation sprint |
| tables/comparators.md | complete - prior-art comparator table | benchmark/tables consolidation sprint |
| tables/m7_gates.md | complete - eight-gate M7 status table | benchmark/tables consolidation sprint |
| tables/sprint_ledger.md | complete - M6/M7 sprint ledger | benchmark/tables consolidation sprint |
| tables/performance_evolution.md | complete - pre-fix / iter-1 / iter-2 performance table | benchmark/tables consolidation sprint |
| tables/skill_evolution.md | complete - CPU / pre-fix / iter-1 / iter-2 skill table | benchmark/tables consolidation sprint |
| tables/test_coverage.md | complete - M6/M7 pytest invariant table | benchmark/tables consolidation sprint |
| figures/timeline.md | complete - figure spec | benchmark/tables consolidation sprint |
| figures/role_taxonomy.md | complete - figure spec | benchmark/tables consolidation sprint |
| figures/validation_pyramid.md | complete - figure spec | benchmark/tables consolidation sprint |
| manifest/environment.json | captured in revision-pass worker-report.md | revision-pass `6f97743` |
| manifest/proof_objects.json | canonical list in `scripts/m7_publication_audit.sh` | revision-pass `6f97743` |
| manifest/git_state.json | TBD - fill at release |

## How to package

Once the benchmark/tables consolidation sprint lands and the Gemini top-level review is incorporated, run:

```bash
bash scripts/m7_publication_audit.sh   # ensures word count + BibTeX parse + proof objects exist
cp publication/draft/paper.md publish/paper/
cp publication/draft/references.bib publish/paper/
cp publication/draft/honesty_audit.md publish/paper/
cp publication/draft/tables/*.md publish/tables/
# user creates github.com/<URL>, imports this folder, tags release, submits to arXiv
```

## Not yet decided

- Public repository URL (user will create when back)
- arXiv version tag
- DOI plan (Zenodo for code + paper version pinning)
- License (project source LICENSE_NOTES.md governs WRF-derivative aspects; arXiv preprint license likely CC-BY-NC or CC-BY)

## Final manager note

This folder exists because the user said: "you can already start with m8 by creating a publish folder, I will create the git later when I'm back." Everything here is staging; the user is the senior corresponding author and final acceptance authority.
