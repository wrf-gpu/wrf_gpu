# v0.14 Release Hygiene Checklist ("clean science-grade release, not AI slop")

Date: 2026-06-12
Owner: manager (Opus 4.8)
Principal directive: the release must look like a proper proof that AI can make
clean releases of complex science-grade software. Two explicit asks:
1. Impressive identity-proof plots (all cells / all times / all internal
   variables, BOTH Canary and Switzerland 72h) showing the port is true to
   WRF v4 — and that a proper identity-proof SYSTEM exists.
2. A pre-push Opus cleanup worker that audits every released md, checks for
   overhead, and guarantees an external user following README.md can run the
   Switzerland case and it just works.

This runs in the release-prep phase: AFTER the final field gates pass, BEFORE
tag + push. The AI-process md files ARE part of the release (transparency), but
must be coherent and professional, not a dump.

## A. README runnability (the headline external-user test)

- A user with a fresh clone + documented prerequisites can run the Switzerland
  case end-to-end and it WORKS — no trivial uncaught error.
- Pin exact prerequisites: OS, Python version, CUDA/driver, JAX version, GPU
  VRAM minimum, conda/pip env file, disk needs.
- Every README command is copy-pasteable and verified on a clean environment
  (ideally a fresh venv/container), not just on this workstation.
- Input data: a clear, working path to obtain or generate the required inputs
  (CPU truth / wrfinput / met_em). If not redistributable, ship or document a
  small reproducible example + a documented full-case route.
- ZERO hardcoded absolute paths (`/home/enric`, `/mnt/data`) in any user-facing
  instruction or default; everything config/flag-driven with sane defaults.
- A fast quickstart smoke (minutes) PLUS the full 72h; expected output shown.
- Documented identity-verification step a reviewer can re-run (the compare +
  atlas), with expected pass criteria.

## B. Repository cleanliness (no shipped slop)

- Remove/relocate stray dirs from the release: `.codex/`, `cache/`, agent
  worktrees, scratch outputs, `/tmp` leakage. Extend `.gitignore` to cover them.
- Decide test fixtures: `tests/savepoint/fixtures/wrf_b6_100step/{golden,patch16}`
  — commit if tests need them (and they are reasonably sized) or gitignore.
- No secrets, API keys, tokens, or personal paths/emails in committed files.
- No `__pycache__`, editor backups, `.DS_Store`, broken symlinks.
- No orphaned/duplicate scripts; consistent naming.

## C. Documentation hygiene (md files — incl. the 2077 .agent md)

- README, KNOWN_ISSUES, PROJECT_PLAN, PROJECT_CONSTITUTION, AGENTS, and the
  current roadmap/decisions: PRISTINE — accurate, current, no stale claims, no
  `PLACEHOLDER`/`<VERDICT…>`/`TODO`/`FIXME`/`TBD`, no half-written sections.
- The AI-process archive (.agent/reviews, sprints, decisions): release as a
  clearly-framed DEVELOPMENT LOG with an index/README that explains it is the
  historical AI process record, so a stale/superseded conclusion is never
  mistaken for current truth. Clean egregious slop and fix or remove broken,
  contradictory, or placeholder-only files. Curate or archive noise; the
  released set must read as coherent process evidence, not a dump.
- KNOWN_ISSUES is honest and current: RAINNC bounded precip sensitivity,
  GRAUPELNC source-fidelity gap, focused-writer field subset, tier3_coupled
  double-count, performance ~1.05× (→ v0.15), any unfixed gate miss.
- Every headline claim (identity, validation, speed, memory) links a proof
  artifact. No over-claiming (honest, measured, falsifiable).
- License, CITATION, CHANGELOG/release-notes, CONTRIBUTING present + correct.
- A clear release narrative: v0.14 = memory + WRF-identity release; what is
  validated (72h field parity both regions); what is deferred to v0.15
  (performance, completeness).

## D. Code hygiene (user-facing + core paths)

- No leftover debug prints, commented-out blocks, dead code, unused imports in
  core/user-facing paths; env-gated debug OFF by default.
- No hardcoded personal paths in code; config/flag-driven.
- Documented test command; the named test suite passes from a clean checkout.

## E. Identity-proof plots (principal ask #1)

- Compelling visual proof of WRF-v4 identity for BOTH Canary L2 d02 and
  Switzerland d01, on the FINAL v0.14 candidate runs:
  - per-variable RMSE/bias time series across all 72 h for ALL core internal
    variables (U,V,W,T,P,PH,MU,QVAPOR,…), with the bound/limit drawn;
  - variable×lead "scoreboard" heatmap (normalized error) — green everywhere;
  - GPU-vs-CPU cell-value scatter / 1:1 identity plots per variable;
  - spatial GPU−CPU difference maps at key leads for the main variables,
    showing differences at/near roundoff across all cells;
  - one polished summary dashboard per region, README-embeddable.
- Existing base to extend: `scripts/build_grid_delta_atlas.py` (currently emits
  timeseries + error heatmaps + 4 spatial maps). Generalize to all gated
  variables + scatter/identity panels + a publication-quality dashboard.
- The procedure must be reproducible (documented command), so the plots are
  evidence of a real identity-proof SYSTEM, not one-off figures.

## F. Honesty / provenance

- Speed framed honestly (~1.05× now; memory+identity release; performance is
  the v0.15 focus). No performance headline for v0.14.
- Identity claims backed by the plots + the atlas/compare gate with the
  pre-declared tolerance manifest.
- Bounded acceptances (RAINNC) recorded with their numeric justification.

## Deliverables

- Identity-proof plot suite (committed assets + the reproducible script) for
  both regions on the final runs.
- A clean, runnable README verified on a fresh environment.
- A curated, coherent released doc set (incl. the AI archive index/framing).
- A short RELEASE_NOTES / CHANGELOG for v0.14.
- An honest KNOWN_ISSUES.
