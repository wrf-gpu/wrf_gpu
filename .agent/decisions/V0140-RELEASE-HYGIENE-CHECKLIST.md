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

## G. Added anti-slop items (Opus release-cleanup worker, 2026-06-12)

Additional things a serious scientific software release should get right that
were not explicit above. `[done]` = handled in the `opus-release-cleanup`
branch; `[manager]` = needs the manager (gate numbers, GPU smoke, or a decision).

- **Version-string consistency.** `pyproject.toml` `version` must match the
  release tag, and the CUDA extra (`jax[cudaNN]`) must match the README's pinned
  CUDA. `[done — bumped 0.13.0 → 0.14.0; reconciled cuda12 → cuda13]`
- **No misrepresented defaults in docs.** A documented default path/value must
  match what the code actually does (e.g. the JIT cache default was documented
  as a private `/mnt/data` path but the code uses a portable per-user dir).
  `[done — resource-profile.md fixed]`
- **Cross-doc figure consistency.** The same quantity must read the same across
  README/quickstart/KNOWN_ISSUES (e.g. the wrfout writer variable count was
  "64" in quickstart vs "104" elsewhere). `[done — quickstart fixed to 104]`
- **Standard repo files present and real.** `LICENSE` (actual license text, not
  just notes), `CITATION`/`CITATION.cff`, `CONTRIBUTING.md` (user-facing, not
  only the agent-process file), `CHANGELOG.md`, top-level `KNOWN_ISSUES.md`.
  `[done — CONTRIBUTING.md, CHANGELOG.md, KNOWN_ISSUES.md added; LICENSE_NOTES
  placeholder cleaned] [manager — LICENSE + CITATION.cff pending the license
  decision]`
- **`pip install -e .` resolves on a clean env** and the package imports +
  console script runs CPU-only (the real new-user path, GPU not required to
  install). `[done — verified in a fresh venv on CPU; jax CpuDevice, gpuwrf
  import OK, `gpuwrf run --help` OK]`
- **Scripts referenced by docs have no hard-coded personal interpreter/data
  paths.** Personal `PYTHON=/home/enric/...` and `/mnt/data/...` defaults in
  user-facing scripts/runbook must be env-overridable with portable defaults or
  clearly framed as the maintainer's reference layout. `[done — run_powered_tost_n15.sh,
  verify/_common.sh, wrf_rrtmg_harness_build.sh, GPU_RUNBOOK.md, IDENTITY_PROOF.md]`
- **`.gitignore` covers agent/tooling sandboxes + editor cruft.** `.codex/`,
  `.claude/worktrees/`, `cache/`, `.agents/`, `*~/*.swp/*.bak/.DS_Store`.
  `[done]`
- **Empty/placeholder-only committed docs.** Egregious 0-byte/placeholder-only
  `.md` files read as half-written; give them an honest one-line note or remove.
  `[done — 3 empty agy review/council .md given honest empty-result notes;
  empty .stderr/.stdout command-logs and .done sentinels left as intentional
  sprint evidence]`
- **Archive framing.** `.agent/README.md` must explicitly frame the archive as a
  point-in-time development log whose verdicts are NOT current truth, and point
  to the authoritative top-level docs. `[done]`
- **Secrets/keys scan.** `[done — repo-wide tracked-file scan: no API keys,
  tokens, private keys, or password literals found]`
- **`<<MANAGER-FILL>>` / `<manager: ...>` placeholders are the only intentional
  blanks** in shipped docs, and they are clearly marked. No other slop markers
  in the top-level/user-facing set. `[done — verified; the gate-number blanks
  are the manager's to fill]`

### Still needs the manager (out of CPU-only scope)

- **Final 72 h gate numbers** for README / RELEASE_NOTES / KNOWN_ISSUES
  (`<<MANAGER-FILL>>` / `<manager: ...>` placeholders).
- **Live GPU smoke** of the documented Switzerland run end-to-end (a real
  forecast hour) after the gates free the card — the one runnability step that
  cannot be CPU-verified.
- **README/RELEASE_NOTES version-narrative bump v0.13.0 → v0.14.0** (the prose
  still reads "v0.13.0"; this is release-content the manager owns alongside the
  gate numbers, kept out of this hygiene branch to avoid inventing results).
- **LICENSE + CITATION.cff** once the license is chosen.
