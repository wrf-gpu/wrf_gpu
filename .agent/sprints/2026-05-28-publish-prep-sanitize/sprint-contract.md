# Sprint Contract — Publish-Prep Sanitize + Install Docs

**Sprint ID**: `2026-05-28-publish-prep-sanitize`
**Created**: 2026-05-28
**Status**: READY — parallel-safe with Sprint #3 RE-DO (CPU-only, operates on `/home/enric/src/wrf_gpu/`, not `/home/enric/src/wrf_gpu2/`)
**Predecessor**: the new public clone at `/home/enric/src/wrf_gpu/` is set up with code/tests/scripts/paper/tables/figures/manifest/proofs + LICENSE + README + NOTICE + CITATION.cff + CONTRIBUTING.md + AI_USE.md + SECURITY.md + CODE_OF_CONDUCT.md + CHANGELOG.md + .gitignore + .github/{ISSUE_TEMPLATE,workflows,PULL_REQUEST_TEMPLATE.md}

## Objective

Sanitize the public clone at `/home/enric/src/wrf_gpu/` so it is shippable as `v0.0.1`. Remove every reference to personal paths, the development machine's data layout, internal manager artefacts, and email. Generalise where the code legitimately needs a configurable path. Add "one-paste install" instructions and a minimal `environment.yml` / `requirements.txt` for reproducible installation.

The goal is: a stranger with an NVIDIA GPU can `git clone` the new repo and follow a single block of install commands to get running unit tests.

## Acceptance

- **AC1 — Path sanitisation**: replace every occurrence of `/home/enric/src/wrf_gpu2/` and `/home/enric/src/wrf_gpu/` in the public repo's `*.md`, `*.py`, `*.sh`, `*.toml`, `*.yml`, `*.json`, `*.cff` files with relative paths anchored at the repo root. Example: `/home/enric/src/wrf_gpu2/.agent/sprints/...` → `proofs/...` or `<dev-history>` depending on context. Inside source code, this typically means accepting the path from a constructor argument or environment variable rather than hardcoding it; do this without changing semantics.

- **AC2 — Site-specific reference data**: replace `/mnt/data/canairy_meteo/...` with a configurable variable. In code: accept the path from `WRF_GPU_REFERENCE_ROOT` env var (or a constructor argument) and document the variable in README + INSTALL. In docs/JSON proofs: replace with the placeholder `<reference-data-root>` and document in a README section.

- **AC3 — Internal manager paths**: every `.agent/sprints/<...>/` reference in *.md and *.py files in the public repo must either point to (a) the corresponding `proofs/<sprint-id>__<filename>.json` already staged, or (b) `<development-history-not-included-in-public-repo>` for cases where the sprint history is genuinely internal. Update `paper/paper.md` and `paper/honesty_audit.md` and the tables to point at `proofs/...` consistently. Do NOT modify the proof JSONs themselves more than path-cleaning.

- **AC4 — Email + author identity**: ensure `Enric R.G.` (without email, without further personal info) is the only human-name reference in the public repo. Remove any `enric.r.g@*`, `@gmail`, full-name variants. Keep "Enric R.G." in author/credits where appropriate (CITATION.cff, paper byline, README acknowledgements).

- **AC5 — INSTALL.md** (NEW at repo root): write a `INSTALL.md` with three sections:
  1. **Unit-test install** (anyone, no GPU needed): one-paste bash block: clone → venv → pip install → pytest -q tests/.
  2. **Full GPU install** (NVIDIA RTX 30/40/50-series, 12+ GB VRAM recommended; tested on RTX 5090): adds JAX-CUDA install, environment-variable setup for reference data path, smoke test command.
  3. **Reproducibility install** (pin exact versions for reviewers): `pip install -r requirements-frozen.txt`, exact Python/JAX/jaxlib/CUDA/driver versions, reproduce wall-clock + skill numbers from the paper.

- **AC6 — requirements files**:
  - `requirements.txt` (loose, current minor versions; the regular path)
  - `requirements-frozen.txt` (exact versions captured at v0.0.1 release; reviewers use this to reproduce paper numbers)
  - `environment.yml` (optional conda equivalent for users who prefer conda)
  Use `pip freeze` against the dev machine's working environment to seed `requirements-frozen.txt`.

- **AC7 — One-paste install verification**: provide a single bash snippet in the README's "Quick start" section that a stranger can copy-paste into a fresh Ubuntu 22.04+ shell and have a working `pip install -e .` + `pytest -q tests/ -k 'not gpu'` running within 10 minutes. The snippet must be tested by the worker via a fresh `python -m venv /tmp/test_install_venv` install.

- **AC8 — Verify-reproducibility script**: write `scripts/verify_reproducibility.sh` that runs the lightweight subset of tests + the audit script and produces a single PASS/FAIL line. Anyone running it reproduces the unit-level evidence in seconds.

- **AC9 — Public README, INSTALL, CONTRIBUTING cross-references**: ensure links between docs work. Add a "How to verify the paper numbers" section pointing reviewers at `scripts/verify_reproducibility.sh` and the relevant `proofs/*.json` files.

- **AC10 — Final scrub check**: run a final grep for these patterns; AC10 fails if any non-empty hit remains in *.md, *.py, *.sh, *.toml, *.yml, *.json, *.cff:
  - `/home/enric`
  - `/mnt/data/canairy`
  - `enric.r.g@`
  - `@gmail`
  Exception: the file `AI_USE.md` and `CHANGELOG.md` may mention "Enric R.G." without paths. The proof JSONs may retain timestamps; only personal paths are scrubbed.

- **AC11 — Worker report**: verdict `PUBLISH_READY_PENDING_TESTS` (meaning: ready for v0.0.1 push *as soon as* Sprint #3 RE-DO completes successfully); the manager flips to PUBLISH_READY when Sprint #3 lands GREEN.

## Files Worker May Modify

- Any file under `/home/enric/src/wrf_gpu/` (the public repo).
- New files: `INSTALL.md`, `requirements.txt`, `requirements-frozen.txt`, `environment.yml`, `scripts/verify_reproducibility.sh`.

## Files Worker Must Not Modify

- Anything under `/home/enric/src/wrf_gpu2/` (the manager repo — Sprint #3 RE-DO is running there).
- Anything under `/tmp/wrf_gpu2_*` (active worktrees).
- `LICENSE` (verbatim AGPL-3.0 must not be edited).
- The proof JSON file *content* beyond path-string cleaning (semantic content stays).

## Hard Rules

1. **No semantic code changes** — only path-string substitutions and configuration plumbing. If a hard-coded path is required for legacy reasons, replace with `os.environ.get("WRF_GPU_REFERENCE_ROOT", <relative-fallback>)` rather than another hard-coded path.
2. **No GPU runtime.** This sprint is CPU-side rewriting.
3. **CPU pinning**: `taskset -c 0-3`.
4. **No git commits to wrf_gpu2** — all commits go to the new public repo at `/home/enric/src/wrf_gpu/` (which is a fresh git repo with origin `git@github.com:wrf-gpu/wrf_gpu.git`).
5. **No remote push.** The manager will push after all sprints + tests land. Commit to local main only.
6. **Verify on a fresh venv** for AC7 — actually run the one-paste snippet.

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-4 h
- Branch: `main` of `/home/enric/src/wrf_gpu/` (the new public repo); commit there directly (it's a fresh repo, no merge conflicts possible at this stage)
- Worktree: `/home/enric/src/wrf_gpu/` (operate directly, no /tmp/ copy)
- GPU usage: NONE

## Post-sprint sequence (manager-driven, not this sprint)

1. Sprint #3 RE-DO lands → merge to wrf_gpu2 manager repo → re-stage the updated proof JSONs into public repo's `proofs/` if needed
2. Sprint #4 opus check of sprint #3 evidence → dispatch
3. Sprint #5 paper rewrite (port-first focus) → dispatch in `/home/enric/src/wrf_gpu/paper/`
4. Sprint #6 opus paper control → dispatch
5. PDF generation against final paper.md
6. Tag v0.0.1 + push to `git@github.com:wrf-gpu/wrf_gpu.git`
