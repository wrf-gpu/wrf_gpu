# v0.18 Public-Sanitization Audit (#62)

**Branch:** `worker/gpt/v018-integration`
**Worktree:** `.wt-v018-integration`
**Pre-scrub HEAD:** `5d2a2d2a0acf21f372903320b52fe783a1620b97`
**Public target repo:** `github.com/wrf-gpu/wrf_gpu` (org remote `wrfgpu`)
**Reference standard:** v0.17 sanitization (`sanitize/public` HEAD `c9bc50ce`, prior `a3aa5dec`).

> Note: this audit deliberately does **not** reproduce the literal scrubbed
> tokens (old home-dir path, old username, private account handle, private
> email). They are written in redacted/placeholder form below so this audit file
> itself contains **zero** live PII and stays clean under any pre-push secrets
> scan. `OLDHOME` = the old `/home/<old-username>` path prefix; `OLDUSER` = the
> old lowercase username; `OLDNAME` = the capitalized personal name; `OLDACCT` =
> the private GitHub account handle.

## Scope decision (important)

This branch is the **full private dev tree** (two remotes: a private dev/backup
remote, and `wrfgpu` = public org). The public release ships only a **curated
subset** of paths. The authoritative public path set was taken from the
`sanitize/public` branch tree (2,879 tracked paths), which already carries the
v0.17 scrub (verified: it contains **0** `OLDHOME` and **0** bare `OLDUSER`).

Sanitization was therefore applied to **exactly the files that ship publicly**
(the public path set), matching the v0.17 convention precisely. Pure-private dev
files that never ship (e.g. `.agent/reviews/*`, `.agent/sprints/*`,
`MORNING-REPORT*`, `codex/`, `publication/`) were intentionally left untouched —
they remain only on the private dev remote.

**Verification that scope was correct:** all 261 changed files are members of the
public path set (Python set-diff: 0 changed files outside the public set).

## Scrub rules applied (match v0.17 exactly)

- `OLDHOME` (the old `/home/<old-username>` prefix) → `/home/user` (handled by the
  word-boundary `OLDUSER` rule below).
- bare username `OLDUSER` → `user`, word-boundary safe (`\b…\b`). Verified this
  never touches `generic` (113 occ. preserved), `enrich`/`enriches`/`enrichment`
  (2 occ. preserved) — every word-boundary match was an `OLDHOME/...` path or a
  username token.
- capitalized personal name `OLDNAME` (6 occ.) → contextual **"the user" / "the
  user's"**, identical wording to v0.17 commit `c9bc50ce`
  (e.g. "On `OLDNAME`'s workstation" → "On the user's workstation";
  "If `OLDNAME` meant" → "If the user meant").
- private dev-repo SSH URL containing the personal GitHub account `OLDACCT`, in
  `.agent/skills/managing-sprints/SKILL.md`, → genericized to "the private
  dev/backup remote". This line is **new in v0.18** (not present in the v0.17
  public SKILL.md) — a real finding. The public org remote
  `git@github.com:wrf-gpu/wrf_gpu.git` was preserved (it is public on purpose).
- `/mnt/data` paths **KEPT** (v0.17 decision — generic data mount, not PII;
  112,094 occ. preserved unchanged).
- repo dir name `wrf_gpu2` in paths **KEPT** (matches v0.17 public, e.g.
  `/home/user/src/wrf_gpu2`).

## Before / after grep counts (public path set)

| Pattern | Before (HEAD 5d2a2d2a) | After |
|---|---|---|
| `OLDHOME` (`/home/<old-username>`) | 1432 | **0** |
| bare `OLDUSER` (lowercase username, word-boundary) | 1452 | **0** |
| capitalized `OLDNAME` (personal name, word-boundary) | 6 | **0** |
| `OLDACCT` (private GitHub account handle) | 1 | **0** |
| private email | 0 | **0** |
| any old-username (case-insensitive, excl. `enrich`/`generic`) | — | **0** |
| `/home/user` (replacement, sanity) | 0 | 1432 |
| `/mnt/data` (KEPT, sanity) | 112094 | 112094 |

**Files changed:** 261 (all in the public path set). Diff is pure substitution:
1460 insertions / 1460 deletions (equal — no lines added or removed).

## Secrets scan (public path set) — all clean (0)

- Telegram bot token pattern (`\d{8,10}:[A-Za-z0-9_-]{30,}`): **0**
- Private keys (`PRIVATE KEY`): **0**
- Bearer/Authorization tokens: **0**
- AWS `AKIA…` keys: **0**
- Personal email addresses (gmail/yahoo/hotmail/outlook/protonmail): **0**
- Other `/home/<name>` username-leaking paths (non-`user`): **0**
- Other personal `github.com/<account>` URLs (besides the public `wrf-gpu` org):
  **0** (the remaining `github.com/kokkos/kokkos.git` is a legitimate public
  third-party dependency URL, not PII).

Note: matches for the *words* "telegram", "hermes", "password", "secret",
"api_key" exist only as **references to the concepts** (env-var name
`GRAFCAN_API_KEY`, "no Hermes/Telegram" agent instructions, "no secrets" hygiene
checklist items) — none are actual secret literals.

## README / doc NITs

- **NIT (a) — DONE.** README Canary L2 d02 caption now states its own score for
  symmetry with the Switzerland caption:
  `**Canary L2 d02 — 72 h, nested (9/10, QVAPOR miss; dynamics/thermo core,
  unchanged in v0.18):**`. The `9/10` + worst-field `QVAPOR` is confirmed by
  `proofs/v018/identity_proof/canary_l2_d02/identity_proof_manifest.json`
  (`hard_gate_passes 9/10`, `worst_field QVAPOR`).
- **NIT (b) — NOT changed (intentional, with reason).** The cite
  `proofs/v017/hostgap_fix_opus.md` in `proofs/v018/acecast_reconciliation.md`
  (and `readme_audit.md`) is **valid in the public context**: the file exists on
  the public tree (`sanitize/public:proofs/v017/hostgap_fix_opus.md`) where these
  proofs ship. It only appears "dead" on this dev branch because v0.17's curated
  public proofs include files this dev branch never carried. Removing or
  re-pointing a cite that resolves correctly in the public tree would be
  incorrect, so it was left intact per the brief's "only if obviously correct"
  guard.

## Functional verification (scrub changed only strings, not behavior)

- All `src/` edits are comment / docstring / source-citation **string-literal**
  changes only. Verified: every added `src/` line matches `/home/user` or
  "the user"; zero logic/path/env-var-name changes.
- `python3 -m py_compile` on all changed `.py` files: **OK**.
- `import gpuwrf` (+ all scrubbed src modules: `config.paths`, `io.gen2_accessor`,
  `runtime.compile_cache`, `runtime.jax_cache`, `init.forcing_decode`): **OK**.
- Fast representative CPU test subset (no GPU, no large fixtures):
  `test_adr_001_structure`, `test_fixture_manifest_schema`, `test_cli`,
  `test_fixture_manifest_edge_cases`, `test_agentos_smoke` → **34 passed**.
  (Identical pre-scrub baseline was green; the comment-only scrub cannot affect
  runtime behavior, so the GPU operational-smoke was not required.)

## FINAL CONFIRMATION

**ZERO PII and ZERO secrets remain in the public-shipping tree.** Confirmed by
final re-audit across the full public path set: 0 old-home-dir paths, 0 bare
old-username, 0 capitalized personal name, 0 private account handle, 0 private
email, 0 token/key/credential literals, 0 private-account URLs. `/mnt/data` and
`wrf_gpu2` preserved per the v0.17 standard. Not pushed or tagged — manager
performs merge + tag + push after review.
