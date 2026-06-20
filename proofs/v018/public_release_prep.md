# v0.18.0 Public Release Preparation — curated 0-PII subset

**Goal:** prepare the v0.18.0 public release for `github.com/wrf-gpu/wrf_gpu`
(remote `wrfgpu`) as a hand-curated subset of the dev tree, provably free of PII.
**Nothing is pushed here** — the manager performs the final push after an
independent 0-PII re-verification.

## Branches / refs

| Ref | SHA | Role |
| --- | --- | --- |
| `sanitize/public` | `c9bc50ce` | Known-good public tree (== `wrfgpu/main` == v0.17.0), 2879 paths, already 0-PII |
| `worker/gpt/v018-integration` | (v0.18 dev) | Source of v0.18 file contents (8184 paths, contains private dirs + PII) |
| `v018-public` (HEAD) | `3cd430a9` | The curated v0.18.0 public release branch (this work) |
| tag `v0.18.0` | annotated, on `3cd430a9` | Staged locally, **NOT pushed** |

## Curation method

`v018-public` was branched from `sanitize/public` (the known-good public tree),
then transformed into the v0.18 public subset by three deterministic operations:

1. **Update already-public paths to v0.18.** For every one of the 2808 paths that
   exist on both `sanitize/public` and `worker/gpt/v018-integration`, the v0.18
   content was taken (`git checkout worker/gpt/v018-integration -- <path>`). 113 of
   them actually changed content; the rest were byte-identical between v0.17 and
   v0.18.
2. **Mirror v0.18 deletions.** 70 paths present on `sanitize/public` were removed in
   v0.18 (superseded v0.16/v0.17 release notes, v0.17 identity-proof assets/proofs
   replaced by v0.18, `data/README.md`, `tests/test_v017_fp32_physics.py`,
   `proofs/perf/v017/*`, `proofs/v016/{dashboard,fp32_verdict,...}`). One additional
   path was a rename (v017→v018 Switzerland identity manifest), bringing the mirrored
   removals to 71 source paths.
3. **Add only vetted new public files.** New v0.18 files were added **only** from the
   allowed public roots: `src/`, `tests/`, `docs/`, `scripts/`, `.agent/skills/`, and
   the v0.18 proof tree `proofs/v018/`. No file outside these roots was added.

**Private content was never added.** The dev tree's private prefixes were excluded
in full: `.agent/{reviews,sprints,decisions,contracts,memory,milestones,...}`,
`publication/`, `codex/`, `artifacts/`, `external/`, `release-assets/`,
`MORNING-REPORT*`, the planning `.md` files (PLANS / RISK_REGISTER / PROJECT_* /
VALIDATION_STRATEGY / PRECISION_POLICY / PERFORMANCE_TARGETS / MILESTONES /
INTERFACE_CONTRACTS / ARCHITECTURE_PRINCIPLES / CONTRIBUTING_AGENT), all `*.pdf`,
and the deep-research `.txt`. `.agent/` on the public tree contains **only**
`.agent/skills`.

## Diff vs `sanitize/public`

```
463 files changed, 119543 insertions(+), 10189 deletions(-)
A (added):    279
M (modified): 113
D (deleted):   70
R (renamed):    1   (proofs/v017/.../switzerland_d01/identity_proof_manifest.json
                     -> proofs/v018/.../switzerland_d01/identity_proof_manifest.json)
```

Path count: `2879 (base) + 279 (added) - 70 (deleted) = 3088`.

## New public files (279)

| Root | Count | Notes |
| --- | --- | --- |
| `src/` | 19 | v0.18 physics/coupling/io/validation modules (cumulus SAS/G3/Grell-Devenyi/KF-previous, PBL Shinhong/GBM, LSM Pleim-Xiu/RUC/SSiB, MP Goddard/SBU-YLin, RA LW HS, LSM static extract, proof_write) |
| `tests/` | 14 | v0.17/v0.18 savepoint-parity, family-status, fail-closed, conditional-state, architecture-boundary tests + `_operational_source_guard.py` |
| `docs/` | 10 | v0.18 identity-proof assets (Switzerland d01 + Canary d02 dashboards/scatter/scoreboard/spatial/timeseries PNGs) |
| `scripts/` | 1 | `build_thompson_aero_tables.py` |
| `.agent/skills/` | 16 | skill CHANGELOGs + 3 `SKILL.proposed.md` + 1 example placeholder README |
| `proofs/v018/` | 219 | full v0.18 proof tree: family-status JSONs (cu/lsm/mp/ra), critic `.md`, perf neutrality/probe JSONs, oracle drivers + savepoints, mp_oracles, identity-proof manifests, integration/honesty/rainnc-qvapor/k2 reports |

(Excluded from `proofs/v018/`: the dev-internal
`identity_proof/canary_l2_d02/RUN_POINTER.json`, a manager-facing run pointer with
an internal release-gate TODO note — not README-referenced, removed from the public
tree.)

## 0-PII gate results (HARD)

All checks run on the committed `v018-public` HEAD (`3cd430a9`):

(PII patterns below are written obfuscated so this doc itself stays 0-PII;
substitute the dev username / dev account for the angle-bracket placeholders.)

| Check | Result |
| --- | --- |
| `git grep -In "<USER_HOME>/<dev-user>"` | **0** |
| private email (`<dev-user>...@`) | **0** |
| garbled over-scrubbed home-path strings (`<USER_HOME>` immediately followed by `-`) | **0** |
| AWS keys (`AKIA…`) | **0** |
| private SSH key blocks (`BEGIN … PRIVATE KEY`) | **0** |
| GitHub tokens (`ghp_…`), Slack (`xox…`), bearer tokens | **0** |
| private SSH remote URL (`git@github.com:<dev-account>/`) | **0** |
| credential assignments (`password/secret/api_key = "…"`) | **0** |
| private-dir files (`.agent/{reviews,sprints,decisions,contracts}`, `publication`, `codex`, `artifacts`, `external`, `release-assets`) | **0** |
| `MORNING-REPORT*` / planning `.md` / `*.pdf` present | **0** |
| `.agent/` subdirs | only `.agent/skills` |
| path count | **3088** (dev tree = 8184) |
| `import gpuwrf` (CPU, `PYTHONPATH=src`) | **OK** |
| version | **0.18.0** (`pyproject.toml` + `src/gpuwrf/__init__.py`) |

Residual `<USER_HOME>` strings (the established sanitize placeholder, already present
on `sanitize/public`) and `<DATA_ROOT>/canairy_meteo` corpus paths (the documented
`GPUWRF_CANAIRY_ROOT` value, no username) are intentional and not PII. The 17
"hermes/telegram" word-matches are proof-object hygiene attestations
(`"no_hermes": true`) carried over unchanged from `sanitize/public`'s
`proofs/v014/*` — no tokens.

## PII scrub applied (4 new files had the dev home path)

Only 4 of the 279 new files contained a `<USER_HOME>/<dev-user>` path; all scrubbed:

- `proofs/v018/path_portability_fix.md` — 7 dev-home-path strings → `<USER_HOME>`
  (and the before→after prose de-garbled so the doc reads correctly).
- `tests/test_v017_lsm_adv.py` — runtime WRF-root read routed through
  `GPUWRF_WRF_ROOT` env (default `<USER_HOME>/...` placeholder); added `import os`.
- `tests/test_v018_mp_family_fail_closed.py` — `WRF_ROOT` constant routed through
  `GPUWRF_WRF_ROOT` env; added `import os`. (Used only under `.exists()`/`skipif`.)
- `tests/test_v017_lsm_pleim_xiu.py` — `wrf_sources` provenance citation strings →
  `<USER_HOME>` (citations, not runtime reads).

## Status

**READY TO PUSH** — pending the manager's independent 0-PII re-verification and the
final push (this worker does not push). The tag `v0.18.0` is staged locally on
`v018-public` HEAD `3cd430a9`.
