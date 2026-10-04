# v0.21.0 release readiness checklist

Date: 2026-06-24
Worktree: `<USER_HOME>/src/wrf_gpu2_wt/v021-integration`
Branch: `worker/opus/vnext-v021-integration`
Scope: docs/release-readiness only; no GPU; no model-code changes; no push.

## Release framing

Binding order:

```text
STABILITY > IDENTITY > SPEED > MEMORY
```

Critical honesty constraint:

- The v0.21.0 dycore change is a **mechanism fix**: acoustic dry-mass-drain
  limiting plus positive `c2a`/`alt` conditioning.
- It is recorded as zero-new-regression against the v0.20.2 known-red CPU
  baseline.
- It does **not** claim full stabilization of the most-extreme 1 km Mont-Blanc
  terrain.

Required public slot kept:

`[SLOT: Mont-Blanc-extreme limitation framing - finalizes after the release A/B decision + the Canary gate result]`

Internal decision dependency: pending the user A/B scope decision plus the final
Canary gate result.

## Done

- [x] Version strings already synced locally: `pyproject.toml` and
  `src/gpuwrf/__init__.py` both report `0.21.0`.
- [x] Public README v0.21.0 section applied to `README.md`.
- [x] Developer / AI-agent README v0.21.0 section applied to `.agent/README.md`.
- [x] `CHANGELOG.md` updated with a v0.21.0 entry.
- [x] `RELEASE_NOTES_v0.21.0.md` created.
- [x] `RELEASE_NOTES.md` updated away from stale v0.14 content and now points to
  `RELEASE_NOTES_v0.21.0.md`.
- [x] Public-facing PII/internal-path scan is clean for:
  `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`,
  `RELEASE_NOTES_v0.21.0.md`, `RELEASE_NOTES_v0.20.2.md`.

PII/internal-path scan command:

```bash
rg -n '(/home/[^ )`]+|<DATA_ROOT>[^ )`]+|\.agent|worker/[A-Za-z0-9._/-]+|git@github.com:<owner>|<owner>/|the user|PRIVATE|SECRET|TOKEN|PASSWORD|CREDENTIAL)' \
  README.md CHANGELOG.md RELEASE_NOTES.md RELEASE_NOTES_v0.21.0.md RELEASE_NOTES_v0.20.2.md || true
```

Result: no matches.

## Pending release slots

- [SLOT] Canary gate result.
- [SLOT] the user A/B scope decision.
- [SLOT] Final Mont-Blanc-extreme limitation wording.
- [SLOT] Final steep-terrain gate artifact: command, fixture version, duration,
  d01/d02 finite status, artifact path, release commit.
- [SLOT] Real prewarm artifact proof: artifact name, version tag, sha256, entry
  count, unpack/readiness proof.
- [SLOT] Parallel-compile target-image proof: accepted/rejected flags and
  no-abort record.
- [SLOT] Fused vs de-fused nested compile A/B: compile RAM, cold wall, warm wall,
  identity result.
- [SLOT] Final measured warm-throughput budget update.
- [SLOT] Paid large-GPU training output: not claimed in v0.21.0.

## Release-protocol checklist

- [x] **Dual README**: public `README.md` and developer `.agent/README.md` updated.
- [x] **Release notes + changelog**: v0.21.0 notes and changelog entry staged in
  the integration worktree.
- [x] **Public PII/internal scan**: clean on release-facing public docs.
- [x] **Local version-sync check**: `pyproject.toml` and `src/gpuwrf/__init__.py`
  both show `0.21.0`.
- [SLOT] **Gap-analysis critic**: run final pre-tag critic after Canary gate and
  A/B decision. The critic must verify no overclaim, no missing Mont-Blanc
  limitation, no stale speed claim, and no public internal-path/PII regression.
- [SLOT] **Final public curation scan**: re-run the public scan after filling
  slots and before tag/push.
- [SLOT] **Version-sync remote check**: after final release commit, before push,
  verify local working tree, `origin/main`, and `wrfgpu/main` carry the same
  version string.

Version-sync commands for the final tag/push operator:

```bash
rg -n "__version__|version =" src/gpuwrf/__init__.py pyproject.toml
for r in origin wrfgpu; do
  git show "$r/main:src/gpuwrf/__init__.py" | grep __version__
  git show "$r/main:pyproject.toml" | grep '^version ='
done
```

Tag/push checklist once slots are filled:

1. Fill all `[SLOT]` release-note and README values from the final GPU gate and
   the user A/B decision.
2. Re-run the public PII/internal-path scan above.
3. Run the final gap-analysis critic.
4. Confirm `git status --short` contains only intended release docs/proofs.
5. Commit final release docs.
6. Tag `v0.21.0` on the private release commit.
7. Prepare the sanitized public `wrfgpu` commit with the same version string.
8. Tag `v0.21.0` on the public commit.
9. Push `origin` branch/tag and `wrfgpu` branch/tag.
10. Run the version-sync remote check and update the release ledger.

This prep pass intentionally did **not** push.

## Commands run

- `sed` reads of `PROJECT_CONSTITUTION.md`, `AGENTS.md`, local docs skills, and
  the v0.21 release prep drafts.
- `git status --short --branch` on the integration worktree.
- `rg --files` / `find` to locate README, changelog, release-note, proof, and
  skill files.
- `rg -n "__version__|version =" src/gpuwrf/__init__.py pyproject.toml`.
- Public PII/internal-path scan command shown above.

## Files changed by this prep pass

- `.agent/README.md`
- `README.md`
- `CHANGELOG.md`
- `RELEASE_NOTES.md`
- `RELEASE_NOTES_v0.21.0.md`
- `proofs/v021/RELEASE_READINESS_v0210.md`

## Unresolved risks

- The final Mont-Blanc wording is deliberately still slotted.
- The Canary gate and the user A/B decision can change final release scope.
- GPU-gated compile/prewarm/de-fuse performance numbers are not filled and are
  not claimed.
- Public release still needs the final gap-analysis critic and version-sync
  remote check before tag/push.
