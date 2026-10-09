# Repository history cleanup — release preparation record

**Status: approved, not yet pushed or verified.** Update this note with the
verified rewritten refs and map checksums before publication.

The release includes a narrowly scoped removal of compromised profiling and
environment-capture artifacts from both repository histories. Forecast code,
release content and model inputs are preserved. Credential rotation is a
separate owner action; approval is not confirmation that it has completed.

Removing an earlier artifact changes that commit's identifier and the
identifiers of its descendants, including affected tags. Old commit hashes
in validation receipts therefore need the retained **old-to-new commit map**;
the ref map records branch and tag changes. Forecast source-tree hashes
and frame SHA256 checksums remain the numerical binding. The cleanup owner
must demonstrate unchanged `src/` and `src/gpuwrf/` trees for cited forecasts,
seal both maps with SHA256, and verify rewritten remote refs before this
preparation record becomes a completed-cleanup notice.

After the verified history replacement, **fresh clones are recommended**.
Save local edits and coordinate outstanding branches and worktrees before
switching to the new history. Do not merge or push a stale branch into the
cleaned repository: that can restore removed objects. Existing installations
do not acquire different forecast numerics merely because commit IDs change;
use the release tag and verified source-tree identity to identify the model.

The release bundle excludes raw profiler binaries, SQLite exports, environment
dumps and raw logs. Cleanup validation scans the complete rewritten object set,
including binary blobs and every retained ref/tag; scanning patch text alone
does not establish that history is clean. Reports contain names, paths, counts
and hashes, never credential values.

## Evidence pending from the cleanup owner

- SHA-sealed commit map and ref map for each repository.
- Pre-rewrite backup refs retained until remote verification completes.
- Exact removed paths, affected commit/tag counts and object/size deltas.
- Full-object rescan and post-push remote ref verification.
- Unchanged forecast source-tree proofs for the cited release evidence.

These are publication dependencies, not claims of completed work. Release
frames and existing receipt hashes remain untouched; map commit citations
through the sealed evidence rather than relabelling old measurements.

## Signature provenance (preparation record audit)

The four invalidated public OpenPGP signatures belong to **commits**, not tags: `1a4b91f4b679874d4c6ca5aebd57c48299800a39`, `4a941f58ed3cb12c5bb468ac3df29b19da205382`, `92c65bc3a04067f65068924ba0d583b123f2a932`, `bd5cb18ba2849f5bbf3d8300811e42275e271df2`. The audit found zero signed tags. Unsigned alias repairs for v0.15.0/v0.21.0/v0.23.4/v0.3.0 are separate. preparation record mirrors are not the final release-inclusive maps. The private rewrite additionally removes the one explicitly approved 199,826,803-byte E170 blob; other oversized local objects remain private and outside selected publication refs. Credential rotation remains unconfirmed.
