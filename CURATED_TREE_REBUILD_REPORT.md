# v0.23.4 curated-public tree rebuild report

## Verdict and provenance

**READY_FOR_MANAGER_REVIEW — NOT PUSHED, TAGGED, OR PUBLISHED.**

The stale v0.23.3-derived overlay was discarded only after a status/diff/mtime
review showed that it was the superseded release-prep snapshot described by the
gap critic. A recoverable patch and untracked-file archive were retained under
`/tmp` before cleanup.

| Item | Value |
|---|---|
| Staging branch | `wrfgpu-v024-release-prep` |
| Public base | `07f4765e1f92599f1587f7fdcf84040010c1ec68` (v0.23.3) |
| Pinned private `main` source | `3ff88021f53e8da3b56ea590e72b9328b7c91b38` |
| Release version | `0.23.4` |
| Publication action | None; manager owns final review, push, and tag |

## Rebuild and curation

The tree was reconstructed from the pinned private commit using the established
public-base-plus-curated-overlay convention. The selected production source,
public tests, four reusable scripts, six proof objects, release/operator
surfaces, docsite tooling/content, and all 25 `RELEASE_NOTES*.md` files were
refreshed from that commit.

The standard global substitutions were then applied:

- live owner-home prefix -> `<USER_HOME>`;
- live data-root prefix -> `<DATA_ROOT>`;
- the historical operational-signoff example uses `owner_signoff`;
- the public sprint-skill heading uses `owner-directed`.

Substitution is not limited to prose: it applies to any selected text file
containing a live workstation path. Consequently, 22 `src/gpuwrf` files and 50
test files contain generic placeholders. After reversing only the two path
placeholders for comparison, all 325 files under `src/gpuwrf` match the pinned
private tree with zero missing, extra, or differing files. All 426 published
test files likewise have zero normalized differences.

The following established exclusions were preserved:

- 210 sprint-specific `scripts/v0234*` diagnostic/replay programs; the four
  reusable scripts selected by the prior release precedent are included;
- 30 large NetCDF savepoint fixtures under
  `tests/savepoint/fixtures/wrf_b6_100step/`; all are binary oracle payloads,
  not test source;
- private management material under `.agent/sprints`, `.agent/decisions`,
  `.agent/reviews`, and other private publication/worker roots;
- ten unreferenced v0.17 identity-proof PNGs that were not part of the
  established public asset set.

The public tree retains three historical v0.20 benchmark plots that already
belong to the public v0.23.3 base. All 47 textual files under `docs/` were
refreshed from the pinned commit and have zero normalized differences.

## Exact source, hash, and version verification

The release-critical files below are byte-identical to the pinned private
source; none contains a substituted path.

| File | SHA-256 | Exact |
|---|---|---|
| `src/gpuwrf/dynamics/flux_advection.py` | `0d784269ab29d36f8726928220b1e118d347b0436915ab53edf5243cc1f1ada4` | yes |
| `src/gpuwrf/runtime/operational_mode.py` | `0bd53237112b64069a8c1204c2d9f7ca3c1b21c67f439b4ea94b8d506d31d501` | yes |
| `src/gpuwrf/runtime/aot_cheap_key.py` | `9b6ea22e13715711b458a6bcda143007a421251ca063012dcbe57ae47e82989e` | yes |
| `src/gpuwrf/runtime/aot_precompile.py` | `5ec9512898204482e0e65cac891e15e748c7add7a58a8cb79b777dad9720cb65` | yes |
| `src/gpuwrf/runtime/domain_tree.py` | `f588414b1c08ff744f6ecf13d21f5931957df83e4f94d9cf14ad3db7ee6aa408` | yes |
| `tests/dynamics/test_pd_moisture_advection.py` | `210662bffd5cfc90620cb8a33aeb7fb2eb504a231eebfb0e8f52c20664f2e7ed` | yes |
| `tests/test_aot_cheap_key.py` | `c99297b791dcb339ca4ad9037dace05226fe7a7324e854dbdc91f5ae2f5560e2` | yes |
| `tests/test_aot_executable.py` | `b7adaa58b208de56817e4e8d68607e0f10785725a675ef3143ee050acf639c63` | yes |

Both canonical declarations parse as `0.23.4`:

- `pyproject.toml` -> `project.version`;
- `src/gpuwrf/__init__.py` -> `__version__`.

## Documentation verification

The docsite was regenerated **after** all substitutions with
`python3 tools/docsite/build.py`.

- 19 content pages plus `search.html` were built.
- Both 134-entry search indices are byte-identical.
- `docs/.nojekyll` is present.
- 908 generated-site local links, assets, and anchors resolve with zero errors.
- The Markdown audit checked 206 local references. Twelve historical references
  intentionally point into excluded private management/publication material;
  there are zero unexpected missing targets.
- No live workstation paths survive in generated output.
- The public release surfaces consistently disclose the sealed correctness
  result, unaffected two-domain range (`0.960-1.018x`), and the known
  occupancy-bound 3+/9-domain d03 regression (`51.67%`).

## PII and secrets scan

A new recursive scan was run on this rebuilt publish tree, excluding Git
metadata but including tracked and untracked publish content. It inspected
3,046 UTF-8 text files among 3,416 total files.

| Pattern class | Matches |
|---|---:|
| Live owner-home/data-root prefixes | 0 |
| Owner name/handle forms | 0 |
| Local private IPv4 ranges | 0 |
| Personal email addresses | 0 |
| GitHub token forms | 0 |
| AWS access-key forms | 0 |
| Google API-key forms | 0 |
| Slack token forms | 0 |
| OpenAI key forms | 0 |
| Telegram bot-token forms | 0 |
| Bearer/secret-assignment forms | 0 |
| Private-key blocks or key-like filenames | 0 |

The remaining generic home examples and `<USER_HOME>` / `<DATA_ROOT>` strings
are established public placeholders, not live owner data.

## Commands and proof checks

The worker ran the following classes of checks:

- `git status`, `git diff`, and mtime review before stale-overlay cleanup;
- exact-commit `git archive` reconstruction and normalized recursive parity;
- `python3 tools/docsite/build.py`;
- SHA-256 and parsed-version verification;
- recursive PII/secrets/filename scanning of the exact publish file list;
- generated HTML link/asset/anchor and Markdown-reference checks;
- `git diff --check` (three upstream-exact blank-line-at-EOF warnings; see
  unresolved risks);
- read-only Python `compile()` over changed/new Python files: 143 parsed,
  zero syntax errors;
- the gap critic's 85-case focused pytest selection.

The focused pytest selection produced 74 passes, 5 skips, and 6 expected
fixture-path failures in this sanitized tree. Those six cases directly open
authenticated CPU-WRF files through the literal `<DATA_ROOT>` placeholder.
The test module also imports one deliberately excluded diagnostic oracle, so
the private scripts directory was supplied read-only on `PYTHONPATH` for
collection. This does not contradict the gap critic's 85/85 result on the
unsanitized private release candidate, but it means that focused gate is not
standalone-runnable from the curated tree without configuring the placeholder
and supplying the private authenticated fixture/oracle.

## Unresolved risks and next decision

- The manager must independently review this commit and decide whether the
  release protocol may proceed to public push and tag.
- `git diff --check` reports a trailing blank line in each of
  `proofs/v024/real1km_9nest_nonfinite/INCIDENT_20260709.md`,
  `tests/test_v0234_post_fable_corner_window.py`, and
  `tests/test_v0234_theta_unlimited_source_repair.py`. These bytes are retained
  because the curated copies must match the pinned private commit exactly.
- The 3+/9-domain d03 performance regression remains the documented shipping
  limitation; this rebuild does not claim to fix or remeasure it.
- No full suite, GPU forecast, profiler, push, tag, or publication action was
  performed by this worker.
