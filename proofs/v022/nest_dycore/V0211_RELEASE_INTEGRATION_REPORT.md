# v0.21.1 Release Integration Report

Date: 2026-06-27
Branch: `release/v0.21.1`
Base: `v0.21.0`

## Objective

Package v0.21.1 as `v0.21.0` plus only the manager-verified Mont-Blanc
specified-boundary fix, replacing the superseded release branch with the verified
boundary mechanism.

## Minimal Changeset

The release branch was reset to `v0.21.0`, then rebuilt from the verified fix
patches:

- `9767a157`: specified-boundary cadence and optional scalar boundary leaves for
  standalone `wrfbdy` roots.
- `a09ad47e`: WRF-faithful specified-domain `zero_grad_bdy(W)` on reconstructed
  physical `W`, including the correct corner source-index behavior.

Excluded from v0.21.1:

- superseded release stack containing the old alternate attempt, including
  `9f4ab43e`;
- old source-branch proof payloads from the fix development worktree;
- `src/gpuwrf/physics/thompson_column.py` and any Thompson cap logic.

Release packaging added only the `0.21.1` version bump, changelog/release notes,
this report, and sanitized gate summaries.

## Gates

CPU focused suite:

```text
JAX_PLATFORMS=cpu PYTHONPATH=src pytest -q \
  tests/test_v014_specified_bdy_cadence.py \
  tests/test_v017_qh_hail_state.py \
  tests/test_v016_thompson_aero_threading.py \
  tests/contracts/test_v060_physics_interfaces.py \
  tests/test_m7_restart_checkpoint_roundtrip.py \
  tests/test_m6_precision_matrix.py

38 passed, 3 skipped in 28.91s
```

Mont-Blanc native-dt max-dom2, 1 h, fresh JIT (`GPUWRF_NESTED_AOT=0`):

- proof summary: `v0211_mb_1h_gate/proof/pipeline_summary.json`
- W summary: `v0211_mb_1h_gate/proof/w_max_stats.json`
- verdict: `PIPELINE_PARTIAL`
- reason for partial: d02 output-count checker expected 3 files and observed 1;
  this was not a physics failure.
- finite status: `all_domains_finite=true`; d01 and d02 final states finite.

d01 physical `W` stayed bounded below 5 m/s:

```text
2024-08-06_12:20:06  max|W|=2.311203718185425 m/s  boundary max=0.8980915546417236 m/s
2024-08-06_12:40:12  max|W|=4.129059791564941 m/s  boundary max=1.0584869384765625 m/s
```

The maxima were interior (`edge_distance_cells` 33 and 32), so the old
specified-boundary corner blow-up was not present. Produced d02 `W` was finite;
max `|W|=8.035907745361328 m/s`.

Canary native-dt max-dom3, 1 h, fresh JIT (`GPUWRF_NESTED_AOT=0`):

- proof summary: `v0211_canary_1h_gate/proof/pipeline_summary.json`
- verdict: `PIPELINE_GREEN`
- `all_domains_finite=true`
- `all_outputs_present=true`
- per-domain counts: d01 2/2, d02 3/3, d03 3/3.

## Gap Analysis

Verdict: PASS for v0.21.1 release readiness, with the Mont-Blanc output-count
caveat above.

- Version consistency: `pyproject.toml` and `src/gpuwrf/__init__.py` are both
  `0.21.1`; this repository has no separate `VERSION` file.
- Docs accuracy: `CHANGELOG.md`, `RELEASE_NOTES.md`, and
  `RELEASE_NOTES_v0.21.1.md` describe the real boundary fix and the actual
  release-branch gates.
- Obsolete-text audit: the v0.21.1 docs contain only the boundary-fix narrative.
- No W masking shortcut: the staged source diff adds no `W` masking,
  `nan_to_num`, finite guard, value clip, or clamp.
- Scope: the curated `wrfgpu` distribution tree and dual README surfaces were
  not touched.
- PII: committed release docs and sanitized proof summaries avoid local home
  paths, GPU UUIDs, and emails. Raw run logs and generated `wrfout` files remain
  local and are not staged.
- Release operations: no push and no tag were performed.

## Proof Objects

- `proofs/v022/nest_dycore/v0211_mb_1h_gate/proof/pipeline_summary.json`
- `proofs/v022/nest_dycore/v0211_mb_1h_gate/proof/w_max_stats.json`
- `proofs/v022/nest_dycore/v0211_canary_1h_gate/proof/pipeline_summary.json`
- `proofs/v022/nest_dycore/V0211_RELEASE_INTEGRATION_REPORT.md`
