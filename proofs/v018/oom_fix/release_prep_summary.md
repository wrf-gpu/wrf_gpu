# v0.18.2 release-prep summary

## Objective

Prepare the v0.18.2 release surfaces for the bit-identical AC1_FIT 1 km nested
VRAM-efficiency fix. This pass made documentation/version edits only; it did not
commit, push, tag, touch any remote, run GPU jobs, or take the GPU lock.

## Files changed and exact claims added

- `pyproject.toml`
  - Bumped package version from `0.18.1` to `0.18.2`.

- `src/gpuwrf/__init__.py`
  - Bumped `__version__` from `0.18.1` to `0.18.2`.

- `CHANGELOG.md`
  - Added top entry `## [0.18.2] — 1 km nested VRAM-efficiency fix (bit-identical)`.
  - Claims added:
    - v0.18.2 is a memory-efficiency patch over 0.18.1.
    - Default numerics are unchanged and validated outputs are bit-identical on
      the default path.
    - AC1_FIT 9/3/1 nested 1 km all-island case now fits one RTX 5090.
    - Prior path OOMed on the 32 GiB card; fixed warm-cache 1-forecast-hour run
      passes at 18.1 GiB peak VRAM.
    - RRTMG radiation column-tile defaults changed 16384→2048 and MYNN
      cold-start BouLac initialization is tiled.
    - 26/26 `wrfout` fields compare exact with `max_abs_diff=0.0`; MYNN
      cold-start `qke`/`pblh` diffs are 0.0.
    - Warm steady-state nested 1 km GPU utilization is ~85–88%; the lower
      full-run aggregate is the one-time load/cold-JIT prefix, not steady-state
      idleness.
    - `data/fixtures` runtime tables for Thompson aerosol and cold collection
      are restored so Thompson cases import from a source-only tree.

- `RELEASE_NOTES_v0.18.2.md`
  - New release notes file following the v0.15.0 decision-oriented style.
  - Claims added:
    - v0.18.2 is a memory-efficiency patch, not a speed release.
    - MEASURED scope: AC1_FIT 9/3/1 nested, d03 = 520x280x45 (~145k columns),
      fp64, mp8 Thompson / MYNN / Noah-MP / RRTMG, reference RTX 5090 (32 GiB).
    - BEFORE: OOM near 31.8/32 GiB, failed 12.72 GiB contiguous-arena request
      plus recurring 2.09 GiB allocations.
    - AFTER: warm-cache 1-forecast-hour run PASS, all 3 domains ended on the
      radiation gate, all finite, 18.1 GiB peak VRAM.
    - 18.1 GiB is cross-confirmed by worker 18.34 GiB and manager demo
      18.10 GiB, leaving roughly 14 GiB headroom.
    - Root cause was transient radiation working memory plus a one-time MYNN
      cold-start dense-BouLac buffer, not persistent State (~2.5 GiB).
    - RRTMG longwave/shortwave column-tile defaults changed 16384 -> 2048; MYNN
      cold-start dense BouLac initialization is tiled over production MYNN column
      width; d03-scale optional `jnp.max` host-sync metadata reductions are gated.
    - Default numerics unchanged: 26/26 `wrfout` fields exact,
      `max_abs_diff=0.0`; MYNN cold-start `qke` diff 0.0 and `pblh` diff 0.0.
    - Switzerland d01 remains one radiation tile (1764 columns); forecast-only
      wall 24.15 s old default vs 23.94 s new default.
    - Warm steady-state utilization: ~88% mean, ~85% samples >=80%, ~6% idle,
      p95 100%; full-run aggregate ~66% due one-time ~81 s domain-load +
      cold-JIT prefix.
    - Warm forecast-only speed is ~1734 s/forecast-hour; compute-bound once warm;
      no new single-card speedup claim.
    - Thompson aerosol and cold-collection runtime tables are restored.
    - PROJECTED label is reserved for extrapolation to other GPUs, longer runs,
      other nested geometries, or multi-GPU throughput.

- `README.md`
  - Added Performance bullet for v0.18.2:
    - AC1_FIT 9/3/1 nested d03 520x280x45 (~145k columns) previously OOMed on
      the 32 GiB reference card.
    - It now passes a warm-cache 1-forecast-hour run at 18.1 GiB peak VRAM via
      radiation column tiling and tiled MYNN cold-start.
    - Warm steady-state utilization is ~85–88%; low-utilization window is the
      one-time domain-load/cold-JIT prefix.
    - This is a VRAM-fit fix, not a new single-card speedup claim.
  - Updated System requirements & resource profile VRAM row:
    - Headline is the 1 km-NESTED all-island AC1_FIT case.
    - Now fits reference RTX 5090 at ~18.1 GiB peak; before v0.18.2 it OOMed
      near 31.8/32 GiB.
    - Retained 72 h gate peaks: 22.9 GiB Switzerland d01 and 29.8 GiB Canary L2
      d02 nested.
    - Retained d01 9 km standalone peak ≈4.7 GiB and 1 km single-domain fresh
      process peak 18.25 GiB.
    - Peak is transient working memory; v0.18.2 levers are bit-identical
      radiation/cold-start column tiling; multi-GPU remains the scale path.
  - Added v0.18.2 version-history row:
    - OOM near 31.8/32 GiB -> 18.1 GiB peak.
    - 16384→2048 radiation column-tile defaults plus tiled MYNN cold-start.
    - 26/26 `wrfout` fields exact; MYNN cold-start `qke`/`pblh` diffs 0.0.
    - Warm steady-state utilization ~85–88%; full-run aggregate lower due
      one-time load/cold-JIT prefix.
    - Thompson aero+cold runtime fixture tables restored.
    - Links `proofs/v018/oom_fix/fix_report.md` and `RELEASE_NOTES_v0.18.2.md`.

- `docs/resource-profile.md`
  - Updated Memory (VRAM) section:
    - MEASURED v0.18.2 AC1_FIT 9/3/1 nested 1 km all-island case now fits one
      reference RTX 5090 at ~18.1 GiB peak VRAM.
    - Same path previously OOMed near 31.8/32 GiB and failed 12.72 GiB plus
      recurring 2.09 GiB allocations.
    - Realized algorithmic lever is bit-identical RRTMG radiation column-tile
      defaults 16384→2048 plus tiled MYNN cold-start BouLac initialization.
    - Peak is transient working memory, not persistent State (~2.5 GiB).
    - Retained 72 h gate peaks: 22.9 GiB Switzerland d01 and 29.8 GiB Canary L2
      d02 nested.
    - Retained d01 9 km standalone peak ≈4.7 GiB and 1 km single-domain fresh
      process peak 18.25 GiB.
  - Updated quick sizing checklist to distinguish the measured 18.1 GiB
    1 km-NESTED AC1_FIT path from the 29.8 GiB retained Canary L2 d02 72 h gate.

- `docs/PERFORMANCE.md`
  - Added v0.18.2 nested 1 km utilization addendum.
  - Claims added:
    - v0.18.2 is a VRAM-fit release, not a speed release.
    - AC1_FIT 9/3/1 nested d03 520x280x45 (~145k columns), fp64, mp8 Thompson /
      MYNN / Noah-MP / RRTMG fits at 18.1 GiB peak VRAM on the reference RTX 5090.
    - Warm steady-state utilization is ~85–88%, p95 100%, idle about 6%.
    - Full-run aggregate is ~66% when including the one-time ~81 s domain-load +
      cold JIT compile prefix.
    - Low-utilization window is cold compilation/setup, not steady-state forecast
      idleness.
    - Warm forecast-only speed is ~1734 s/forecast-hour; no new single-card
      speedup claim.

- `KNOWN_ISSUES.md`
  - Updated title to v0.18.2.
  - Added `Resolved in v0.18.2` note for the 1 km nested all-island VRAM/OOM path.
  - Claims added:
    - Prior AC1_FIT 9/3/1 nested 1 km path OOMed near 31.8/32 GiB.
    - It failed 12.72 GiB contiguous-arena request plus recurring 2.09 GiB
      allocations.
    - v0.18.2 resolves it with bit-identical RRTMG radiation column tiling
      16384→2048 plus tiled MYNN cold-start BouLac initialization.
    - Warm-cache 1-forecast-hour run PASS, 18.1 GiB peak VRAM, 26/26 `wrfout`
      fields exact with `max_abs_diff=0.0`.
    - Proof link: `proofs/v018/oom_fix/fix_report.md`.

- `proofs/v018/oom_fix/release_prep_summary.md`
  - This proof summary, listing changed files, exact claims, commands, proof
    objects, risks, and next decision.

- `proofs/v018/oom_fix/RELEASE_PREP_DONE`
  - Final empty sentinel file; no claims. It is touched as the last action after
    validation.

## Commands run

- `sed -n '1,240p' /tmp/v0182_release_prep_brief.txt`
- `sed -n '1,260p' PROJECT_CONSTITUTION.md`
- `sed -n '1,260p' AGENTS.md`
- `find .agent/skills -maxdepth 3 -type f | sort`
- `sed -n '1,260p' .agent/skills/updating-docs-minimally/SKILL.md`
- `sed -n '1,220p' .agent/skills/reporting-to-human/SKILL.md`
- `sed -n '1,320p' RELEASE_NOTES_v0.15.0.md`
- `sed -n '1,260p' CHANGELOG.md`
- `sed -n '1,260p' README.md`
- `sed -n '290,420p' README.md`
- `sed -n '500,670p' README.md`
- `sed -n '1,260p' docs/resource-profile.md`
- `sed -n '1,280p' docs/PERFORMANCE.md`
- `sed -n '1,260p' KNOWN_ISSUES.md`
- `sed -n '1,240p' proofs/v018/oom_fix/fix_report.md`
- `rg -n "version|0\\.18\\.1|__version__" pyproject.toml src/gpuwrf/__init__.py`
- `python -m py_compile src/gpuwrf/__init__.py`
- `JAX_PLATFORM_NAME=cpu PYTHONPATH=src python -c "import tomllib; data=tomllib.load(open('pyproject.toml','rb')); assert data['project']['version']=='0.18.2'; import gpuwrf; assert gpuwrf.__version__=='0.18.2'; print('version ok')"`
- Public release-file scan for local home paths and the local username pattern.
- `git diff --check`

## Proof objects produced

- `RELEASE_NOTES_v0.18.2.md`
- `proofs/v018/oom_fix/release_prep_summary.md`
- `proofs/v018/oom_fix/RELEASE_PREP_DONE` (final sentinel, touched last)

## Unresolved risks

- No new GPU measurement was run in this pass by contract. The release claims rely
  on the existing measured proof artifacts in `proofs/v018/oom_fix/` and
  `proofs/v018/oom_rootcause/`.
- Pre-existing dirty working-tree changes from the OOM fix were left intact and
  not reverted.

## Next decision needed

Manager review of the diff, then manager-owned commit/tag/public-push flow if the
release-prep diff is accepted.
