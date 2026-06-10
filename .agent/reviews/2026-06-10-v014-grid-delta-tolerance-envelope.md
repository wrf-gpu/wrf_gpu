# V0.14 Grid-Delta Tolerance Envelope Review

Date: 2026-06-10
Worker: GPT-5.5 xhigh

## Findings

1. No blocking scope violation found. I only added the three sprint-authorized proof/review artifacts and did not edit `src/gpuwrf/**`, TOST runners, Switzerland scripts, README, release notes, or Fable Step-1 files.
2. The candidate avoids post-hoc widening. The hard dynamic thresholds are the existing predeclared RMSE limits from `scripts/equivalence_demo.py`, `docs/equivalence-demo.md`, KI-9, and `proofs/v014/switzerland_validation_plan.md`.
3. `P`, `PH`, `MU`, and `RAINC` remain critical report-only fields. They are mandatory atlas inventory fields, but I found no frozen v0.14 all-cell threshold for them. Material drift should be reviewed as a release risk, not silently accepted.
4. Current atlas tooling reads tolerance metrics from `fields`, but it does not enforce manifest-declared mandatory presence fields. Final scoring should preserve the builder default mandatory core list or pass explicit `--mandatory-field` options if the tool changes.

## Objective

Produce a reviewable candidate tolerance envelope for the v0.14 Grid-Delta Atlas before final Grid-Delta, Switzerland, or TOST scoring.

## Files Changed

- `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`
- `proofs/v014/grid_delta_atlas/TOLERANCE_MANIFEST_CANDIDATE.md`
- `.agent/reviews/2026-06-10-v014-grid-delta-tolerance-envelope.md`

## Commands Run

- `sed`/`rg`/`jq`/small Python read-only inspections of the sprint contract, atlas gate, validation plan, method docs, tolerance ladder, atlas scripts, KI-9, and historical proof objects.
- `python -m json.tool proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json >/tmp/v014_grid_delta_tolerance_candidate.validated.json` - pass.
- `python scripts/build_grid_delta_atlas.py --help` - pass.
- `python scripts/compare_wrfout_grid.py --help` - pass.
- `git diff --check` - pass.
- `python` consistency check confirmed every static field listed in `static_exactness_groups.exact_copy_fields` has a parser-visible tolerance record; only `P`, `PH`, `MU`, and `RAINC` are report-only field records.

## Proof Objects Produced

- Machine-readable candidate manifest: `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`
- Human rationale: `proofs/v014/grid_delta_atlas/TOLERANCE_MANIFEST_CANDIDATE.md`
- This review: `.agent/reviews/2026-06-10-v014-grid-delta-tolerance-envelope.md`

## Recommended Manager Decision

`FREEZE_AFTER_REVIEW`

Freeze the ten existing hard dynamic RMSE limits and the static exact/tight checks after manager review. Do not promote `P`, `PH`, `MU`, `RAINC`, or other diagnostics from report-only without a separate pre-result threshold review.

## Risks

- Static exactness may fail stale or convention-mismatched writer artifacts. That is intentional for release scoring, but reviewer triage may need to distinguish writer convention from dynamics.
- Pooled RMSE can hide local drift; final atlas plots and p99/max/worst-cell records remain required.
- The manifest records mandatory fields, but current tools primarily enforce mandatory presence through script defaults, not the JSON.

## Next Decision Needed

Manager should decide whether to freeze this candidate as the v0.14 pre-result manifest, or request independent threshold work for `P`, `PH`, `MU`, and `RAINC` before final scoring.
