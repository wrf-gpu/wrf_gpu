# Sprint Contract: V0.14 Grid-Delta Tolerance Envelope Candidate

Date: 2026-06-10 WEST
Owner: GPT-5.5 xhigh in tmux
Manager: `worker/gpt/v013-close-manager`

## Objective

Produce a reviewable candidate tolerance envelope for the v0.14 Grid-Delta
Atlas. The goal is to prevent post-hoc tolerance tuning after TOST/Switzerland
runs. This sprint is analytical and documentation/proof-only: no model source
changes.

## Context

The v0.14 final validation gate requires:

- station TOST; and
- Grid-Delta Atlas over paired CPU-WRF/GPU wrfout files.

`scripts/build_grid_delta_atlas.py` and `scripts/compare_wrfout_grid.py` already
support `--tolerance-json`, but the final v0.14 tolerance manifest is not frozen.
Without a manifest, the tools correctly emit
`REPORT_ONLY_NO_TOLERANCE_MANIFEST`.

## Required Work

1. Read:
   - `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`
   - `.agent/decisions/V0140-VALIDATION-PLAN.md` B4/B7 sections
   - `proofs/v014/grid_comparison_method.md`
   - `proofs/v014/switzerland_validation_plan.md`
   - `src/gpuwrf/validation/tolerance_ladder.json`
   - `scripts/build_grid_delta_atlas.py`
   - `scripts/compare_wrfout_grid.py`
   - historic equivalence proof summaries, especially
     `docs/KNOWN_ISSUES.md` KI-9 and `proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`
2. Propose a candidate machine-readable tolerance manifest for v0.14 Atlas
   scoring. It must separate:
   - hard release-gate core fields;
   - report-only fields;
   - static/exactness fields;
   - Switzerland-specific vs Canary/TOST-specific envelopes if needed.
3. For every hard field, include units, metric thresholds, rationale, and risk.
   Prefer pre-existing documented thresholds where they exist; do not invent
   permissive thresholds merely to pass old red runs.
4. Identify fields that should remain report-only until independent review
   because their tolerances would be speculative.
5. Run only cheap offline validation/syntax checks. If a tiny existing wrfout
   smoke is useful, write outputs to `/tmp`; do not start GPU or model runs.

## Allowed Files

May write:

- `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`
- `proofs/v014/grid_delta_atlas/TOLERANCE_MANIFEST_CANDIDATE.md`
- `.agent/reviews/2026-06-10-v014-grid-delta-tolerance-envelope.md`

Do not edit `src/gpuwrf/**`, TOST runners, Switzerland scripts, README, release
notes, or active Fable Step-1 files.

## Gates

Required:

```bash
python -m json.tool proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json >/tmp/v014_grid_delta_tolerance_candidate.validated.json
python scripts/build_grid_delta_atlas.py --help
python scripts/compare_wrfout_grid.py --help
git diff --check
```

Optional smoke, only if existing paired wrfout files are available and it stays
offline:

```bash
python scripts/build_grid_delta_atlas.py \
  --cpu-dir <existing_cpu_case> \
  --gpu-dir <existing_gpu_case> \
  --case-id <case> \
  --domain d02 \
  --min-lead 1 --max-lead 1 \
  --field T2 --field U10 --field V10 \
  --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
  --proof-dir /tmp/v014_grid_delta_tolerance_candidate_smoke \
  --asset-dir /tmp/v014_grid_delta_tolerance_candidate_assets \
  --no-plots
```

## Handoff Requirements

Write a concise review with:

- objective
- files changed
- commands run
- proof objects produced
- recommended manager decision: `FREEZE_AFTER_REVIEW`, `NEEDS_REVIEW`, or
  `REJECT`
- risks and fields requiring independent review
- next decision needed

Completion marker:

```bash
tmux send-keys -t 0:2 'GPT GRID_DELTA_TOLERANCE_ENVELOPE DONE - see proofs/v014/grid_delta_atlas/TOLERANCE_MANIFEST_CANDIDATE.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
