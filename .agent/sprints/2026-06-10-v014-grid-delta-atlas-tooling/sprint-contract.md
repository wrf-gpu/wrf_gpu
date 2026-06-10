# Sprint Contract: V0.14 Grid-Delta Atlas Tooling

Date: 2026-06-10 02:47 WEST
Owner: GPT-5.5 xhigh worker in tmux, isolated worktree
Manager: `worker/gpt/v013-close-manager`
Branch: `worker/gpt/v014-grid-delta-atlas`
Worktree: `/home/enric/src/wrf_gpu2/.codex/worktrees/v014-grid-delta-atlas`

## Objective

Build the offline Grid-Delta Atlas tooling required by the v0.14 validation
gate, without running the long GPU/TOST campaign.

Endpoint: a tested command-line tool that compares paired CPU-WRF and GPU
wrfout outputs across all common numeric fields and emits compact machine and
human artifacts:

- `proofs/v014/grid_delta_atlas/manifest.json`
- `proofs/v014/grid_delta_atlas/grid_delta_summary.json`
- `proofs/v014/grid_delta_atlas/GRID_DELTA_ATLAS.md`
- selected deterministic plots under `docs/assets/v014/grid_delta_atlas/`

This sprint prepares the gate. It does not claim v0.14 equivalence because the
real validation data are not ready until grid parity closes.

## Accepted Requirements

Read:

- `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `docs/GPU_RUNBOOK.md`
- existing `scripts/compare_wrfout_grid.py` if present.

The atlas must:

- pair CPU and GPU files by domain, valid time, and lead where possible;
- compare every common numeric field;
- record missing/non-numeric fields explicitly;
- handle exact-shape and WRF staggered fields without silent crop;
- compute count, finite counts, max_abs, RMSE, MAE, bias, p50/p95/p99/p99.9,
  safe relative metrics where meaningful, correlation, and worst index/value;
- compute per-field lead-time drift summaries;
- generate deterministic compact plots and a README-ready dashboard image;
- keep top-level console output short.

## File Ownership

Allowed files on this isolated branch:

- new script(s), preferably `scripts/build_grid_delta_atlas.py`;
- focused tests under `tests/`, preferably synthetic tiny NetCDF/NPZ fixtures;
- `docs/assets/v014/grid_delta_atlas/.gitkeep` or generated tiny smoke plots;
- proof docs under `proofs/v014/grid_delta_atlas/`;
- `.agent/reviews/2026-06-10-v014-grid-delta-atlas-tooling.md`;
- sprint closeout files in this sprint folder.

Do not edit model kernels, coupling/runtime source, memory/FP32 code, TOST
runner semantics, Switzerland case generation, README release claims, or
`docs/KNOWN_ISSUES.md` in this sprint. Those wait for real validation results.

## Acceptance Criteria

- The tool runs on synthetic paired data in CI-style tests without real GPU/WRF
  corpus.
- The generated summary and manifest schemas are stable and compact.
- Missing fields, nonfinite values, and shape mismatches are explicit hard/error
  or inventory records; no silent exclusions.
- Plots are deterministic and small; if matplotlib is unavailable, fail with a
  clear message or degrade to JSON/Markdown while tests cover installed deps.
- The tool can be invoked later by TOST/Switzerland validation without changing
  model code.

## Validation Commands

Required, manager-rerunnable:

```bash
python -m py_compile scripts/build_grid_delta_atlas.py
PYTHONPATH=src pytest -q tests/test_grid_delta_atlas.py
python scripts/build_grid_delta_atlas.py --help
git diff --check
```

If you add more tests, list them in the report.

## Proof Object

Primary proof object for this tooling sprint:

- `proofs/v014/grid_delta_atlas/GRID_DELTA_ATLAS_TOOLING.md`

It must state:

- command syntax;
- synthetic test coverage;
- what real campaign inputs are still required;
- what fields/metrics/plots are emitted;
- limitations and any dependency assumptions.

## Handoff Requirements

Commit your finished work on branch `worker/gpt/v014-grid-delta-atlas` and push
it. Write a concise worker report and review summary.

Completion marker:

`GPT GRID_DELTA_ATLAS_TOOLING DONE - see proofs/v014/grid_delta_atlas/GRID_DELTA_ATLAS_TOOLING.md`

Notify manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT GRID_DELTA_ATLAS_TOOLING DONE - see proofs/v014/grid_delta_atlas/GRID_DELTA_ATLAS_TOOLING.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
