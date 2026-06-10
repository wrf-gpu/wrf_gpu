# Sprint Contract: V0.14 Fable Canary h08 Field-Drift Analysis

Date: 2026-06-10
Owner: manager
Assignee: Fable high, fresh tmux worker
Status: CLOSED_ACCEPTED

## Objective

Analyze the first v0.14 Canary L2 d02 72h field-gate intermediate result. The
active GPU run is still running, but the CPU-only h08 grid comparator already
reports `verdict: FAIL` with growing dynamic pressure/mass residuals. Determine
whether this is:

- a known non-blocking manifest/static-boundary artifact;
- a validation/comparator/tolerance-manifest issue;
- a writer/base-state/output-semantics issue;
- a native-init/live-nest/boundary/root-input issue;
- a real dynamics/physics kernel bug;
- or an expected bounded free-run divergence that needs a different v0.14 gate.

Produce a manager-actionable diagnosis and next action. If a local, low-risk fix
is obvious, propose it precisely; do not make broad source edits unless the fix
is both small and proven with CPU-only evidence.

## Key Facts

- Current branch: `worker/gpt/v013-close-manager`, latest pushed commit
  `c42fc7ba` (`v014 unblock field gates after theta EOS fix`).
- Active GPU run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_20260610T142426Z`
- Selected Canary case:
  `20260501_18z_l2_72h_20260519T173026Z`
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- GPU output dir:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_20260610T142426Z/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- h08 comparator:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_20260610T142426Z/canary_d02_h08_intermediate_grid_compare.md`
  and `.json`
- h08 top signal:
  - `verdict: FAIL`
  - paired files: 8
  - `PSFC` RMSE slope/h `73.614`, worst lead RMSE `644.780`
  - `MU` RMSE `243.067`, worst lead RMSE `390.396`
  - `P` RMSE `191.523`, worst lead RMSE `284.558`
  - `PH` RMSE `154.005`, worst lead RMSE `234.500`
  - `PB/MUB` static max spikes remain boundary-frame-like but are not the main
    concern; dynamic PSFC/MU/P/PH growth is.
- Post-EOS h1 adjudication had allowed launch but warned PSFC/MU/P/PH were
  report-only risks:
  `proofs/v014/post_eos_h1_residual_adjudication.md`

## Inputs To Read First

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`
- `proofs/v014/post_eos_h1_residual_adjudication.md`
- `proofs/v014/eos_theta_semantics.md`
- h08 compare markdown and JSON under the run root above
- relevant `scripts/compare_wrfout_grid.py`
- relevant source only after looking at the evidence.

## Constraints

- Do not use GPU; the active Canary run owns it.
- Prefer CPU-only NetCDF/statistical analysis of existing h1-h8 outputs.
- Keep context use low: summarize tables, do not paste huge JSON.
- Do not modify stable roadmap/skills/manager files.
- Source edits are allowed only if a small, local, strongly evidenced fix is
  found; otherwise write analysis only.

## Required Output

Write:

`.agent/reviews/2026-06-10-v014-fable-canary-h08-drift-analysis.md`

Required structure:

1. Verdict paragraph: BLOCK / PROCEED / NEED_MORE_DATA.
2. Root-cause ranking table, max 8 rows: hypothesis, evidence for, evidence
   against, next falsifier, expected wall-clock.
3. Field-drift summary: PSFC/MU/P/PH/PB/MUB/T/U/V/QVAPOR/T2/U10/V10 over h1-h8.
4. Decision: should manager stop the 72h GPU run now, let it continue to h24,
   or let it complete for data?
5. Exact next commands or proof scripts the manager should run.
6. If a fix is proposed: changed files, why it is WRF-faithful, proof gate.
7. Context-sparing handoff, max 10 bullets.

Completion marker:

`FABLE CANARY_H08_DRIFT_ANALYSIS DONE - see .agent/reviews/2026-06-10-v014-fable-canary-h08-drift-analysis.md`
