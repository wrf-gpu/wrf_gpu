# Sprint Contract: V0.14 Short Field H1 Residual Classification

Date: 2026-06-10 WEST
Owner: Fable medium/high in fresh tmux window
Manager: `worker/gpt/v013-close-manager`
Base commit: `41468af4`

## Objective

Classify and, if local/safe, close the current v0.14 Field-Parity release
blocker exposed by the 1h Canary d02 short falsifier.

Endpoint:

- **Proceed signal:** prove the residual is a run-provenance/native-init
  artifact or acceptable tolerance class, with exact evidence and the command
  for the long GPU gate; or
- **Fix signal:** implement a local, GPU-native, performance-compatible fix and
  prove it with a rerun/comparator; or
- **Block signal:** identify the exact remaining bug class, owned files,
  fastest next command, and whether it is stale runner/input root, Base-State or
  writer, native-init tolerance, physics/dycore kernel, or comparator/tolerance
  policy.

Do not return a vague hypothesis. The manager needs a yes/no decision on
whether 72h GPU field-parity gates can start.

## Current Evidence

- Short GPU falsifier root:
  `/mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T122005Z`.
- GPU output:
  `/mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T122005Z/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z`.
- CPU truth compared:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`.
- Comparator proof:
  `proofs/v014/short_field_falsifier_h1_grid_compare.{json,md}`.
- Comparator verdict is `REPORT_ONLY_NO_TOLERANCE_MANIFEST`, not release-green.
  Top h1 deltas: `PSFC` RMSE `323.115 Pa`, bias `-313.780 Pa`; `P` RMSE
  `129.754 Pa`; `MU` RMSE `121.961 Pa`; `PBLH` RMSE `78.950 m`; `HFX` RMSE
  `38.186 W/m2`; `LH` RMSE `53.896 W/m2`; `PB` p99 only `0.105 Pa` but local
  max `249.875 Pa`; `MUB` p99 `18.194 Pa`, local max `250.664 Pa`.
- Earlier accepted base/live-nest proofs fixed the large stale base-state class:
  `proofs/v014/live_nest_base_source_fix.md`,
  `proofs/v014/step1_transient_adjust_base_fix.md`, and
  `proofs/v014/step1_live_nest_theta_qv_wiring.md`.
- New manager observation: the short GPU run command used legacy runner
  `proofs/v0120/powered_tost_n15/run_one_case_v0120.py` and run root
  `/tmp/v0120_merged_run_root`. The selected case under that root symlinks
  `wrfinput_d01`, `wrfinput_d02`, `wrfbdy_d01`, and `namelist.input` to
  `/mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z`,
  while comparison used CPU truth under
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/...`. That makes
  stale/provenance mismatch a first-class hypothesis to prove or dismiss.
- Switzerland 72h CPU-WRF baseline is running separately under
  `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z`;
  do not interfere with it.

## Required Work

1. Verify git base and read the governing docs:
   `PROJECT_CONSTITUTION.md`, `AGENTS.md`,
   `.agent/skills/managing-sprints/SKILL.md`,
   `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`, and this contract.
2. Inspect the short-run provenance:
   runner command/log, `/tmp/v0120_merged_run_root` symlinks, CPU truth root,
   inputs, namelists, and available metadata. Determine whether GPU and CPU
   truth are same-case/same-input enough for a release gate.
3. Quantify the residual class using existing NetCDF outputs:
   compare CPU h0/h1, GPU h1, available inputs, and any matching WRF run roots.
   Localize whether `PB/MUB`, `MU/PSFC/P`, and surface flux/PBL residuals are
   static/input-root, writer/state export, native-init drift, physics, dycore, or
   comparator artifacts.
4. If provenance is wrong, propose the exact clean rerun command/root layout and
   optionally patch the runner/tooling if the bug is local and low risk.
5. If provenance is clean and a local code/tool bug is proven, fix it only if it
   is performance-compatible and file ownership is within this sprint.
6. Write a compact proof/report with a clear `PROCEED`, `FIXED`, or `BLOCKED`
   verdict for the manager.

## File Ownership

Allowed:

- `proofs/v014/short_field_h1_residual_classification.*`
- `scripts/compare_wrfout_grid.py` only for comparator provenance/tolerance
  metadata fixes, if proven necessary.
- `proofs/v0120/powered_tost_n15/run_one_case_v0120.py` only for low-risk
  run-root/provenance hygiene fixes, if proven necessary.
- Focused docs/review file:
  `.agent/reviews/2026-06-10-v014-short-field-h1-residual-fable.md`.

Allowed production code only if a small, exact, proven blocker is found:

- narrow writer/runtime input/output plumbing under `src/gpuwrf/runtime/` or
  `src/gpuwrf/io/`, with manager-readable justification.

Do not edit:

- Core dycore/physics kernels unless explicitly required by a proof and small
  enough to safely close in this sprint.
- Switzerland CPU run files, Canary CPU truth files, long GPU runners, FP32,
  memory lanes, or unrelated dirty files.

## Acceptance Criteria

- One compact proof object exists and validates as JSON if JSON is produced.
- Report includes:
  - verdict: `PROCEED`, `FIXED`, or `BLOCKED`;
  - exact bug class;
  - evidence table for `PSFC`, `P`, `MU`, `PB`, `MUB`, `T`, `U`, `V`, `QVAPOR`,
    `HFX`, `LH`, `PBLH`;
  - run-root/provenance finding;
  - next command for the manager.
- If code/tooling changed, run relevant py_compile/tests and `git diff --check`.
- If a clean short rerun is required, specify it exactly and whether it needs
  GPU lock; do not launch GPU without manager approval.

## Validation Commands

Minimum:

```bash
python -m json.tool proofs/v014/short_field_falsifier_h1_grid_compare.json >/tmp/short_h1_grid_compare.validated.json
git diff --check
```

If creating Python proof tooling:

```bash
python -m py_compile proofs/v014/short_field_h1_residual_classification.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/short_field_h1_residual_classification.py
python -m json.tool proofs/v014/short_field_h1_residual_classification.json \
  >/tmp/short_field_h1_residual_classification.validated.json
```

## Constraints

- CPU-only unless the manager grants a GPU lock.
- Keep report context-sparing.
- Preserve GPU-native model design; no clamps, CPU-WRF runtime dependency, or
  host/device transfers in timestep loops.
- Respect existing dirty worktree; do not stage/revert unrelated files.

## Handoff Requirements

Write:

- `.agent/reviews/2026-06-10-v014-short-field-h1-residual-fable.md`
- proof JSON/Markdown if produced
- worker-report style summary inside the review or separate `worker-report.md`

Completion marker to manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'FABLE SHORT_FIELD_H1_RESIDUAL DONE - see .agent/reviews/2026-06-10-v014-short-field-h1-residual-fable.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
