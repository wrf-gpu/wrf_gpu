# Sprint Contract

## Objective
Replace the stale v0.14 powered-TOST release gate with the principal-approved
field-parity/stability gate and start the CPU baseline path:

- Switzerland/Gotthard 72h CPU-WRF vs GPU-JAX field parity/stability.
- Canary 72h CPU-WRF vs GPU-JAX field parity/stability.
- Grid-Delta Atlas summary and plots as the primary release/paper artifact.
- Powered TOST retained only as secondary station sanity evidence.

## Non-Goals
- No source/kernel changes.
- No TOST marathon launch in this sprint.
- No claim that the old Switzerland 24h run is resumable unless `wrfrst_*` and
  72h boundary coverage are actually present.
- No fresh Canary CPU backfill if complete retained CPU-WRF truth already exists
  for the selected 72h d02 gate case.

## File Ownership
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`
- `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`
- `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
- `.agent/decisions/PAPER-STRATEGIC-FRAMING.md`
- `PROJECT_PLAN.md`
- sprint folder `.agent/sprints/2026-06-10-v014-field-parity-release-gate/`

## Inputs
- Principal decision, 2026-06-10: TOST gate is replaced by two 72h
  field-parity/stability gates.
- Existing Switzerland 24h CPU truth under
  `/mnt/data/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z`.
- Existing Canary L2 CPU-WRF 72h truth under
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output`.
- Resource monitor `scripts/monitor_resource_usage.sh`.

## Acceptance Criteria
- Roadmap/checklist/paper-gate docs state the new v0.14 release gate clearly.
- Switzerland 24h resume feasibility is checked and recorded.
- Canary d02-vs-d03 choice is evidence-based and recorded.
- If feasible, start or stage the Switzerland 72h CPU baseline with memory
  tracking.
- Running GPU work is not collided with a long CPU-WRF launch on the same cores.

## Validation Commands
- `git diff --check`
- `python scripts/close_sprint.py .agent/sprints/2026-06-10-v014-field-parity-release-gate`
- Operational checks of current GPU/CPU processes before launch.

## Performance Metrics
- CPU baseline resource CSVs:
  `*_process_usage.csv`, `*_system_memory.csv`, and no-GPU flag.
- Later GPU gates must include `*_gpu_usage.csv`, `*_process_usage.csv`, and
  `*_system_memory.csv`.

## Proof Object
- `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
- CPU-baseline run root under `/mnt/data/wrf_gpu_validation/` if launched.

## Risks
- Switzerland 72h needs new GFS/WPS/real output because the 24h run has no
  restart files and only 24h boundary coverage.
- A fresh CPU-WRF run should not fight the current short GPU falsifier for cores.
- Canary d03 is attractive scientifically, but retained 72h CPU truth appears to
  be d02; d03 is currently 24h-retained only.

## Handoff Requirements
- State exact run roots, PIDs, logs, rc files, and CSV paths for any launched
  baseline.
- State whether TOST is still running, paused, or only optional.
