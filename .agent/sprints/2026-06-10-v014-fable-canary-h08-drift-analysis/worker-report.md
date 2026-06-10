# Worker Report

Summary: Fable high diagnosed the Canary h08/h10/h18/h24 field drift as a
root-domain wrfbdy lateral-boundary cadence bug, implemented a narrow
root-only fix, and produced a CPU-only proof object.

## Objective

Classify the first Canary d02 long-gate drift and return a manager-actionable
stop/continue/fix decision without using the GPU.

## Result

Verdict: `LBC_CADENCE_ROOT_CAUSE_PROVEN_FIX_GATE_PASS`.

The standalone root d01 was walking 6-hourly `wrfbdy_d01` boundary leaves at a
3600 s replay cadence. That consumed the forcing 6x too fast and then froze at
the final wrfbdy record. The patch sets the root boundary cadence from
`boundary_meta["interval_seconds"]` and synthesizes the terminal wrfbdy leaf
level from the last `_BT*` tendency.

## Files Changed

- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/lbc_cadence_root_cause.py`
- `proofs/v014/lbc_cadence_root_cause.json`
- `proofs/v014/lbc_cadence_root_cause.md`
- `.agent/reviews/2026-06-10-v014-fable-canary-h08-drift-analysis.md`

## Proof Objects

- `proofs/v014/lbc_cadence_root_cause.md`
- `proofs/v014/lbc_cadence_root_cause.json`
- pre-fix diagnostic run:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_20260610T142426Z`

## Remaining Risk

The sprint does not fix the secondary quasi-static GPU PSFC vapor-light floor
near `-210 Pa`; that becomes a separate dycore lane if the fixed Canary run
confirms it.
