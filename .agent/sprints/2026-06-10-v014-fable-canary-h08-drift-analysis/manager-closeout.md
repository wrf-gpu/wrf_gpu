# Manager Closeout: V0.14 Fable Canary h08 Field-Drift Analysis

Date: 2026-06-10
Merge Decision: `ACCEPTED_FIX_AND_PROOF`

## Outcome

Fable found and fixed the Canary h08/h10/h18/h24 field drift root cause. The
pre-fix Canary 72h GPU run was intentionally stopped at h26 with `gpu_rc=143`
because it used wrong-time lateral-boundary forcing and could not provide
release evidence.

## Root Cause

The standalone root d01 consumed 6-hourly `wrfbdy_d01` boundary leaves at the
hourly replay cadence (`update_cadence_s=3600` instead of
`interval_seconds=21600`), then clamped/froze at the final wrfbdy record. This
created the domain-wide `PSFC/MU/P/PH` drift. Switzerland would have hit the
same bug at 3x-fast forcing because its boundary interval is 10800 s.

## Accepted Files

- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/lbc_cadence_root_cause.py`
- `proofs/v014/lbc_cadence_root_cause.json`
- `proofs/v014/lbc_cadence_root_cause.md`
- `.agent/reviews/2026-06-10-v014-fable-canary-h08-drift-analysis.md`

## Manager Rerun Gates

- `python -m py_compile src/gpuwrf/integration/d02_replay.py src/gpuwrf/integration/nested_pipeline.py proofs/v014/lbc_cadence_root_cause.py`:
  pass
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 0-3 python proofs/v014/lbc_cadence_root_cause.py`:
  `LBC_CADENCE_ROOT_CAUSE_PROVEN_FIX_GATE_PASS`
- `python -m json.tool proofs/v014/lbc_cadence_root_cause.json`: pass
- `git diff --check`: pass
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 0-7 python -m pytest tests/test_m6_boundary_apply.py tests/test_v013_tost_wrfbdy_fix.py tests/test_p0_1a_nesting.py tests/test_gwd_operational_wiring.py -q`:
  `23 passed, 1 skipped`

## Follow-Up

Commit and push the fix, clear the stale GPU lock, then relaunch the Canary
L2 d02 72h GPU field gate from the fixed commit. Do not launch Switzerland GPU
until the fix is merged. If the fixed Canary run leaves a flat `PSFC` floor near
`-210 Pa`, open a separate dycore moist-pressure sprint; that is not the LBC
cadence bug.
