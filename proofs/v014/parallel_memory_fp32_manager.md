# v0.14 Parallel Memory/FP32 Manager

- Verdict: `MERGE_NOW`
- Branch: `worker/gpt/v014-memory-fp32-manager`
- HEAD: `cfb3ad6c6bb8`
- CPU-only: `True`
- GPU used: `False`

## Collision Map

- Implemented source edit: `src/gpuwrf/coupling/scan_adapters.py` only.
- Locked paths avoided: `src/gpuwrf/dynamics/**`, `runtime/operational_mode.py`, `integration/d02_replay.py`, `nesting/**`, boundary/carry/init/restart/state-contract files.
- Moisture velocity reuse, acoustic carry split, limiter workspace reduction, state aliasing, and all FP32 source work collide with the active fp64 grid-parity lock.

## Memory Fix

- WDM6 `slmsk` old shape: `[12, 10]`.
- WDM6 `slmsk` new shape: `[12]`.
- Exact old-layout vs new-layout State leaves: `True`.
- Target 641x321x50 fp64 transient saving: `76.922 MiB` (`0.075119 GiB`).
- Semantics note: this preserves the existing 1.0 land / 0.0 water adapter values; it is not a land-sea physics correction.

## FP32 Status

- Absolute-total fp32 1 mPa recovered delta: `0.0` Pa.
- Perturbation-form fp32 1 mPa recovered delta: `0.00099945068359375` Pa.
- One-column pressure error ratio absolute-total32 / perturbation32: `1.361e+06`.
- One-column geopotential error ratio absolute-total32 / perturbation32: `1.841e+06`.
- Source verdict: `FP32_SOURCE_WORK_INFEASIBLE_WITH_CURRENT_LOCKS`.

## Next Gate

Merge only the WDM6 shape-only cleanup after validation. Reopen FP32 R0/R1 only after the primary manager releases the live-nest P/MU/W perturbation-state initialization lock.
