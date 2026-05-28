# Worker Report - M9.A Trace Harness

## Verdict

`M9A_PARTIAL`

The diagnostic scripts and proof-object paths were created, but the sprint cannot close as complete in this environment:

- The WRF Fortran operational trace is absent from the searched Gen2/savepoint locations.
- The required 1000-step GPU run could not execute because no JAX GPU backend is visible (`nvidia-smi` cannot communicate with the driver; JAX reports only `CpuDevice(id=0)`).

## Objective

Build and run:

- `scripts/operational_trace_compare.py`
- `scripts/m6b6_coupled_step_compare_1000.py`
- `proofs/m9/operational_trace_360steps.json`
- `proofs/m9/savepoint_parity_1000.json`

No `src/**` model code was modified.

## Files Changed

- `scripts/operational_trace_compare.py`
- `scripts/m6b6_coupled_step_compare_1000.py`
- `proofs/m9/operational_trace_360steps.json`
- `proofs/m9/savepoint_parity_1000.json`
- `.agent/sprints/2026-05-28-m9a-trace-harness/worker-report.md`

## Commands Run

- `taskset -c 0-3 python -m py_compile scripts/operational_trace_compare.py scripts/m6b6_coupled_step_compare_1000.py`
- `taskset -c 0-3 pytest -q tests/test_m6b6_coupled_step_parity.py` -> `5 passed in 30.53s`
- `taskset -c 0-3 python scripts/operational_trace_compare.py --case 20260521 --horizon-steps 360 --output proofs/m9/operational_trace_360steps.json`
- `taskset -c 0-3 nvidia-smi -L` -> failed to communicate with NVIDIA driver
- `taskset -c 0-3 python -c 'import jax; print(jax.default_backend()); print(jax.devices())'` -> backend `cpu`, devices `[CpuDevice(id=0)]`
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare_1000.py --output proofs/m9/savepoint_parity_1000.json`
- `taskset -c 0-3 python -m json.tool proofs/m9/operational_trace_360steps.json`
- `taskset -c 0-3 python -m json.tool proofs/m9/savepoint_parity_1000.json`
- `taskset -c 0-3 git add scripts/operational_trace_compare.py scripts/m6b6_coupled_step_compare_1000.py proofs/m9/operational_trace_360steps.json proofs/m9/savepoint_parity_1000.json .agent/sprints/2026-05-28-m9a-trace-harness/worker-report.md` -> failed: git worktree metadata is read-only in this sandbox

## Proof Objects Produced

- `proofs/m9/operational_trace_360steps.json`
  - Status: `M9A_PARTIAL_MISSING_WRF_REFERENCE_TRACE`
  - Operators: empty because no independent WRF trace was available.
  - Searched the canonical 20260521 Gen2 run trace/savepoint locations and sprint-local `wrf_reference_trace`.
  - Includes exact generation instructions for the missing WRF trace.

- `proofs/m9/savepoint_parity_1000.json`
  - Status: `FAIL`
  - Depth: `1000`
  - Run not executed because GPU is unavailable.
  - Device evidence: `default_backend=cpu`, `gpu_devices=[]`, `visible_devices=["cpu:0"]`.

## Unresolved Risks

- No first-divergence operator can be identified until the WRF Fortran operational trace exists.
- No 1000-step dycore ratchet result can be claimed until this is rerun on a GPU-visible worker.
- The operational trace harness currently supports trace JSON or `.npz` operator arrays, but this was not exercised against a real Fortran trace because none was present.
- The requested git commit was not possible from this sandbox because `/home/enric/src/wrf_gpu2/.git/worktrees/wrf_gpu2_m9a/index.lock` cannot be created on a read-only filesystem.

## Next Decision Needed

Provide or generate the WRF Fortran operational trace for case `20260521`, then rerun this sprint on a worker with one visible GPU.
