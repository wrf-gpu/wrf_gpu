# V0.14 Parallel Memory/FP32 Manager Review

Date: 2026-06-09
Branch: `worker/gpt/v014-memory-fp32-manager`
Manager: GPT-5.5 xhigh side manager

## Verdict

Recommendation: `MERGE_NOW`

The branch implements one non-conflicting bit-identical memory cleanup: WDM6
`slmsk` now passes one scalar mask per column instead of materializing a
full-column `(ncol, nlev)` broadcast. The source edit is limited to
`src/gpuwrf/coupling/scan_adapters.py`.

FP32 acoustic source work is not currently implementable in this lane without
colliding with the primary fp64 grid-parity debug. The proof bundle refreshes
the CPU mechanism evidence and records exact blocked source surfaces.

## Collision Map

Locked production paths avoided:

- `src/gpuwrf/dynamics/**`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/nesting/**`
- `src/gpuwrf/boundary*`
- `src/gpuwrf/contracts/state.py`
- live-nest/base-state/init/restart/boundary/carry files

Implemented:

- `wdm6_slmsk_shape_only_cleanup`: `src/gpuwrf/coupling/scan_adapters.py`

Blocked by active locks:

- moisture transport velocity reuse: `runtime/operational_mode.py`,
  `dynamics/flux_advection.py`
- acoustic carry split / pad cleanup: dycore plus runtime
- moisture limiter workspace reduction: dycore plus runtime
- state total/perturbation/base alias reduction: `contracts/state.py` and
  init/restart/wrfout/boundary compatibility paths
- FP32 R0/R1/R2 source work: runtime, dycore, boundary/restart/init/carry
  surfaces overlapping active live-nest `P/MU/W` initialization debug

## Proofs

- `proofs/v014/parallel_memory_fp32_manager.py`
- `proofs/v014/parallel_memory_fp32_manager.json`
- `proofs/v014/parallel_memory_fp32_manager.md`
- refreshed `proofs/v014/exact_branch_memory_preflight.{json,md}`

Key proof results:

- WDM6 old full-column `slmsk` layout vs new per-column layout is exact across
  all `State` leaves on the CPU proof case.
- Target 641x321x50 fp64 transient saving is `76.92176055908203 MiB`
  (`0.07511890679597855 GiB`).
- The cleanup preserves the existing adapter values: `1.0` land and `0.0`
  water. It intentionally does not attempt a WRF `1/2` land-sea semantic fix.
- FP32 absolute-total storage still drops a 1 mPa pressure update at `90100 Pa`;
  perturbation-form fp32 recovers `0.00099945068359375 Pa`.
- One-column absolute-total32 / perturbation32 error ratios are `1.361e6` for
  pressure and `1.841e6` for geopotential.

## Commands Run

```bash
git merge-base --is-ancestor 131b27cd HEAD
python -m py_compile proofs/v014/fp32_acoustic_probes.py proofs/v014/exact_branch_memory_preflight.py proofs/v014/empirical_memory_map.py
python -m py_compile src/gpuwrf/coupling/scan_adapters.py proofs/v014/parallel_memory_fp32_manager.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/exact_branch_memory_preflight.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/parallel_memory_fp32_manager.py
JAX_COMPILATION_CACHE_DIR=/tmp/gpuwrf_jax_cache JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/parallel_memory_fp32_manager.py
JAX_COMPILATION_CACHE_DIR=/tmp/gpuwrf_jax_cache JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q tests/test_wdm6_savepoint_parity.py
JAX_COMPILATION_CACHE_DIR=/tmp/gpuwrf_jax_cache JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q 'tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[16]'
python -m json.tool proofs/v014/parallel_memory_fp32_manager.json >/tmp/parallel_memory_fp32_manager.validated.json
python -m json.tool proofs/v014/exact_branch_memory_preflight.json >/tmp/exact_branch_memory_preflight.validated.json
git diff --check
git diff -- src/gpuwrf
git add src/gpuwrf/coupling/scan_adapters.py proofs/v014/exact_branch_memory_preflight.json proofs/v014/exact_branch_memory_preflight.md proofs/v014/parallel_memory_fp32_manager.py proofs/v014/parallel_memory_fp32_manager.json proofs/v014/parallel_memory_fp32_manager.md .agent/reviews/2026-06-09-v014-parallel-memory-fp32-manager.md
```

Validation results:

- py_compile passed.
- `proofs/v014/parallel_memory_fp32_manager.py` passed and wrote JSON/MD.
- exact-branch memory preflight refresh passed in audit-only `NO_RUN_PLAN` mode.
- `tests/test_wdm6_savepoint_parity.py`: 85 passed.
- operational WDM6 adapter smoke: 1 passed.
- JSON validation passed for both proof JSON files.
- `git diff --check` passed.
- `git diff -- src/gpuwrf` shows only `scan_adapters.py`.
- The worker's own commit/push attempt failed before staging because its
  sandbox could not create
  `/home/enric/src/wrf_gpu2/.git/worktrees/v014-memory-fp32-manager/index.lock`
  on the shared Git metadata path. The primary manager re-ran the proof, JSON
  validation, `diff --check`, WDM6 savepoint parity, and WDM6 operational smoke
  before committing from the manager shell.

GPU status:

- No GPU run was performed.
- The refreshed preflight recorded `nvidia-smi` unavailable in this sandbox, so
  this lane stayed CPU-only.

## Risks

- The existing WDM6 adapter maps water to `0.0`, while the WDM6 kernel docstring
  says the scheme-level mask is `1=land, 2=water`. This branch deliberately
  preserves current values for bit identity; a semantic correction needs a
  separate physics-proof sprint.
- The exact-branch memory preflight is audit-only here. It is not a completed
  memory-fit GPU proof and not a long validation substitute.
- FP32 source work remains blocked until the primary manager closes or releases
  the live-nest `P/MU/W` perturbation-state initialization lock.

## Next Decision

Merge the WDM6 shape-only cleanup if the primary manager accepts a small
bit-identical memory commit during the current fp64 debug. Reopen FP32 R0/R1
only after the active grid-parity lock is released.
