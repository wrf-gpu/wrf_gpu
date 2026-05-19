# Worker Report

Summary: Implemented the M3 JAX/XLA state-grid-halo skeleton within the worker-owned paths: hashable/pytree `GridSpec`, GPU-resident fp64 `State` and `Tendencies`, no-op future-compatible `HaloSpec`, a single JITed `jax.lax.scan` dummy loop, transfer audit, spacetime budget, HLO proof, ADR-002 draft, and focused tests. Objective was the complete M3 skeleton with no dycore or physics math.

## Files Changed

- `src/gpuwrf/contracts/__init__.py`
- `src/gpuwrf/contracts/grid.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/halo.py`
- `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/timestep/__init__.py`
- `src/gpuwrf/timestep/dummy_loop.py`
- `src/gpuwrf/profiling/__init__.py`
- `src/gpuwrf/profiling/transfer_audit.py`
- `src/gpuwrf/profiling/budget.py`
- `scripts/m3_run_audits.py`
- `artifacts/m3/transfer_audit.json`
- `artifacts/m3/spacetime_budget.json`
- `artifacts/m3/hlo_dump/dummy_loop.txt`
- `artifacts/m3/maintainability.md`
- `artifacts/m3/agent_success.json`
- `.agent/decisions/ADR-002-state-layout.md`
- `tests/test_m3_grid.py`
- `tests/test_m3_state.py`
- `tests/test_m3_halo.py`
- `tests/test_m3_dummy_loop.py`
- `tests/test_m3_transfer_audit.py`
- `.agent/sprints/2026-05-19-m3-state-grid-halo-skeleton/worker-report.md`

## Commands Run

All Python validation used `PATH=/home/enric/src/wrf_gpu2/data/scratch/m2-jax-venv/bin:$PATH` so the pinned `jax==0.10.0` CUDA environment was active. I installed the repo editable plus `pytest` into that scratch venv only.

- `bash scripts/m2_run_jax.sh` -> exit 0; output: `gpu [CudaDevice(id=0)]`.
- `python -m gpuwrf.contracts.state --self-test` -> exit 0; output: `ok state_bytes=38656 tendency_bytes=38656 device=gpu`.
- `python scripts/m3_run_audits.py` -> exit 0; emitted zero transfer bytes and budget values listed below.
- `python -m json.tool artifacts/m3/transfer_audit.json` -> exit 0; `host_to_device_bytes_post_init=0`, `device_to_host_bytes_post_init=0`, `iterations=1000`, `jax_version=0.10.0`.
- `python -m json.tool artifacts/m3/spacetime_budget.json` -> exit 0.
- `head -50 artifacts/m3/hlo_dump/dummy_loop.txt` -> exit 0; HLO includes `%fused_subtract` in `jit(run_dummy_loop)/while/body/closed_call`.
- `pytest -q` -> exit 0; `250 passed in 56.09s`.
- `python scripts/check_m1_done.py` -> exit 0; `{"ok": true, "errors": [], "sprints_closed": 3}`.
- `python scripts/check_m2_done.py` -> exit 1; M2 artifacts satisfied but pre-existing lifecycle blocker remains: `sprint 2026-05-19-m2-adr-001-backend-selection: no tester log found to verify AI provenance`.
- `python scripts/check_m3_done.py` -> exit 1; code/proof objects present, but lifecycle gates are not worker-owned: M2 tester provenance, reviewer/tester/manager report stubs, missing M3 milestone closeout, M3 reviewer decision not accepted, and no M3 tester log.
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5` -> exit 0; largest tracked files: `1540850 WRF GPU Porting_ Architecture & Verification.pdf`, `97515 wrf to gpu gpt5.5 deep research.pdf`, `61080 fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`, `38916 tests/test_m2_triton_edge_cases.py`, `31304 tests/test_m2_jax_edge_cases.py`.

## Spacetime Budget

| Entry | Value | Justification |
|---|---:|---|
| state_bytes | 38656 | Sum of eight fp64 prognostic leaves allocated once on GPU. |
| tendency_bytes | 38656 | Same shapes as state, allocated once as preallocated tendency buffers. |
| temporary_bytes_per_step | 0 | Scan body has no array constructors; HLO fuses the theta add/sub chain. |
| halo_buffer_bytes | 0 | M3 single-GPU halo is a no-op interface stub. |
| total_persistent_bytes | 77312 | `state_bytes + tendency_bytes + halo_buffer_bytes`. |
| flops_per_cell_per_step | 4 | Multiply/add/subtract/multiply over `theta` only; no physics claim. |
| kernel_launches_per_step | 3 | Conservative HLO-derived fused-op count, within the <=5 gate. |
| wall_time_per_step_us | 2.8281795093789697 | Median of 100 compiled 1000-step loop runs after warmup. |

## Allocation Audit

- `src/gpuwrf/contracts/state.py:32` `jnp.zeros(...)`: necessary init-time allocation for frozen `State` / `Tendencies` leaves on GPU.
- `src/gpuwrf/contracts/grid.py:186` `jnp.linspace(...)`: necessary init-time eta-level coordinate for the Canary 3 km template.
- `src/gpuwrf/contracts/grid.py:187` `jnp.zeros(...)`: necessary init-time analytic terrain-height placeholder for the Canary 3 km template.
- Hot path `src/gpuwrf/timestep/dummy_loop.py`: no `jnp.array`, `jnp.asarray`, `jnp.zeros`, `jnp.empty`, `jnp.ones`, `jnp.full`, or `jnp.linspace` calls.

## Proof Objects

- `artifacts/m3/transfer_audit.json`
- `artifacts/m3/spacetime_budget.json`
- `artifacts/m3/hlo_dump/dummy_loop.txt`
- `artifacts/m3/maintainability.md`
- `artifacts/m3/agent_success.json`
- `.agent/decisions/ADR-002-state-layout.md`

## Risks

- Transfer counting uses `jax.profiler.trace` event scanning, not Nsight/CUPTI counters, because workstation perfmon permission is documented as blocked. JSON includes the trace directory for reviewer inspection.
- `check_m2_done.py` and therefore `check_m3_done.py` remain blocked by lifecycle/provenance artifacts owned by other roles, not by this worker patch.
- Running `scripts/m2_run_jax.sh` as required touched already-dirty M2 artifact JSONs; I did not stage or commit M2 files.

## Handoff

Objective complete for worker-owned M3 implementation and proof objects. Next decision needed: tester/reviewer/manager should validate allocation audit, complete their reports/logs, resolve the pre-existing M2 tester provenance gate, and finalize ADR-002 / M3 closeout.
