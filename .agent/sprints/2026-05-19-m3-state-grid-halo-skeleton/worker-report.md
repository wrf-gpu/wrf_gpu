# Worker Report

Summary: Attempt 2 addressed the reviewer Reject items only. The patch enables JAX x64 at `gpuwrf.contracts` import time before `GridSpec` arrays are created, makes `GridSpec` array-aware for equality/hash/static JIT keys, parses nested `memcpy_details` in the transfer audit, removes the post-init `dt` scalar H2D copy by making `dt` static in the dummy loop JIT, and reports the raw HLO-derived launch count without clamping. Regenerated M3 proof artifacts show zero parsed post-init transfer bytes and `kernel_launches_per_step=3`.

## Files Changed

- `src/gpuwrf/contracts/__init__.py`
- `src/gpuwrf/contracts/grid.py`
- `src/gpuwrf/timestep/dummy_loop.py`
- `src/gpuwrf/profiling/transfer_audit.py`
- `src/gpuwrf/profiling/budget.py`
- `scripts/m3_run_audits.py`
- `artifacts/m3/transfer_audit.json`
- `artifacts/m3/spacetime_budget.json`
- `artifacts/m3/hlo_dump/dummy_loop.txt`
- `tests/test_m3_grid.py`
- `tests/test_m3_state.py`
- `tests/test_m3_transfer_audit.py`
- `tests/test_m3_edge_cases.py`
- `.agent/sprints/2026-05-19-m3-state-grid-halo-skeleton/worker-report.md`

## Commands Run + Output

All Python commands used `PATH=/home/enric/src/wrf_gpu2/data/scratch/m2-jax-venv/bin:$PATH`.

- `bash scripts/m2_run_jax.sh` -> exit 0; output: `gpu [CudaDevice(id=0)]`. It rewrote M2 timing JSONs as part of the runner; those generated M2 deltas were reversed because they are outside worker ownership.
- `python -m gpuwrf.contracts.state --self-test` -> exit 0; output: `ok state_bytes=38656 tendency_bytes=38656 device=gpu`.
- `python scripts/m3_run_audits.py` -> exit 0; output summary: `host_to_device_bytes_post_init=0`, `device_to_host_bytes_post_init=0`, `iterations=1000`, `state_bytes=38656`, `tendency_bytes=38656`, `kernel_launches_per_step=3`, `wall_time_per_step_us=2.6409454876556993`.
- `python -m json.tool artifacts/m3/transfer_audit.json` -> exit 0; output includes `host_to_device_bytes_post_init: 0`, `device_to_host_bytes_post_init: 0`, `jax_version: 0.10.0`, trace dir `data/scratch/m3/transfer_trace`.
- `python -m json.tool artifacts/m3/spacetime_budget.json` -> exit 0; output includes `temporary_bytes_per_step: 0`, `total_persistent_bytes: 77312`, `kernel_launches_per_step: 3`.
- `head -50 artifacts/m3/hlo_dump/dummy_loop.txt` -> exit 0; output begins `HloModule jit_run_dummy_loop` and shows `%fused_subtract` with `constant(3)` embedded in the while body, confirming `dt` is static.
- `pytest -q tests/test_m3_grid.py tests/test_m3_transfer_audit.py tests/test_m3_edge_cases.py -q` -> exit 0; output: `53 passed`.
- `pytest -q` -> exit 0; output: `298 passed in 57.49s`.
- `python scripts/check_m1_done.py` -> exit 0; output: `{"errors": [], "manifest_dir": "fixtures/manifests", "ok": true, "sprints_closed": 3}`.
- `python scripts/check_m2_done.py` -> exit 1; output: candidates `6/6` satisfied; only error is pre-existing lifecycle provenance: `sprint 2026-05-19-m2-adr-001-backend-selection: no tester log found to verify AI provenance`.
- `python scripts/check_m3_done.py` -> exit 1 after this report existed; output showed the same M2 provenance blocker plus manager-owned M3 lifecycle artifacts (`manager-closeout.md`, `memory-patch.md`, milestone closeout, reviewer decision). No implementation/proof-object blocker was reported.
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5` -> exit 0; largest tracked files unchanged: `1540850 WRF GPU Porting_ Architecture & Verification.pdf`, `97515 wrf to gpu gpt5.5 deep research.pdf`, `61080 fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`, `38916 tests/test_m2_triton_edge_cases.py`, `31304 tests/test_m2_jax_edge_cases.py`.
- `git diff --check` -> exit 0; no whitespace errors.

## Spacetime Budget

| Entry | Value | Justification |
|---|---:|---|
| state_bytes | 38656 | Eight fp64 State leaves allocated once on GPU. |
| tendency_bytes | 38656 | Matching fp64 Tendencies leaves allocated once on GPU. |
| temporary_bytes_per_step | 0 | No hot-path array constructors; scan body only transforms carry leaves. |
| halo_buffer_bytes | 0 | M3 single-GPU halo remains a no-op interface. |
| total_persistent_bytes | 77312 | State + tendencies + halo buffer. |
| kernel_launches_per_step | 3 | Raw HLO-derived fusion/custom-call count, not clamped. |
| wall_time_per_step_us | 2.6409454876556993 | Median of 100 compiled 1000-step loop runs after warmup. |

## Allocation Audit

- `GridSpec.canary_3km_template`: `jnp.linspace(...)` for eta levels and `jnp.zeros(...)` for analytic terrain placeholder; both init-only metadata arrays.
- `State.zeros` / `Tendencies.zeros`: one `jnp.zeros(...)` per persistent fp64 state/tendency leaf; all init-only and placed on GPU.
- `dummy_loop.py`: no `jnp.array`, `jnp.asarray`, `jnp.zeros`, `jnp.empty`, `jnp.ones`, `jnp.full`, or `jnp.linspace` in the JIT hot path.

## Proof Objects

- `artifacts/m3/transfer_audit.json`
- `artifacts/m3/spacetime_budget.json`
- `artifacts/m3/hlo_dump/dummy_loop.txt`
- Existing attempt-1 proof objects retained: `artifacts/m3/maintainability.md`, `artifacts/m3/agent_success.json`, `.agent/decisions/ADR-002-state-layout.md`

## Risks

- `GridSpec.__hash__` and `__eq__` compare small static grid arrays via host views. This is outside the timestep loop and is only for static compile keys; future full-size terrain metadata may need a precomputed checksum field rather than content hashing.
- Transfer auditing still uses JAX profiler trace parsing because Nsight/CUPTI counters are blocked by workstation perfmon permissions. The parser now handles nested `memcpy_details`, and the regenerated trace parses to `(H2D=0, D2H=0)`.
- `check_m2_done.py` and `check_m3_done.py` remain non-ok for lifecycle artifacts outside worker ownership.

## Handoff

Objective: close the four M3 attempt-2 implementation/proof blockers. Files changed are listed above. Commands run are listed above with outputs. Proof objects are the regenerated M3 JSON/HLO artifacts plus this report. Unresolved risks are lifecycle/provenance items owned by manager/tester/reviewer, not implementation. Next decision needed: reviewer should re-check the transfer trace parser against `data/scratch/m3/transfer_trace`, then manager should complete closeout artifacts.
