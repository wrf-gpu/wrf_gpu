# Memory Manager 2026-06-09

- Recommendation: `REVIEW_ONLY`
- Branch: `worker/gpt/v013-close-manager`
- HEAD: `6ced5a8e293a938504c4bf5e64c1018140f29702`
- CPU-only: `True`
- GPU used: `False`
- Production source edits: `False`

## Current Lock

- Active grid verdict: `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP`
- Active grid worker observed: `True`
- Current blocker: exact WRF start-domain base-state source boundary before the hypsometric `AL/ALT` pass.
- Resulting lock: no memory/FP32 source work, no GPU, no `d02_replay.py`, runtime, dycore, state-contract, boundary, restart, init, wrfout, or live-nest/base-state edits.

## Prior Closed Work

- Prior side-manager recommendation: `MERGE_NOW`.
- Closed memory fix: `WDM6_SLMSK_SHAPE_CLEANUP_EXACT`.
- WDM6 target saving: `76.92176055908203` MiB at 641x321x50.
- FP32 source verdict: `FP32_SOURCE_WORK_INFEASIBLE_WITH_CURRENT_LOCKS`.

## Collision Map

| Item | Status | Blocking lock | Resume gate |
|---|---|---|---|
| `exact_branch_memory_preflight` | `defer_gpu_run_until_grid_parity_branch_stabilizes` | none | primary manager releases GPU and selected post-grid-parity branch |
| `moisture_transport_velocity_reuse` | `blocked_by_grid_parity_source_lock` | src/gpuwrf/runtime/operational_mode.py, src/gpuwrf/dynamics/flux_advection.py | runtime/dynamics locks released and active moisture advection is on the validation path |
| `non_radiation_physics_column_tiling_pilot` | `measure_first_after_grid_parity` | active-grid-debug-touched files if any overlap at dispatch time | exact branch preflight or HLO/RSS evidence names a measured non-radiation offender |
| `moisture_limiter_workspace_reduction` | `blocked_by_grid_parity_source_lock` | src/gpuwrf/dynamics/flux_advection.py, src/gpuwrf/runtime/operational_mode.py | dycore/runtime locks released and limiter liveness measured as material |
| `acoustic_carry_split_or_pad_cleanup` | `blocked_by_same_fault_surface` | src/gpuwrf/dynamics/**, src/gpuwrf/runtime/operational_mode.py | P/MU/W live-nest/base-state grid-parity lock released |
| `state_alias_reduction` | `blocked_by_state_contract_lock_and_adr` | src/gpuwrf/contracts/state.py, boundary, restart, init, wrfout compatibility files | ADR approved after grid-parity branch stabilizes |
| `R0 precision-mode contract` | `review_only_until_grid_lock_released` | src/gpuwrf/runtime/operational_mode.py | default-off fp64 bit identity and cache-key/report-label tests |
| `R1 explicit base-state plumbing` | `blocked_by_active_base_state_boundary_debug` | src/gpuwrf/dynamics/**, src/gpuwrf/runtime/operational_mode.py, boundary/restart/init/carry staging | focused acoustic prep/finish exactness plus one-step operational carry test |
| `R2 perturbation-authoritative acoustic state` | `blocked_by_dycore_lock` | src/gpuwrf/dynamics/** | small-step WRF savepoint parity, idealized dry gates, transfer audit, VRAM proof |
| `R3 CPU scalar and one-column probes` | `already_available_proof_only` | none | refresh only if ADR or precision formulas change |

## Exact-Branch Preflight

- Current status: `NO_RUN_PLAN`.
- GPU attempted in this lane: `False`.
- Prepared command: `['scripts/run_gpu_lowprio.sh', '--cores', '0-23', '--', 'python', 'proofs/v014/exact_branch_memory_preflight.py', '--run-gpu', '--nested-input', '<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z', '--max-dom', '3', '--hours', '1', '--timeout-s', '600.0']`.
- This is deliberately not a long validation substitute.

## GPU Usage

- GPU used: `False`.
- Peak VRAM: `null`.
- Lock protocol evidence: no GPU command was launched; the queued preflight command uses `scripts/run_gpu_lowprio.sh`; active grid worker `gpt-base-boundary` was visible in tmux.

## Top 5 Next Tasks After Grid Parity

1. Run exact-branch memory preflight on the stabilized post-grid-parity branch - gate: repo GPU lock wrapper, peak VRAM, allocator mode, output count, finiteness record.
2. Resume FP32 R0/R1 only after the live-nest/base-state lock is released - gate: default-off fp64 bit identity and explicit base-state plumbing proof.
3. Moisture transport velocity reuse if active moisture advection is in the validation path - gate: default exactness plus active-path conservation, positivity, no-transfer audit.
4. Measurement-led non-radiation column-tiling pilot - gate: one scheme tile-vs-untiled exact proof, short GPU VRAM suite, real-case smoke.
5. Co-design acoustic carry split with FP32 acoustic after fp64 parity is repaired - gate: fp64 exactness, acoustic savepoint parity, warm-bubble/Straka/terrain-rest gates.

## Files Changed

- `proofs/v014/memory_manager_260609.py`
- `proofs/v014/memory_manager_260609.json`
- `proofs/v014/memory_manager_260609.md`
- `.agent/reviews/2026-06-09-v014-memory-manager-260609.md`

## Commands Run

- `python -m py_compile proofs/v014/memory_manager_260609.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/memory_manager_260609.py`
- `python -m json.tool proofs/v014/memory_manager_260609.json >/tmp/memory_manager_260609.validated.json`
- `git diff --check`
- `git diff -- src/gpuwrf`

## Proof Objects

- `proofs/v014/memory_manager_260609.json`
- `proofs/v014/memory_manager_260609.md`
- `.agent/reviews/2026-06-09-v014-memory-manager-260609.md`

## Recommendation

`REVIEW_ONLY`: there is no production source change to merge from this lane. Keep the report as the current memory/FP32 side-manager handoff, and resume source work only after the primary grid-parity manager releases the relevant locks.
