# v0.11.0 Nesting Status

## objective

Implement a WRF-faithful live multi-domain nesting runtime for the Canary 9/3/1 km hierarchy on branch `worker/gpt/v0110-nesting`, using parent-produced live boundary packages, child subcycling, per-domain cadence, multi-domain output, and optional two-way feedback behind a gate.

The active sprint contract used for this worker was the principal dispatch in this session. I did not find a checked-in v0.11.0 sprint contract file under `.agent/sprints`.

## files changed

- `src/gpuwrf/runtime/domain_tree.py` (new): live domain-tree orchestration calling the existing single-domain operational step entry.
- `src/gpuwrf/coupling/boundary_feedback.py` (new): optional two-way feedback behind an explicit runtime gate.
- `src/gpuwrf/contracts/grid.py` (SHARED-FILE EDIT): append-only `DomainNest` and `DomainHierarchy` metadata. No edits to `src/gpuwrf/runtime/operational_mode.py` or `src/gpuwrf/contracts/state.py`.
- `tests/test_v0110_domain_tree.py`, `tests/test_v0110_boundary_feedback.py`: focused unit coverage.
- `proofs/v0110/nesting_cpu_gates.py`, `proofs/v0110/nesting_cpu_gates.json`: CPU structural proof.
- `proofs/v0110/nesting_live_smoke.py`, `proofs/v0110/nesting_live_smoke.json`: GPU-locked live nested real-data smoke proof.

## commands run

- `git log -1 --oneline`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_COMPILATION_CACHE_DIR= pytest -q tests/test_v0110_domain_tree.py tests/test_v0110_boundary_feedback.py tests/test_p0_1a_nesting.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_COMPILATION_CACHE_DIR= python proofs/v0110/nesting_cpu_gates.py`
- `/tmp/wrf_gpu_run.sh taskset -c 0-27 env PYTHONPATH=src python proofs/v0110/nesting_live_smoke.py --max-dom 3 --root-steps 1`

## proof objects produced

- `proofs/v0110/nesting_cpu_gates.json`: PASS.
  - 5-domain hierarchy accepted (`d01 -> d02`, `d02 -> d03/d04/d05`, `max_dom=5`).
  - Expected/observed subcycling over 2 root steps: `d01=2`, `d02=6`, `d03=d04=d05=18`.
  - 20 live force events were observed in recursive order.
  - Multi-domain output was fine-to-coarse synchronized.
  - Feedback conservation probe residual was exactly zero with feedback still default-off.

- `proofs/v0110/nesting_live_smoke.json`: FAIL due `qke` only.
  - Live d01->d02->d03 real-data run executed under the GPU mutex.
  - Observed steps matched hierarchy ratios: `d01=1`, `d02=3`, `d03=9`.
  - Multi-domain output emitted synchronized states for `d03` step 9, `d02` step 3, `d01` step 1.
  - Parent-produced live boundary packages reached children (`d02`/`d03` boundary time levels = 2).
  - Core prognostic fields stayed finite on all domains: `theta`, `qv`, `u`, `v`, `w`, `p_perturbation`, `ph_perturbation`, `mu_perturbation`.
  - `qke` finite fractions: `d01=0.9994532531437944`, `d02=0.9959024204307223`, `d03=0.9902508960573476`.

## unresolved risks

- Full all-field finite nested equivalence is not closed. The raw live smoke fails because MYNN `qke` goes non-finite in a small fraction of cells even on `d01`, where no child boundary or feedback can influence the parent. This points to the existing operational `qke` robustness class, not the new nesting orchestration, but it still blocks a clean FULL equivalence claim.
- No 24 h, RMSE, CPU-WRF, profiler, or transfer-audit claim is made here.
- Two-way feedback is implemented and unit-proven for conservation/extent behavior but not enabled in the live forecast proof.
- Nested in-loop `w` relaxation is off in the live smoke pending a longer stability gate.

## next decision needed

Dispatch second-line Opus debug on the surviving `qke` non-finite blocker, or accept this merge as the domain-tree/live-boundary/subcycling implementation with the all-field nested equivalence gate still open.
