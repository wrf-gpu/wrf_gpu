# v0.11.0 Nesting Status

## objective

Implement a WRF-faithful live multi-domain nesting runtime for the Canary 9/3/1 km hierarchy on branch `worker/gpt/v0110-nesting`, using parent-produced live boundary packages, child subcycling, per-domain cadence, multi-domain output, and optional two-way feedback behind a gate.

This 2026-06-05 continuation was CPU-only by manager instruction. No GPU command was run in this round; the 24 h/RMSE/profiler/live GPU reruns are intentionally deferred until the recompile-diagnosis work releases the GPU.

## files changed

- `src/gpuwrf/runtime/domain_tree.py` (new): live domain-tree orchestration calling the existing single-domain operational step entry. Continuation hardening: per-domain global step clocks for subcycled children, and feedback weights precomputed so the runtime feedback gate can flip on without rebuilding the tree.
- `src/gpuwrf/coupling/boundary_feedback.py` (new): optional child-to-parent feedback machinery behind an explicit runtime gate.
- `src/gpuwrf/nesting/boundary_construction.py`: parent-to-child forcedown validation now rejects unknown registrations and invalid boundary widths instead of accepting ambiguous fallback behavior.
- `src/gpuwrf/physics/mynn_pbl.py`: MYNN `qke` post-solve bound now uses WRF `MAX/MIN` semantics (`fmax/fmin`) matching `phys/module_bl_mynnedmf.F:3106-3107`, which converts NaN solver outputs to `qkemin` instead of propagating NaN.
- `src/gpuwrf/contracts/grid.py` (SHARED-FILE EDIT from earlier nesting commit): append-only `DomainNest` and `DomainHierarchy` metadata. This continuation made no edits to `src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/contracts/state.py`, or `src/gpuwrf/contracts/grid.py`.
- `tests/test_v0110_domain_tree.py`, `tests/test_v0110_boundary_feedback.py`, `tests/test_v0110_qke_finiteness.py`, `tests/test_p0_1a_nesting.py`: focused CPU coverage for subcycling clocks, feedback gate behavior, qke finiteness semantics, and forcedown validation.
- `proofs/v0110/nesting_cpu_gates.py`, `proofs/v0110/nesting_cpu_gates.json`: refreshed CPU structural proof for the continuation.
- `proofs/v0110/nesting_live_smoke.py`, `proofs/v0110/nesting_live_smoke.json`: prior GPU-locked live nested real-data smoke proof, retained as pre-fix evidence.

## commands run

- `git log -1 --oneline`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_COMPILATION_CACHE_DIR= pytest -q tests/test_v0110_domain_tree.py tests/test_v0110_boundary_feedback.py tests/test_v0110_qke_finiteness.py tests/test_p0_1a_nesting.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_COMPILATION_CACHE_DIR= python -m py_compile src/gpuwrf/runtime/domain_tree.py src/gpuwrf/nesting/boundary_construction.py src/gpuwrf/physics/mynn_pbl.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_COMPILATION_CACHE_DIR= pytest -q tests/test_m5_mynn_tridiagonal.py tests/test_m5_mynn_tier2.py tests/test_mynn_edmf_oracle.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_COMPILATION_CACHE_DIR= python proofs/v0110/nesting_cpu_gates.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_COMPILATION_CACHE_DIR= pytest -q tests/test_v0110_domain_tree.py tests/test_v0110_boundary_feedback.py tests/test_v0110_qke_finiteness.py tests/test_p0_1a_nesting.py tests/test_m5_mynn_tridiagonal.py tests/test_m5_mynn_tier2.py tests/test_mynn_edmf_oracle.py` (`31 passed`)
- Prior round only: `/tmp/wrf_gpu_run.sh taskset -c 0-27 env PYTHONPATH=src python proofs/v0110/nesting_live_smoke.py --max-dom 3 --root-steps 1`

## proof objects produced

- `proofs/v0110/nesting_cpu_gates.json`: PASS.
  - 5-domain hierarchy accepted (`d01 -> d02`, `d02 -> d03/d04/d05`, `max_dom=5`).
  - Expected/observed subcycling over 2 root steps: `d01=2`, `d02=6`, `d03=d04=d05=18`.
  - Child subcycle start clocks are domain-global: first `d03` chunks start at `1, 4, 7, 10, 13, 16`.
  - 20 live force events were observed in recursive order.
  - Multi-domain output was fine-to-coarse synchronized.
  - Feedback conservation probe residual was exactly zero.
  - Runtime feedback gate is live: feedback callback calls were `off=0`, `on=1`.
  - MYNN qke finiteness probe passed: `[nan, -1, qkemin*0.1, 0.25, 200]` maps to finite `[qkemin, qkemin, qkemin, 0.25, 150]`.

- `proofs/v0110/nesting_live_smoke.json`: stale pre-qke-fix FAIL due `qke` only.
  - Live d01->d02->d03 real-data run executed under the GPU mutex in the prior round.
  - Observed steps matched hierarchy ratios: `d01=1`, `d02=3`, `d03=9`.
  - Multi-domain output emitted synchronized states for `d03` step 9, `d02` step 3, `d01` step 1.
  - Parent-produced live boundary packages reached children (`d02`/`d03` boundary time levels = 2).
  - Core prognostic fields stayed finite on all domains: `theta`, `qv`, `u`, `v`, `w`, `p_perturbation`, `ph_perturbation`, `mu_perturbation`.
  - Pre-fix `qke` finite fractions were `d01=0.9994532531437944`, `d02=0.9959024204307223`, `d03=0.9902508960573476`; this must be rerun after the GPU is released.

## unresolved risks

- No GPU live smoke was rerun in this CPU-only continuation, so full all-field finite nested equivalence is not yet re-proven after the MYNN qke fix.
- No 24 h, RMSE, CPU-WRF, profiler, or transfer-audit claim is made here.
- Two-way feedback is now live-gated and unit-proven, including state overlap update and total-field rebuild behavior, but it has not yet been enabled in a long live forecast proof.
- Nested in-loop `w` relaxation remains deferred to a longer stability gate.

## next decision needed

When the GPU is released, rerun the d01->d02->d03 live smoke and then the 24 h/RMSE/profiler gates. If qke remains non-finite after this WRF min/max semantics fix, dispatch the second-line Opus debug with the refreshed proof object and pre/post-fix qke evidence.
