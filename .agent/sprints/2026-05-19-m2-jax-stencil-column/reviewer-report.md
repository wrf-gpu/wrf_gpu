# Reviewer Report

## Findings

Blocker: none.

Major: none.

Minor: `kernel_launches` is a compute-kernel count, not a complete XLA thunk count for the column path. The profile reports `kernel_launches: 1` in `artifacts/m2/jax/column_profile.json:27`, and the compute body is generated from `src/gpuwrf/backends/jax/column.py:32`, but the preserved XLA dump also contains one trivial `kCopy` thunk for `pressure_next`. This does not violate the sprint target of <=5 launches, but ADR-001 should record the column evidence as "one fused compute kernel plus one copy thunk" rather than "one total XLA action."

Note: `achieved_bandwidth_gbps` is fallback-derived from transfer bytes divided by the timed JIT call (`src/gpuwrf/backends/jax/bench.py:264` and `src/gpuwrf/backends/jax/bench.py:395`). The profile discloses this limitation in `artifacts/m2/jax/stencil_profile.json:30` and `artifacts/m2/jax/column_profile.json:30`. Treat it as a cross-candidate fallback metric, not measured DRAM bandwidth.

Note: The tester worktree includes uncommitted timing refreshes for earlier M2 candidates, outside the JAX worker diff. I did not review those as part of this JAX candidate decision and did not modify them.

## Contract Compliance

Compliant. Worker changes are within the contract-owned implementation and artifact paths plus the worker report. The JAX code uses `jax.jit`/`jax.numpy` only (`src/gpuwrf/backends/jax/stencil.py:45`, `src/gpuwrf/backends/jax/column.py:13`) and does not add JAX as a project dependency. `scripts/m2_run_jax.sh:16` creates/reuses the isolated venv under `data/scratch/m2-jax-venv/`, and `scripts/m2_run_jax.sh:33` verifies the GPU backend. The required artifacts exist under `artifacts/m2/jax/`: profile JSONs, correctness JSON, maintainability narrative, and agent-success log.

Profile fields satisfy the M2 bounds: both problems report `kernel_launches: 1`, `local_memory_bytes: 0`, registers 48/22, and occupancy 83.33%. The column local-memory requirement is satisfied in `artifacts/m2/jax/column_profile.json:28`.

## Correctness Risks

No correctness defect found. The JAX stencil and column formulas match the M1 analytic oracle definitions in `src/gpuwrf/fixtures/analytic.py:96` through `src/gpuwrf/fixtures/analytic.py:103` and `src/gpuwrf/fixtures/analytic.py:144` through `src/gpuwrf/fixtures/analytic.py:155`. Independent `compare_fixture` spot checks passed for both `analytic-stencil-3d-advdiff-v1` and `analytic-column-thermo-v1` with zero max absolute difference across all variables.

Residual risk: this is still only the M2 analytic fixture scale. It proves the JAX candidate row for the contracted bakeoff problems; it does not prove production-scale WRF physics behavior.

## Performance Risks

Nsight Compute counters are unavailable because of `ERR_NVGPUCTRPERM`; registers/local memory come from XLA dump PTX plus `ptxas`/`cuobjdump`, and occupancy/bandwidth are fallback-derived. My independent spot checks confirmed the preserved resource dumps show column `REG:22`, `LOCAL:0`, `0 bytes spill stores`, and `0 bytes spill loads`, and stencil `REG:48`, `LOCAL:0`, with a 40-byte stack frame but zero spills.

The fallback bandwidth numbers should not be used as absolute GPU memory-bandwidth evidence. They are acceptable for this sprint because the same limitation is disclosed and consistent with prior M2 candidates.

## Required Fixes

None for this sprint. The minor launch-count wording should be carried into ADR-001 when the manager compares backend candidates.

Independent commands run:

- `python -m json.tool artifacts/m2/jax/stencil_profile.json`
- `python -m json.tool artifacts/m2/jax/column_profile.json`
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/m2-jax/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/m2-jax/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz`
- `data/scratch/m2-jax-venv/bin/python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"`
- `grep` spot checks over JAX XLA thunk/resource dumps
- `python scripts/validate_agentos.py`
- `python scripts/check_m1_done.py`
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5`

## Decision

Decision: Accept
