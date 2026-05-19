# Reviewer Report

Role: reviewer / opus-reviewer
Sprint: 2026-05-19-m2-cupy-stencil-column
Branch: reviewer/opus/m2-cupy-stencil-column

## Findings

No blocker, major, or minor findings.

- note: Nsight Compute metrics are unavailable under the local permission state, so the candidate relies on the contract-approved fallback path; this is explicitly disclosed in both profile JSONs via `profiler_limitation` at `artifacts/m2/cupy_or_numba/stencil_profile.json:19` and `artifacts/m2/cupy_or_numba/column_profile.json:19`.
- note: The stencil kernel reports 64 B local memory at `artifacts/m2/cupy_or_numba/stencil_profile.json:17`. The contract requires zero local memory only for the column kernel, which is met at `artifacts/m2/cupy_or_numba/column_profile.json:17`.
- note: The column implementation is fixture-shaped around one 40-level column and a 64-thread block at `src/gpuwrf/backends/cupy/column.py:107` and `src/gpuwrf/backends/cupy/column.py:109`. This is acceptable for this M2 fixture, but should not be interpreted as production column generality.

## Contract Compliance

Pass. The worker stayed within the declared file ownership for implementation and artifacts. The implementation uses CuPy RawKernel, not Numba and not idiomatic CuPy array expressions: the kernels are defined as CUDA source strings and compiled through `cp.RawKernel` at `src/gpuwrf/backends/cupy/stencil.py:165` and `src/gpuwrf/backends/cupy/column.py:78`.

Acceptance criteria status:
- install/smoke: pass. `scripts/m2_run_cupy.sh` creates/reuses `data/scratch/m2-cupy-venv/`, pins `cupy-cuda13x==14.0.1` at `scripts/m2_run_cupy.sh:31`, and printed CUDA runtime `13000`.
- correctness: pass. Independent `compare_fixture` runs passed for both analytic fixtures with `first_failure: null`.
- profile JSON: pass. Both profile JSON files parse, include required numeric fields, include `profiler_limitation`, and report one kernel launch per problem.
- column local memory: pass. `local_memory_bytes` is 0 in `artifacts/m2/cupy_or_numba/column_profile.json:17`.
- maintainability and agent-success artifacts: pass. `maintainability.md` is under 300 words and covers install, error, debugger, and agent-friction topics; `agent_success.json` is present.
- tests and hygiene: pass with the caveat that global M2 completion is not expected yet.

## Correctness Risks

Low for the sprint target. I inspected the CUDA math against `src/gpuwrf/fixtures/analytic.py`; the stencil uses the same 4th-order horizontal, 2nd-order vertical, periodic update and the column kernel uses the same condensation/evaporation equations. Both candidate NPZs compare against the M1 samples inside tolerance, and the direct `pytest -q` run passed.

Residual risk: the column launch shape is specialized to the current 40-level fixture; that is acceptable for the M2 bakeoff proof object but should be revisited before any backend decision extrapolates to real physics-column dimensions.

## Performance Risks

The performance evidence is adequate for this sprint but not a full profiler result. Local `ncu` is blocked by `ERR_NVGPUCTRPERM`, so occupancy/register/local-memory values come from CuPy/CUDA attributes and bandwidth is fallback-derived. This matches the contract's permitted fallback pattern and is clearly recorded in the artifacts.

The measured wall times are well under the contract's 5 s sanity bound, and kernel launches are 1 for both problems. Register counts are within bounds: 58 for stencil and 24 for column.

## Required Fixes

None.

Recommended follow-up for ADR-001 evidence synthesis: treat CuPy's profiler numbers as fallback-derived rather than peer to candidates with successful hardware-counter reports, unless the local performance-counter permission is later changed and the sprint is rerun.

## Commands Run

- `bash scripts/m2_run_cupy.sh` -> exit 0, printed `13000`.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/m2-cupy/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` -> pass true.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/m2-cupy/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz` -> pass true.
- `python -m json.tool artifacts/m2/cupy_or_numba/stencil_profile.json` and column profile -> valid JSON.
- `pytest -q tests/test_m2_cupy.py` -> 2 passed.
- `pytest -q` -> 111 passed.
- `python scripts/validate_agentos.py` -> ok true.
- `python scripts/check_m1_done.py` -> ok true on serial rerun.
- `python scripts/check_m2_done.py` -> expected false: future M2 candidates, ADR-001, closeout, and manager artifacts are not complete yet.
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5` -> no new >100 KB sprint file observed.

## Proof Objects Reviewed

- `artifacts/m2/cupy_or_numba/stencil_profile.json`
- `artifacts/m2/cupy_or_numba/column_profile.json`
- `artifacts/m2/cupy_or_numba/correctness.json`
- `artifacts/m2/cupy_or_numba/maintainability.md`
- `artifacts/m2/cupy_or_numba/agent_success.json`
- `data/scratch/m2-cupy/stencil_out.npz`
- `data/scratch/m2-cupy/column_out.npz`

## Decision

Decision: Accept
