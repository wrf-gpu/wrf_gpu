# Reviewer Report

## Findings

- minor — `tests/test_m2_kokkos.py:28-29` runs `scripts/m2_run_kokkos.sh` from pytest, and `scripts/m2_run_kokkos.sh:183-190` rewrites tracked Kokkos proof artifacts with fresh timing-dependent JSON. This matches the existing M2 pattern and does not invalidate the proof objects, but it creates repeat-validation churn in `artifacts/m2/*/*_profile.json` and should be cleaned up by a later shared M2 test-hygiene pass.
- note — `scripts/m2_run_kokkos.sh:154-168` computes `achieved_bandwidth_gbps` as host/device transfer bytes divided by kernel wall time. The artifacts label this as `fallback-derived` and document `ERR_NVGPUCTRPERM`, so this is acceptable for this sprint, but ADR-001 must not rank Kokkos by this value as if it were on-GPU streaming bandwidth.
- note — stencil register pressure is exactly at the contract ceiling: `artifacts/m2/kokkos/stencil_profile.json:21-24` reports `local_memory_bytes=0` and `registers_per_thread=64`. This passes, but leaves no headroom for future stencil changes.

## Contract Compliance

Compliant. The diff is within the worker-owned Kokkos paths plus the required worker report. Kokkos is built through `src/gpuwrf/backends/kokkos/build.sh:19-35` using Kokkos tag `4.7.01`, CUDA enabled, and `Kokkos_ARCH_BLACKWELL120=ON`; the bench is built and copied at `src/gpuwrf/backends/kokkos/build.sh:37-43`. The implementation uses Kokkos views and CUDA default execution space (`src/gpuwrf/backends/kokkos/host.cpp:23-27`) with one Kokkos launch for stencil (`src/gpuwrf/backends/kokkos/stencil.cpp:107-111`) and one for column (`src/gpuwrf/backends/kokkos/column.cpp:62-74`).

Required proof artifacts are present: `artifacts/m2/kokkos/stencil_profile.json`, `artifacts/m2/kokkos/column_profile.json`, `artifacts/m2/kokkos/correctness.json`, `artifacts/m2/kokkos/maintainability.md`, and `artifacts/m2/kokkos/agent_success.json`. Profile bounds pass: one launch each, CUDA execution space, runtime CC 12.0, column local memory zero, and registers 64/40.

## Correctness Risks

No blocking correctness risk found. Independent spot checks re-ran both fixture comparisons against the existing Kokkos outputs and both passed tier 1. The column `mse_delta` relative difference is large because the denominator is near zero, but `artifacts/m2/kokkos/correctness.json:82-89` shows the absolute error is far below the manifest tolerance and the variable passes.

## Performance Risks

Nsight Compute counters remain unavailable because of `ERR_NVGPUCTRPERM`; the sprint uses the already established M2 fallback path. The profiler JSON values are enough to populate the Kokkos row for ADR-001, but performance comparison should treat wall time, launch count, register count, local memory, and the fallback limitation separately rather than over-interpreting derived bandwidth.

## Required Fixes

None required before manager integration. Recommended follow-up: move artifact-regenerating pytest coverage to temporary outputs or an explicit regeneration mode across all M2 candidates, so ordinary `pytest -q` does not rewrite committed timing artifacts.

## Decision

Decision: Accept

## Reviewer Handoff

- objective: independently review the M2 Kokkos bakeoff worker/tester output against the sprint contract.
- files changed: `.agent/sprints/2026-05-19-m2-kokkos-stencil-column/reviewer-report.md`.
- commands run: `data/scratch/kokkos/bench config`; `cuobjdump --dump-sass data/scratch/kokkos/bench | grep -m1 'arch'`; both `gpuwrf.validation.compare_fixture` commands for stencil and column; `python -m json.tool` for both profile JSON files; `python scripts/validate_agentos.py`; `python scripts/check_m1_done.py`; `python scripts/check_m2_done.py || true`.
- proof objects produced: this reviewer report. Existing proof objects reviewed: the five Kokkos artifacts under `artifacts/m2/kokkos/`.
- unresolved risks: profiler-counter permission gap, stencil register headroom of zero, pytest artifact churn.
- next decision needed: manager can integrate this sprint and carry the artifact-churn cleanup as cross-candidate M2 hygiene, not a Kokkos acceptance blocker.
