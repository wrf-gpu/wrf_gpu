# Reviewer Report

Sprint: `2026-05-19-m2-triton-stencil-column`
Role: reviewer / Codex gpt-5.5 acting as opus-reviewer
Branch: `reviewer/opus/m2-triton-stencil-column`

## Findings

- **note** — `src/gpuwrf/backends/triton/bench.py:55` now parses `cuobjdump --dump-resource-usage` by exact kernel symbol, and `src/gpuwrf/backends/triton/bench.py:99` selects `_stencil_advdiff_kernel` vs `_column_thermo_kernel` per problem before accepting resource metrics. This fixes the attempt-1 blocker where the column profile could inherit the stencil cubin's REG count.
- **note** — `artifacts/m2/triton/column_profile.json:19` and `artifacts/m2/triton/column_profile.json:21` now report `occupancy_pct=100.0` and `registers_per_thread=34`. Independent `cuobjdump --dump-resource-usage data/profiler_artifacts/triton/column_triton_0.cubin` shows `Function _column_thermo_kernel: REG:34 STACK:0 SHARED:0 LOCAL:0`, so the amended AC #6/#14a evidence is internally consistent.
- **minor** — `tests/test_m2_triton_edge_cases.py:18`-`22` still describe the worker bench as using "max-of-all-cubins". That comment is stale after the attempt-2 fix. It does not affect test behavior or sprint evidence, but it should be cleaned in a later documentation/test-comment pass to avoid confusing ADR readers.

## Contract Compliance

The sprint satisfies the amended contract for this candidate. Direct `@triton.jit` kernels exist in `src/gpuwrf/backends/triton/stencil.py:29` and `src/gpuwrf/backends/triton/column.py:29`; torch remains scoped to the data venv runner and is not added to `pyproject.toml`; both profile JSONs are valid; both problems report one Triton launch; column `local_memory_bytes` is 0; and the resource metrics are now kernel-symbol-aware.

`python scripts/check_m1_done.py` passes when run sequentially. `python scripts/check_m2_done.py` remains false for M2-wide reasons outside this sprint's worker scope: the sprint is not manager-closed yet, GT4Py artifacts are absent, ADR-001 is absent, M2 closeout is absent, and the M2 bakeoff reviewer decision is not accepted.

## Correctness Risks

No sprint-blocking correctness issue found. I independently reran both fixture comparisons against the generated Triton outputs in `data/scratch/m2-triton/`. Stencil passed with all max diffs 0. Column passed within manifest tolerance, with `qv_next` max abs diff `4.336808689942018e-19` and `mse_delta` max abs diff `1.0802470029602773e-12`.

The main residual correctness caveat is scope: these are M2 analytic fixtures, not real WRF physics. That is expected for this milestone and must not be overstated in ADR-001.

## Performance Risks

The column resource evidence is now usable for ADR-001: profile JSON, copied cubin, and independent cuobjdump agree on REG 34, LOCAL 0, STACK 0, and derived 100% occupancy. The stencil evidence also agrees on REG 60 and LOCAL 0, with `STACK:48`; the project should continue treating `local_memory_bytes` as the M2 convention's LOCAL field, not as a full spill audit.

`achieved_bandwidth_gbps` remains fallback-derived from transfer bytes and wall time because local Nsight Compute performance counters are permission-blocked. The profile fields document this limitation, so this is acceptable as bakeoff evidence but should not be cited as measured DRAM bandwidth.

## Required Fixes

None before manager integration. Optional cleanup: update the stale explanatory comment in `tests/test_m2_triton_edge_cases.py`.

## Independent Spot Checks

- `python -m gpuwrf.validation.compare_fixture ... analytic-stencil ...` -> pass.
- `python -m gpuwrf.validation.compare_fixture ... analytic-column ...` -> pass.
- `python -m json.tool artifacts/m2/triton/{stencil_profile,column_profile}.json` -> pass.
- `cuobjdump --dump-resource-usage data/profiler_artifacts/triton/{column,stencil}_triton_0.cubin` -> column REG 34 / LOCAL 0 / STACK 0; stencil REG 60 / LOCAL 0 / STACK 48.
- `pytest -q tests/test_m2_triton_edge_cases.py::test_column_profile_registers_match_column_kernel_cubin tests/test_m2_triton_edge_cases.py::test_stencil_profile_registers_match_stencil_kernel_cubin tests/test_m2_triton_edge_cases.py::test_cuobjdump_resource_usage_artifact_has_kernel_section tests/test_m2_triton_edge_cases.py::test_run_json_matches_profile_numbers` -> 4 passed.
- `pytest -q tests/test_m2_triton.py tests/test_m2_triton_edge_cases.py` -> 46 passed.
- `python scripts/validate_agentos.py` -> ok.
- `python scripts/check_m1_done.py` -> ok.
- `python scripts/check_m2_done.py` -> expected false for M2-wide closeout/GT4Py/ADR gaps listed above.

## Decision

Decision: Accept

## Handoff

- objective: independent review of the attempt-2 M2 Triton stencil/column candidate.
- files changed: `.agent/sprints/2026-05-19-m2-triton-stencil-column/reviewer-report.md` only.
- commands run: mandatory governance/contract reads; worker/tester/diff inspection; fixture comparisons; JSON validation; independent cuobjdump checks; focused Triton cubin tests; Triton test files; `validate_agentos.py`; `check_m1_done.py`; `check_m2_done.py`.
- proof objects produced: this reviewer report.
- unresolved risks: M2-wide closeout remains incomplete; bandwidth is fallback-derived; stencil has a reported stack frame even though LOCAL is 0.
- next decision needed: manager can integrate this sprint evidence into the M2 bakeoff set, then decide whether to run GT4Py/remediation or proceed to ADR-001 per the sprint contract.
