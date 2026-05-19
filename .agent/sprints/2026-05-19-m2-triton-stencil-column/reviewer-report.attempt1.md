# Reviewer Report

## Findings

- **blocker** — `src/gpuwrf/backends/triton/bench.py:95` copies every recent cubin for a problem and `src/gpuwrf/backends/triton/bench.py:112` parses the combined cuobjdump text with `_parse_resource_usage`, whose `src/gpuwrf/backends/triton/bench.py:65` return value is `max(regs)` across all functions. In the `--problem both` run this lets the column profile see the earlier stencil cubin and report the stencil register count.
- **blocker** — `artifacts/m2/triton/column_profile.json:20` and `artifacts/m2/triton/column_profile.json:22` report `occupancy_pct=70.83333333333333` and `registers_per_thread=60`, but the committed cuobjdump artifact shows the actual column kernel at `data/profiler_artifacts/triton/column_cuobjdump_resource_usage.txt:6`-`7` as `_column_thermo_kernel: REG:34 STACK:0 SHARED:0 LOCAL:0`. The same artifact also includes the unrelated stencil kernel at `data/profiler_artifacts/triton/column_cuobjdump_resource_usage.txt:14`-`15`, explaining the inflated profile value. This is load-bearing M2 evidence, because the sprint contract makes column registers/occupancy the main Triton-vs-JAX signal for ADR-001.
- **minor** — `src/gpuwrf/backends/triton/stencil.py:177` and `src/gpuwrf/backends/triton/column.py:191` pass only the problem name into the resource callback. That interface does not identify the expected kernel symbol, so future kernels with multiple Triton-generated cubins can repeat the same artifact contamination unless the parser becomes symbol-aware or each problem compiles in an isolated cache.

## Contract Compliance

Pass: direct `@triton.jit` kernels are present, torch is scoped to the data venv, `pyproject.toml` was not changed, both fixture comparisons pass, JSON is syntactically valid, `local_memory_bytes` is 0 for the column kernel, and tracked file size bounds are unchanged.

Fail: profile JSON resource metrics are not reliably "from cuobjdump on the cubin" for the column problem. The artifact is from cuobjdump, but the value is from the wrong function section. `python scripts/check_m1_done.py` and `python scripts/check_m2_done.py` also fail in this worktree because the added tester edge case exposes the same profile mismatch; M2 additionally remains incomplete for expected milestone-level reasons such as gt4py/ADR/closeout gaps.

## Correctness Risks

Numerical correctness looks good for this sprint scope. I independently reran both `compare_fixture` commands: stencil passed with all max diffs 0, and column passed within manifest tolerance (`qv_next` max abs diff `4.336808689942018e-19`, `mse_delta` max abs diff `1.0802470029602773e-12`). The live correctness risk is not physics parity; it is that ADR-001 could consume false performance/resource evidence.

## Performance Risks

The column profile currently understates Triton's column result relative to the actual cubin: the profile says 60 registers and 70.83% occupancy, while `cuobjdump` says the column kernel uses 34 registers, zero stack, zero local memory. Any backend comparison using the committed `column_profile.json` would make the wrong decision about Triton's column behavior. Bandwidth remains fallback-derived from transfer bytes and wall time, not hardware DRAM throughput, which is acceptable only because the profile documents the ncu permission limitation.

## Required Fixes

1. Make Triton resource parsing kernel-symbol-aware (`_stencil_advdiff_kernel` vs `_column_thermo_kernel`) or isolate/clear the Triton cache between stencil and column artifact generation.
2. Regenerate `artifacts/m2/triton/column_profile.json`, `data/scratch/m2-triton/column_run.json`, and the column cuobjdump artifact so the profile reports the column kernel's own REG/LOCAL and derived occupancy.
3. Re-run the focused tester guard `pytest -q -p no:cacheprovider tests/test_m2_triton_edge_cases.py::test_column_profile_registers_match_column_kernel_cubin`, then the sprint validation commands needed to refresh the worker/tester reports.

## Decision

Decision: Reject

## Handoff

- objective: independent review of the M2 Triton stencil/column sprint.
- files changed: `.agent/sprints/2026-05-19-m2-triton-stencil-column/reviewer-report.md` only.
- commands run: `compare_fixture` for stencil and column, `json.tool` for both Triton profiles, `cuobjdump --dump-resource-usage` on `column_triton_0.cubin` and `column_triton_1.cubin`, venv smoke import, `validate_agentos.py`, `check_m1_done.py`, `check_m2_done.py`, focused pytest for `test_column_profile_registers_match_column_kernel_cubin`, tracked file size check.
- proof objects produced: this reviewer report; no source edits.
- unresolved risks: existing uncommitted tester/report/profile changes were present before this report write and were not modified by me.
- next decision needed: bounce to the worker for the narrow resource-parser/profile regeneration fix, then rerun tester/reviewer sign-off.
