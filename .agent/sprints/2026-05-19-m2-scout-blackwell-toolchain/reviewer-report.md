# Reviewer Report

## Findings

- note - No blocker, major, or minor findings against the contracted worker deliverable. The changed worker files are limited to the contract ownership set plus the required worker report, and the proof-object paths match the sprint contract's matrix, narrative, hello-GPU, wrapper, and test requirements (`.agent/sprints/2026-05-19-m2-scout-blackwell-toolchain/sprint-contract.md:37`, `.agent/sprints/2026-05-19-m2-scout-blackwell-toolchain/sprint-contract.md:58`, `artifacts/m2/scout/toolchain_support_matrix.json:8`, `scripts/m2_scout_hello_gpu.sh:8`, `tests/test_m2_scout_matrix.py:45`).
- note - GT4Py is correctly treated as a blocked scout outcome, not as evidence that the family is impossible. The matrix and narrative cite a concrete DaCe/SymPy/Python 3.13 failure, and the tester independently reproduced that failure while preserving the caveat that another Python/DaCe combination may work (`artifacts/m2/scout/toolchain_support_matrix.json:36`, `artifacts/m2/scout/toolchain_report.md:25`, `.agent/sprints/2026-05-19-m2-scout-blackwell-toolchain/tester-report.md:23`, `.agent/sprints/2026-05-19-m2-scout-blackwell-toolchain/tester-report.md:110`).
- note - Upstream URL liveness was not independently checked by the tester because that run had no web access; I verified the local version pins against the sprint venv and reran the local smoke/test suite. This is acceptable for this scout because the contract's hard acceptance criteria are local build/run evidence on the RTX 5090 (`.agent/sprints/2026-05-19-m2-scout-blackwell-toolchain/tester-report.md:110`, `.agent/sprints/2026-05-19-m2-scout-blackwell-toolchain/sprint-contract.md:85`).

## Contract Compliance

Pass. Acceptance criteria status:

- AC1-2 matrix structure: pass. Matrix has ISO timestamp, target hardware, and exactly six candidates in the required families (`artifacts/m2/scout/toolchain_support_matrix.json:1`, `artifacts/m2/scout/toolchain_support_matrix.json:8`).
- AC3-5 hello-GPU evidence: pass. Five non-blocked candidates have programs, output, and zero exit codes with device-side multiply-by-2 evidence; blocked GT4Py has a specific failure rationale and nonzero captured output (`artifacts/m2/scout/hello_gpu/jax/output.txt:1`, `artifacts/m2/scout/hello_gpu/triton/output.txt:1`, `artifacts/m2/scout/hello_gpu/kokkos/output.txt:1`, `artifacts/m2/scout/hello_gpu/cupy_or_numba/output.txt:1`, `artifacts/m2/scout/hello_gpu/cuda_tile/output.txt:1`, `artifacts/m2/scout/hello_gpu/gt4py/output.txt:1`).
- AC6 narrative report: pass. Report covers target hardware, all candidates in fixed order, and a closing recommendation, and is well under 2000 words (`artifacts/m2/scout/toolchain_report.md:5`, `artifacts/m2/scout/toolchain_report.md:7`, `artifacts/m2/scout/toolchain_report.md:61`).
- AC7 idempotence: pass. I reran `bash scripts/m2_scout_hello_gpu.sh`; it reported `5 pass, 0 fail`, and tracked scout artifacts remained content-clean.
- AC8-9 tests: pass. I reran `pytest -q`; result was `63 passed in 4.66s`.
- AC10-13 CI/hygiene: pass. `python scripts/validate_agentos.py` returned ok, `python scripts/check_m1_done.py` returned ok, `git diff --check main...HEAD` was clean, and no tracked scout file exceeds 100 KB.

## Correctness Risks

- GT4Py remains an environment-conditional blocked verdict, not a final technical exclusion. A follow-up Python 3.11/3.12 plus compatible DaCe/GT4Py scout is the right way to decide whether it stays in the bakeoff.
- CuPy evidence is array-expression evidence, not raw-kernel ergonomics evidence. That is inside this sprint's scope but should be addressed in the candidate implementation sprint.
- The CUDA C++ smoke program does not check CUDA API return codes; acceptable for a scout because the result and exit code passed, but future CUDA candidate code should fail explicitly on runtime errors.

## Performance Risks

- No profiler, occupancy, transfer-audit, launch-count, or bandwidth evidence is present. This is compliant because the sprint explicitly excludes performance measurement (`.agent/sprints/2026-05-19-m2-scout-blackwell-toolchain/sprint-contract.md:31`), but no performance claim should be made from these artifacts.
- Kokkos requires a local source build under `data/scratch`, so implementation sprints should budget toolchain setup time and keep build products out of git (`artifacts/m2/scout/toolchain_support_matrix.json:49`).

## Required Fixes

- None for sprint closeout.

## Decision

Decision: Accept
