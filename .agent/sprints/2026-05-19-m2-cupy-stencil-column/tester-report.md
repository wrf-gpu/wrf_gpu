# Tester Report

Role: tester / sonnet-test-engineer (Claude Opus 4.7, cross-AI verification of gpt-5.5 worker).
Branch: `tester/sonnet/m2-cupy-stencil-column`.

## Tests Added Or Run

Re-ran every command from the contract's Validation Commands list in a clean shell on the worker's branch (no edits to `src/` or `scripts/`):

- `bash scripts/m2_run_cupy.sh` — exit 0, prints `13000` (CUDA 13 runtime), idempotent on second run (no re-install, reuses `data/scratch/m2-cupy-venv/`).
- `python -m gpuwrf.validation.compare_fixture` for both manifests against `data/scratch/m2-cupy/{stencil,column}_out.npz` — `pass: true`, `first_failure: null` for every variable.
- `python -m json.tool` on both profile JSONs — valid JSON.
- `python scripts/validate_agentos.py` — `ok: true`.
- `python scripts/check_m1_done.py` — `ok: true`, no regression.
- `python scripts/check_m2_done.py` — CuPy candidate row satisfied; remaining `ok: false` is expected (other M2 candidates, ADR-001, M2 closeout not yet authored — out of this sprint's scope).
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5` — largest new file is `src/gpuwrf/backends/cuda_tile/host.cpp` (23658 B); no new >100 KB blobs introduced by this sprint.

Added `tests/test_m2_cupy_edge_cases.py` (23 tests, all passing) covering four layers:

1. **Profile-JSON schema & sanity bounds.** Required-keys set, types (int-but-not-bool), `wall_time_s ≤ 5`, `kernel_launches ≤ 5` (contract Performance Metrics), `registers_per_thread ≤ 64` stencil / `≤ 128` column, occupancy in `[0, 100]`. Explicit guard on `local_memory_bytes == 0` for column (AC #13) and a regression guard at `≤ 1024 B` for the stencil (which the contract allows non-zero for).
2. **Cross-AI invariants from the contract Handoff.**
   - (a) Kernel source strings contain `__global__`, `extern "C"`, `threadIdx`, and the modules use `cp.RawKernel`. A separate test scans for forbidden idiomatic-CuPy ops (`cp.matmul`, `cp.einsum`, `cp.ElementwiseKernel`, `cp.ReductionKernel`, …) — none are present.
   - (b) Re-compiled both RawKernels from the sprint venv via a sub-process and read `Function.attributes` independently. Observed values exactly match the worker's JSON: stencil `num_regs=58`, `local_size_bytes=64`; column `num_regs=24`, `local_size_bytes=0` (independent confirmation of AC #13).
   - (c) Walked `data/scratch/m2-cupy-venv/lib/python3.13/site-packages/cupy_cuda13x-*.dist-info` and asserted version == `14.0.1` with no other cupy distribution present; also scanned `scripts/m2_run_cupy.sh` to confirm the install line pins `cupy-cuda13x==14.0.1`.
3. **Internal consistency.** `achieved_bandwidth_gbps == host_device_transfer_bytes / wall_time_s / 1e9` (rel-tol 1e-3); `host_device_transfer_bytes` ≥ a fixture-size floor computed from the actual NPZ arrays (catches fabricated tiny numbers); all `artifact_paths` are relative and resolve under repo root; `deliberate_kernel_bug.txt` contains a real NVRTC compile-error line (not a "compiled successfully" placeholder); maintainability.md ≤ 300 words and mentions install/error/debug topics from AC #6.
4. **GPU-execution edge cases via the sprint venv.** Reproducibility (same input twice → bitwise-equal `phi_next` and `temperature_next`); `run_stencil`/`run_column` raise `KeyError` on NPZ missing a required key; `run_stencil` raises `FileNotFoundError` on a missing fixture path; `bench.py --problem column --skip-artifacts` honors both flags (no `stencil_run.json`, no `column_profile.json`).

## Results

- `pytest -q` — **111 passed in 17.88s** (88 prior baseline + 23 new edge cases).
- Contract round-trip: both `compare_fixture` runs pass with `max_abs_diff = 0.0`, `max_rel_diff = 0.0` on every variable for both fixtures.
- Independent kernel-attribute probe confirms: `stencil_advdiff_kernel` → `num_regs=58`, `local_size_bytes=64`, `binary_version=120` (sm_120 / Blackwell). `column_thermo_kernel` → `num_regs=24`, `local_size_bytes=0`, `binary_version=120`. Both within contract bounds.
- Venv: real `cupy-cuda13x==14.0.1` installed in the sprint venv with `cuda-pathfinder` and `numpy` as only transitives.
- Bandwidth math: stencil 0.7364 GB/s == 118272 / 0.000160601 / 1e9 (matches at rel-tol 1e-3); column 0.06610 GB/s == 2560 / 3.873e-5 / 1e9 (matches). Bandwidth is correctly labeled `fallback-derived` (ncu blocked locally by `ERR_NVGPUCTRPERM`, captured in `profiler_limitation`).
- Deliberate-kernel-bug capture: file present, contains the expected NVRTC `error: expected an expression` line — error-legibility claim in `maintainability.md` is grounded in real evidence.

## Fixtures Used

- `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml` + `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
- `fixtures/manifests/analytic-column-thermo-v1.yaml` + `fixtures/samples/analytic-column-thermo-v1.npz`
- Synthetic in-process derivatives (short-column NPZ, missing-key NPZ) generated in `tempfile.TemporaryDirectory` under `data/scratch/`; no on-disk artifacts persisted.

## Gaps

- **No real `ncu` metrics.** The Nsight Compute launch in `m2_run_cupy.sh` is invoked but rejected by the local kernel `ERR_NVGPUCTRPERM`. All occupancy / register / local-memory numbers therefore come from `Function.attributes` and the CUDA Occupancy API fallback; bandwidth is fallback-derived. This is the same limitation M2-S2 cuda_tile hit and the contract explicitly accepts it via the `profiler_limitation` field. Reviewer may want to revisit once the user grants performance-counter permission.
- **Column block size is hard-coded at 64.** The current column kernel launches `block=(64, 1, 1)` with `if (k >= levels) return`. It correctly handles `levels < 64` (verified manually), but I did not write a regression test that runs with a 128-level column because the worker's fixture is fixed at 40 levels and constructing a synthetic 128-level NPZ that the kernel can ingest would require building a one-off in-tree fixture. Acceptable for this sprint, but worth flagging when M2 picks a final candidate.
- **Idempotency proxy.** I confirmed the second `bash scripts/m2_run_cupy.sh` is fast and produces exit 0, but I did not strictly measure pip's "no-op" cost (the script uses a fast `cupy.__version__ == "14.0.1"` probe that short-circuits before pip is invoked, which is what the contract asked for).
- **Stencil non-zero `local_memory_bytes=64`.** Contract AC #13 only requires zero for the column kernel, so this is in-spec. Worker noted the same in `worker-report.md`. Reviewer may still want to ask the worker whether a small `tile`-buffer redesign could eliminate it.

## Decision: Accept

All contract Acceptance Criteria are satisfied. The CuPy raw-CUDA candidate is real (RawKernels with `extern "C" __global__` source), reproducible, correct against both M1 analytic fixtures, properly pinned to `cupy-cuda13x==14.0.1`, and reports register/local-memory/occupancy numbers that match an independent re-compilation. The cross-AI invariants the contract specifically asked the tester to verify all hold. Profiler-permission limitation is openly disclosed in the JSON and is the same limitation accepted in M2-S2. Recommend reviewer Accept.
