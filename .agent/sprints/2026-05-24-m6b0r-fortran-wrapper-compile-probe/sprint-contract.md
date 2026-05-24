# Sprint Contract — M6B0-R Fortran Wrapper Compile Probe (opus tester, parallel)

## Objective

In parallel with the M6B0-R codex worker, probe whether the proposed Fortran wrapper module (`dyn_em/savepoint_wrapper.F90`, `#ifdef WRF_SAVEPOINT`) compiles cleanly **in isolation** against the canonical WRF source tree at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/`. Surface link/build issues early so the codex worker doesn't fight them after spending compute on the full WRF rebuild.

This is a **probe sprint** — write a minimal wrapper module skeleton, compile only that module standalone (or as a tiny test program), demonstrate the HDF5 Fortran API works under nvfortran 26.3, and produce a "feasibility verdict" memo.

## Non-Goals

- NO full WRF rebuild.
- NO modifications to `module_small_step_em.F` or `solve_em.F`.
- NO modifications to the operational `wrf.exe`.
- NO commitment to the wrapper module's final API — this is a feasibility probe only.
- NO sub-sprint dispatch.
- NO claim that the M6B0-R worker can skip its own Stage 1 build.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_wrapprobe` on branch `tester/opus/m6b0r-fortran-wrapper-compile-probe`.

Write-only:
- `.agent/sprints/2026-05-24-m6b0r-fortran-wrapper-compile-probe/wrapper_probe_memo.md` (deliverable)
- `.agent/sprints/2026-05-24-m6b0r-fortran-wrapper-compile-probe/probe_skeleton/` (tiny standalone module + test program)
- `.agent/sprints/2026-05-24-m6b0r-fortran-wrapper-compile-probe/proof_*.txt`

Read-only everywhere else.

## Inputs

1. `.agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/env_audit_memo.md` (R3 instrumentation strategy, R1 HDF5 choice, toolchain table)
2. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/sprint-contract.md` (Stage 1 — the wrapper module surface this probes)
3. `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh` (mandatory source)
4. `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F` (read-only reference)
5. NVHPC 26.3 docs (use `--help` and `man`; no web search needed)

## Acceptance Criteria

### Part 1 — Build a minimal `savepoint_wrapper.F90` skeleton

In `probe_skeleton/`:
- One module `savepoint_wrapper` with:
  - One subroutine `sp_write_real8_3d(name, units, stagger, rkstage, acstep, arr)` that writes a 3D `REAL*8` array to an HDF5 file using the HDF5 Fortran API.
  - Type definitions for the metadata struct (matching what the M6B0-R worker's schema will need).
- A minimal `program test_savepoint` that:
  - Allocates a 4×4×4 fp64 array
  - Calls `sp_write_real8_3d` to dump it to `/tmp/wrapprobe_test.h5`
  - Reads the file back with `h5py` (Python) and verifies shape + values

### Part 2 — Compile + link

```bash
source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
cd probe_skeleton/
make 2>&1 | tee ../proof_compile_log.txt
./test_savepoint 2>&1 | tee ../proof_run_log.txt
python verify_roundtrip.py 2>&1 | tee ../proof_h5py_verify.txt
```

All must succeed. Capture the Makefile and the linker line.

### Part 3 — Verdict memo

`wrapper_probe_memo.md` answers:
1. Did nvfortran 26.3 compile the wrapper module cleanly? Any warnings of concern?
2. Did the HDF5 Fortran API link without dependency mismatch against the WRF-bundled HDF5 1.14.5? Any libpath surprises?
3. Did the round-trip h5py read return the exact bytes the Fortran side wrote? Any endian / dtype surprises?
4. Estimated wall-time impact of the wrapper on a real WRF step (back-of-envelope: HDF5 file open + write + close per call × ~12 ops × ~6 substeps per step)?
5. GO / NO-GO recommendation to the M6B0-R codex worker.

### Part 4 — No regression

`pytest --collect-only 2>&1 | tail -3` — confirm nothing was touched outside the probe dir.

## Validation Commands

See Part 2.

## Performance Metrics

Per-call wall-time of one HDF5 dump (optional; for the M6B0-R worker's planning).

## Proof Object

- `wrapper_probe_memo.md`
- `probe_skeleton/` (compiles + runs)
- `proof_compile_log.txt`, `proof_run_log.txt`, `proof_h5py_verify.txt`, `proof_no_touch.txt`
- Branch `tester/opus/m6b0r-fortran-wrapper-compile-probe`

Time budget: **60–120 min**.

## Risks

- HDF5 Fortran API may need explicit `-I` for module files and `-L` for libs that env script doesn't add; probe explicitly captures the linker line.
- nvfortran may reject some HDF5 Fortran calls (rare); fallback: hand-rolled binary writer + Python loader (less elegant but works).
- DO NOT install or modify global packages — probe stays inside the worktree.

## Handoff Requirements

When all proofs on disk + memo committed on branch `tester/opus/m6b0r-fortran-wrapper-compile-probe`: stop. Manager merges memo into the M6B0-R worker's context for Stage 1.
