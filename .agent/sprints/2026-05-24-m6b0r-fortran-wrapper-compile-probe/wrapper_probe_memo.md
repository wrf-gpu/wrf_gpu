# Wrapper Probe Memo — M6B0-R Fortran Wrapper Compile Probe

Worker: opus tester
Branch: `tester/opus/m6b0r-fortran-wrapper-compile-probe`
Worktree: `/tmp/wrf_gpu2_wrapprobe`
Date: 2026-05-24

## Verdict: GO

The HDF5 1.14.5 Fortran API compiles and links cleanly under nvfortran 26.3 against the WRF-bundled libraries. A 4×4×4 fp64 array plus 3 string and 3 integer attributes round-trips bit-exact through h5py. **No surprises that block Stage 1 of the M6B0-R worker.** Two minor gotchas documented below — both are caller-side, not toolchain-side.

## Q1. Did nvfortran 26.3 compile the wrapper module cleanly? Any warnings of concern?

Yes — clean. Zero warnings. Single-pass build of both files plus link. Module file `savepoint_wrapper.mod` and HDF5 module files in `$HDF5/include/` resolved without any `-module` or `-I` adjustment beyond the one explicit `-I$(HDF5)/include` flag in the Makefile.

The HDF5 modules used: `hdf5` (umbrella). Internally pulls `h5lib`, `h5f`, `h5d`, `h5s`, `h5t`, `h5a`. All present in `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_libs/include/` (verified `hdf5.mod`, `h5fortran_types.mod`, etc.).

Compile flags (matching WRF's idiom):

```
nvfortran -O0 -g -Mfreeform -Mbackslash -I$(HDF5)/include -c savepoint_wrapper.F90
```

`-O0 -g` is debug; the wrapper subroutines are HDF5-bound and not in any hot path, so optimisation level is immaterial.

## Q2. Did the HDF5 Fortran API link without dependency mismatch against the WRF-bundled HDF5 1.14.5? Any libpath surprises?

Yes — clean link, **first try**, with the linker line:

```
nvfortran -O0 -g -Mfreeform -Mbackslash \
  -I/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_libs/include \
  -o test_savepoint savepoint_wrapper.o test_savepoint.o \
  -L/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_libs/lib \
  -lhdf5_fortran -lhdf5 -lz -ldl -lm
```

Key observations:

- HDF5 1.14.5 in `wrf_gpu_libs/lib` is **static-only** (`libhdf5.a`, `libhdf5_fortran.a`; no `.so`). `libhdf5.la::dependency_libs` confirms the minimal transitive set is `-lz -ldl -lm`. No `-lsz`, `-lcurl`, `-lcrypto` needed (HDF5 was built with neither szip nor S3/ROS3 enabled).
- `libhdf5.settings` confirms the Fortran library was built with the **same** nvfortran 26.3 (`-tp znver5`) used by this probe — no ABI mismatch risk.
- The env script's `LDFLAGS`/`LIBS` exports happen to match what we need almost verbatim; the M6B0-R worker can lift `$(LDFLAGS) $(LIBS)` straight from `env_wrf_gpu.sh` (after dropping `-lnetcdff -lnetcdf` for the wrapper-only test).

No libpath surprises. No `-rpath` needed (static linking).

## Q3. Did the round-trip h5py read return the exact bytes the Fortran side wrote? Any endian / dtype surprises?

Yes — **bit-exact** equality on a deterministic fill (`arr(i,j,k) = 100*i + 10*j + k`). Endianness: file is little-endian (matches x86-64 host); h5py reports `dtype.byteorder == '<'`. All 6 metadata attributes (3 strings, 3 ints) read back exactly.

**Two caller-side gotchas the M6B0-R worker MUST handle in the comparator:**

1. **Axis order**. HDF5 preserves the declared dataspace shape `[d1,d2,d3]` and writes the underlying memory contiguously. Fortran is column-major; numpy is row-major. A Fortran `arr(NX,NY,NZ)` appears in h5py with `shape == (NX,NY,NZ)` but `numpy[a,b,c] == fortran arr(c+1, b+1, a+1)` (axes reversed). The comparator must either:
   - Transpose `numpy_array.transpose(2,1,0)` to align with the Fortran indexing the WRF code uses, **or**
   - Build the JAX reference in the same memory-contiguous layout (preferred — zero overhead).
   This is documented at the top of `probe_skeleton/verify_roundtrip.py`.

2. **Attribute types**. `h5py.Dataset.attrs['rkstage']` returns a length-1 numpy array, not a scalar. `attrs['name']` returns either a 0-D `numpy.bytes_` or a bytes scalar depending on h5py version. The comparator must coerce with `int(arr.ravel()[0])` and `bytes.decode().strip()` (see `_attr_to_int` / `_attr_to_text` helpers).

No dtype surprises: REAL(8) ↔ float64 ↔ H5T_NATIVE_DOUBLE all native fp64.

## Q4. Estimated wall-time impact of the wrapper on a real WRF step

Per-call cost of one `open → write → close → all-attrs` HDF5 dump:

| Filesystem | Array size                 | Wall-time per call | Effective MB/s |
|------------|----------------------------|---------------------|----------------|
| tmpfs (/tmp) | 4×4×4 fp64 (512 B)        | 58 µs               | n/a (overhead-bound) |
| NVMe (ext4)  | 4×4×4 fp64 (512 B)        | 97 µs               | n/a (overhead-bound) |
| NVMe (ext4)  | 64×40×44 fp64 (~900 KB)   | 2 116 µs            | 426 MB/s       |

Pinning: `taskset -c 0-3` on all measurements.

Back-of-envelope for the M6B0-R operator set:

- 12 operator boundaries × 6 acoustic substeps = **72 calls per RK step**, then × ~3 RK stages ≈ **216 calls per WRF step**.
- At Tier-3 array size (~900 KB/call) on NVMe: 216 × 2.1 ms ≈ **0.45 s/step** of I/O.
- At Tier-1 column / Tier-2 16×16 (small arrays): 216 × 0.1 ms ≈ **0.02 s/step**.

For a CPU WRF step at d02 resolution that normally runs ~1–3 s/step (unverified — Codex worker should benchmark), Tier-3 I/O is **~15–45 % overhead**, well within budget for a correctness sprint. Tier 1 and Tier 2 are **negligible** (<2 %).

**Optimisation lever if needed**: reuse a single open file across all 216 calls per step (one `H5Fopen` outside the substep loop; one `H5Dcreate` per call inside) — would drop the open/close ~50 µs overhead per call, gaining ~10–20 % at Tier-3. Not required for M6B0-R; flagged for M6B1+ if I/O becomes the bottleneck.

## Q5. GO / NO-GO recommendation to the M6B0-R codex worker

**GO.** The probe answers all five Stage-1 risk questions affirmatively:

1. nvfortran 26.3 compiles `hdf5`-module-based Fortran cleanly (no warnings, no ABI issues).
2. Static link against the WRF-bundled HDF5 1.14.5 + libz works first-try.
3. h5py reads the file the Fortran code writes, byte-exact, with all metadata.
4. Per-call cost is acceptable (<0.5 s/step at Tier-3, <30 ms at Tier-1/2).
5. No fallback to a hand-rolled binary writer needed — the HDF5 Fortran API works.

**Things the M6B0-R worker can lift directly**:

- The `savepoint_wrapper.F90` module skeleton in `probe_skeleton/` — the subroutine surface, type signature `sp_metadata_t`, and the `write_str_attr` / `write_int_attr` helpers are reusable as-is. Add the additional metadata fields (`operator`, `wrf_commit`, `namelist_hash`, `dt`, `domain_idx`, map factors, vertical-grid params) as additional attributes by extending the same pattern.
- The Makefile linker line.
- The two caller-side gotchas (axis order, attribute coercion) — fold into the comparator helpers from day 1.

**Risks the worker still owns** (NOT covered by this probe):

- Integration with WRF's `configure.wrf` / `compile em_real`. The probe builds in isolation; the full WRF build environment may layer additional `CPP_OPTS`, `Werror`, or `-Mextend` flags that change the picture. Mitigation: keep the wrapper module `Mfreeform` and use only standard F2008 syntax (the probe code does).
- `solve_em.F.patch` correctness — unrelated to the I/O probe.
- Tolerance-ladder and parity correctness — that's Stage 4/5, not Stage 1.

## Files Delivered

- `probe_skeleton/savepoint_wrapper.F90` — the module
- `probe_skeleton/test_savepoint.F90` — driver + per-call wall-time loop
- `probe_skeleton/verify_roundtrip.py` — h5py round-trip checker
- `probe_skeleton/Makefile`
- `proof_compile_log.txt` (full make output)
- `proof_run_log.txt` (test_savepoint stdout incl. perf)
- `proof_h5py_verify.txt` (verifier output — 9 OKs, 0 FAILs)
- `proof_no_touch.txt` (pytest collect — 583 tests, no errors)

## AGENT REPORT

Built a 4-file standalone Fortran-HDF5 probe (`savepoint_wrapper.F90` module + `test_savepoint` driver + Python verifier + Makefile) and compiled it against the WRF-bundled HDF5 1.14.5 with nvfortran 26.3 using only the libs and flags exported by `env_wrf_gpu.sh`. Compile, link, and run were clean first-try; h5py reads the Fortran-written file bit-exact, including 3 string and 3 integer attributes. Verdict is **GO** for M6B0-R Stage 1: the Fortran-HDF5 path is viable, the linker line is `-L$(HDF5)/lib -lhdf5_fortran -lhdf5 -lz -ldl -lm` (static, no rpath), and per-call wall time is 97 µs (4×4×4) to 2.1 ms (64×40×44) on NVMe — Tier-3 overhead estimated at 15–45 % of WRF step time, acceptable for a correctness sprint. Two caller-side gotchas surfaced (Fortran→numpy axis reversal; h5py attrs return length-1 ndarrays not scalars) and are flagged for the comparator. The codex worker can lift `savepoint_wrapper.F90` and the Makefile directly into `external/wrf_savepoint_patch/dyn_em/`.
