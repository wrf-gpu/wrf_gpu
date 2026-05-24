# M6B0 Environment Audit Memo (opus tester)

- **Sprint contract**: `.agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/sprint-contract.md`
- **Supports**: `.agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/sprint-contract.md` (codex worker)
- **Worktree**: `/tmp/wrf_gpu2_envaudit` (read-only audit; no compile, no run)
- **Branch**: `tester/opus/m6b0-wrf-instrumentation-env-audit`
- **Date**: 2026-05-24

---

## Part 1 — Toolchain probe

| Tool | Status | Version | Path / Note |
|---|---|---|---|
| `nvfortran` (NVHPC) | DETECTED via `env_wrf_gpu.sh` | NVHPC 26.3 (per env var `NVHPC_ROOT=.../nvhpc/Linux_x86_64/26.3`) | `/home/enric/src/canairy_meteo/Gen2/artifacts/nvhpc/Linux_x86_64/26.3/compilers/bin/nvfortran`. NOT on default `PATH` — codex worker MUST source `env_wrf_gpu.sh` before compiling. |
| `gfortran` fallback | **NOT_DETECTED** | n/a | Only `libgfortran5` runtime installed; no `gfortran` binary. If nvfortran build is fragile, codex must `apt install gfortran` OR reuse the conda env `/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build` (presumed-CPU gfortran). |
| HDF5 + Fortran | DETECTED (system) | 1.14.5 (conda) — `/home/enric/miniconda3/bin/h5cc`, `h5fc` | WRF build path uses its own bundled HDF5 at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_libs/` (per `LIB_EXTERNAL` in `configure.wrf`). Codex should re-use the bundled libs to match the operational build ABI. |
| NetCDF + Fortran | DETECTED | netCDF 4.9.3 (conda); WRF-bundled `libnetcdff` in `wrf_gpu_libs` | `nc-config` present; `nf-config` not on default `PATH` (in WRF-bundled tree only). |
| MPI runtime | DETECTED via NVHPC | NVHPC-bundled (`comm_libs/mpi/bin/{mpif90,mpirun}`) | Same caveat: must source `env_wrf_gpu.sh`. Type: HPC-X / OpenMPI flavor bundled with NVHPC. |
| Project Python venv | **No dedicated venv** | base conda Python 3.13.11 | `pyproject.toml` declares deps but no `.venv` exists in `/home/enric/src/wrf_gpu2/`. Codex must either create one OR continue using base conda. |
| `jax` | DETECTED | 0.10.0 | base conda |
| `numpy` | DETECTED | 2.4.4 | base conda |
| `xarray` | DETECTED | 2026.2.0 | base conda |
| `h5py` | DETECTED | 3.16.0 | base conda |
| `netCDF4` | DETECTED | 1.7.4 | base conda |
| `Serialbox` (ICON/MPAS savepoint lib) | **NOT_DETECTED** | n/a | Not installed in any standard path, not in pip. Adoption would require build-from-source — non-trivial. See recommendations. |

Raw probe: `proof_toolchain.txt`.

**Toolchain blockers**: none hard, but two **soft warnings**:
1. `env_wrf_gpu.sh` must be sourced for every `nvfortran` invocation; codex's build script must include it.
2. No `gfortran` fallback — if NVHPC build chokes on instrumented code (e.g., I/O calls in OpenACC region), debugging requires either installing gfortran or switching to NVHPC-only with `#ifndef _OPENACC` guards.

---

## Part 2 — WRF source audit (read-only)

### Source-of-truth identification

The operational `wrf.exe` at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` has provenance `git rev: 115e5756f98ee2370d62b6709baac6417d8f7338`. That git head **exactly matches** `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/` (`git log -1`: `115e5756... Add WSM6 active-path OpenACC boundary tightening`). **Canonical WRF source tree for M6B0**: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/`.

### `module_small_step_em.F` map (2089 lines, 80 KB)

| Operator | Line range | M6B0 hook? |
|---|---|---|
| `small_step_prep` | 16–290 | Stage 1 op #2 (MU/MUTS/ww start-of-step state) |
| `small_step_finish` | 295–434 | Stage 1 op #2 (end-of-step state) |
| `calc_p_rho` | 438–568 | Stage 1 op #6 (pressure/geopotential restore) |
| `calc_coef_w` | 570–652 | Stage 1 op #1 (**vertical-solve coefficients — PRIMARY M6B0 PARITY TARGET**) |
| `advance_uv` | 654–967 | not in 8-op list, but useful neighbor |
| `advance_mu_t` | 969–1175 | Stage 1 op #3/4 (`t_2ave`, `ph_tend`) |
| `advance_w` | 1178–1597 | Stage 1 op #5 (entry/exit) |
| `sumflux` | 1601–1761 | Stage 1 op #7/8 (full acoustic substep boundary) |
| `init_module_small_step` | 1765–1766 | stub |
| `advance_all` | 1770–1957 | optional |
| `save_ph_mu` / `restore_ph_mu` | 1961–2085 | Stage 1 op #6 (geopotential restoration) |

All eight M6B0 hook points have a clean Fortran SUBROUTINE container — instrumentation can be wrapper-based (call site instrumentation in `solve_em.F` around each `CALL`) rather than in-tree edits. See recommendations.

### Compile-time interference scan

- `!$acc` directives in `module_small_step_em.F`: **0** (pure CPU file).
- `!$omp` directives in `module_small_step_em.F`: **0**.
- `FCOPTIM = -O3 -acc -gpu=cc120` (configure.wrf:261). The flag enables OpenACC across the build, but this source file has no ACC pragmas — it compiles as host code.
- **Critical structural finding**: `dyn_em/Makefile` builds `module_small_step_em.o` **alongside** GPU companion files `small_step_gpu.F90` / `small_step_gpu2.F90` / `small_step_gpu3.F90` / `small_step_gpu4.F90` / `small_step_gpu5.F90`. `solve_em.F` `USE`s both module sets and calls `advance_w` (CPU) OR `advance_w_gpu` (GPU) based on runtime config flags. Concretely, `solve_em.F` contains call sites at lines: 2544/2572 (`small_step_prep`/`_gpu`), 2628/2645/2658 (`calc_p_rho`/`_gpu`), 2676/2692/2705 (`calc_coef_w`/`_gpu`), 3088/3108/3135 (`advance_uv`/`_gpu`), 3398/3419/3435 (`advance_mu_t`/`_gpu`), 3802/3837/3858/3880 (`advance_w`/`_gpu`), 4412/4430/4448 (`small_step_finish`/`_gpu`).
- **Implication for M6B0**: the codex worker must pick a **single code path** for first parity (recommend the CPU path — `module_small_step_em.F`), and the namelist used for the instrumented run must select CPU operators (or instrumentation hooks must guard against being called from both). The Gen2 operational build has BOTH paths; switching is via config.

Raw probe: `proof_wrf_source_audit.txt`.

---

## Part 3 — Storage budget estimate

### d02 dimensions (from real wrfout in `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/wrfout_d02_2026-05-22_00:00:00`)

- Unstaggered: `west_east=159`, `south_north=66`, `bottom_top=44` → 461 736 cells/field.
- Staggered (worst case for u/v/w): `160 × 67 × 45` → 482 400 cells/field.

### Per-step savepoint size (fp64, 10 fields × 12 ops)

| Scope | Bytes / acoustic step | Human |
|---|---|---|
| 1 column (1×1×45)         |       43 200 |    42 KB |
| 16×16 patch               |   11 059 200 |    10.5 MB |
| 128×128 sub-domain        |  707 788 800 |   675 MB |
| Full d02 (160×67×45)      |  463 104 000 |   442 MB |

### Per simulated hour (dt=18 s, 200 timesteps × 6 acoustic substeps)

| Scope | Per hour |
|---|---|
| 1 column                  |     50 MB |
| 16×16 patch               |     12.4 GB |
| 128×128 sub-domain        |    791 GB |
| Full d02                  |    517 GB |

### Filesystem headroom

| FS | Total | Free | Note |
|---|---|---|---|
| `/tmp` (tmpfs)            |  47 GB |  ~47 GB | volatile, RAM-backed; OK for scratch but not for long bundles |
| `/mnt/data`               | 2.8 TB | **348 GB (87 % full)** | shared with WRF baseline runs — competing demand |
| `/home` (root)            | 962 GB | **199 GB (79 % full)** | worktree FS |

### Recommendation to codex

- **Tier-1 (1-column)** and **Tier-2 (16×16 patch)** are trivially affordable — even an hour of full-acoustic-step capture is ≤ 12 GB.
- **Tier-3 full d02** at full acoustic rate is **NOT affordable** (517 GB/hr would consume `/home` headroom in 24 minutes). Codex must restrict Tier-3 to short pulses (e.g., 10 acoustic substeps = ~4.4 GB) or sub-sample (every Nth substep).
- Default dump location: `/home/enric/src/wrf_gpu2/external/wrf_savepoint_patch/savepoints/` (worktree-local, on `/home`). If size escapes, move to `/mnt/data/wrf_savepoints/` with explicit cleanup policy.

Raw probe: `proof_storage_estimate.txt`.

---

## Part 4 — Operational-build protection sanity check

| Item | Value |
|---|---|
| Operational `wrf.exe` path | `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` |
| Operational `wrf.exe` sha256 | `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37` |
| Sidecar `sha256.txt` agrees? | YES (identical hash) |
| File mode | `0775 -rwxrwxr-x` (writable by owner enric) — **soft risk: not write-protected** |
| Modify time | 2026-05-09 20:18:26 +0100 (untouched by this audit; sha unchanged at end of run — see `proof_no_touch.txt`) |
| Canonical-reference cross-check | `.agent/references/cpu-wrf-baseline.md:15` pins this exact path. Gen2 session log also references it. CONFIRMED canonical. |
| Proposed instrumented build path | `/home/enric/src/wrf_gpu2/external/wrf_savepoint_patch/build/wrf.exe.instrumented` (path does **not** yet exist — clean to create). |

**Hard requirement spelled out for codex (must appear in ADR-025 and in the M6B0 build script)**:

```bash
# Pre-build snapshot
EXPECTED_SHA=1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37
test "$(sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe | cut -d' ' -f1)" = "$EXPECTED_SHA" \
  || { echo "FATAL: operational wrf.exe sha changed BEFORE M6B0 build"; exit 1; }

# ... do M6B0 build into /home/enric/src/wrf_gpu2/external/wrf_savepoint_patch/build/ ...

# Post-build verification
test "$(sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe | cut -d' ' -f1)" = "$EXPECTED_SHA" \
  || { echo "FATAL: operational wrf.exe was modified during M6B0 build — REVERT"; exit 2; }
```

**Recommended hardening (optional but cheap)**: before M6B0 build, `chmod a-w` the operational `wrf.exe` so accidental `cp -f` or wrong `make install` cannot clobber it. Codex worker should NOT do this without manager sign-off (it changes file mode).

Raw probe: `proof_operational_protection.txt`.

---

## Part 5 — Recommendations to the M6B0 codex worker

### R1 — Savepoint file format: **HDF5** (default)

Rationale:
- `h5py` 3.16.0 is available, well-known, supports compound dtypes for the savepoint metadata, supports compression (gzip / blosc) which matters for 16×16 patches at GB/hour rate.
- WRF-bundled HDF5 1.14.5 already linked into the operational binary; Fortran-side write from the instrumented `wrf.exe` reuses the same libraries (no extra dependency drag).
- HDF5 is also the Serialbox storage layer, so "switch to Serialbox later" is a non-breaking upgrade.

Alternatives considered:
- **NetCDF**: also available (`netCDF4` 1.7.4, WRF-bundled), but CF conventions clash with savepoint metadata (free-form attrs, run-IDs, RK indices) and NetCDF4-classic mode (per `env_wrf_gpu.sh: NETCDF_classic=1`) restricts feature use.
- **NPY bundle**: trivial to write from numpy but very weak metadata support; Fortran write side has no native NPY emitter (would need ad-hoc binary protocol). **Reject.**

### R2 — Serialbox: **DO NOT adopt in this sprint**

Reasoning:
- Not installed in any standard path (`/opt`, `/usr/local`, `pip`, conda).
- Build-from-source has its own Boost dependency and Fortran-binding gen step — out of scope for a 1-week sprint focused on first parity.
- HDF5 + a minimal Python-side schema dataclass gives 80 % of Serialbox's value (schema, metadata, replay) in <500 LOC.
- Re-evaluate at M6B2 if comparator infrastructure grows past 1500 LOC.

### R3 — Instrumentation strategy: **Fortran wrapper module at call sites in `solve_em.F`**, not in-tree edits to `module_small_step_em.F`

Reasoning:
- Each of the 8 operators is a clean SUBROUTINE with explicit arg lists (Part 2 audit). Wrapping each `CALL X(...)` with `CALL X_with_savepoint(...)` requires touching only `solve_em.F` and adding one new file `dyn_em/savepoint_wrapper.F90`.
- Source-patch-only approach minimizes diff size against `git rev 115e5756...`, simplifies code review, simplifies eventual upstream-back-merge.
- Compile-time gating: surround the new `CALL`s with `#ifdef WRF_SAVEPOINT` / `#endif` and add the macro to the instrumented build's `configure.wrf` only — guarantees zero impact when the macro is absent.
- A preprocessor-only approach (insert savepoint calls via `cpp` macros) is more elegant but more fragile under WRF's existing `cpp` pipeline; defer.

Reject alternatives:
- **In-tree patch of `module_small_step_em.F`**: blows up the diff and risks accidental ABI changes to subroutine signatures.
- **Pure preprocessor hook**: hard to debug, harder to grep, breaks WRF's existing tools/standard.sed pipeline.

### R4 — Operational `wrf.exe` immutability (HARD REQUIREMENT)

Worker MUST embed the pre/post sha256 check from Part 4 into the M6B0 build script. The CI / handoff acceptance criterion is: `sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` returns `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37` after the M6B0 build completes. No exceptions.

### R5 — Code-path selection (CPU vs GPU)

The operational binary contains BOTH `module_small_step_em` (CPU) and `small_step_gpu*` (GPU companion) paths, selected at runtime by config flags. M6B0 first-parity target is the CPU path (`module_small_step_em.F`). The codex must:
- Verify the M6B0 instrumented run uses a namelist that routes through CPU operators (not GPU companions).
- Document the selector flag in ADR-025 so M6B1 can decide whether to also instrument `_gpu` variants.

### R6 — Toolchain reuse

- Codex MUST source `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh` before invoking `nvfortran` or the WRF `compile` script.
- For the Python comparator side, base conda Python 3.13 with current jax/numpy/xarray/h5py/netCDF4 is sufficient — no new venv required this sprint. If isolation is desired, codex can create `/home/enric/src/wrf_gpu2/.venv` from `pyproject.toml`.
- CPU pinning: codex Fortran build is fine on cores 4–31 (Claude/Python work is on 0–3). The `compile` job can use `make -j 8` without violating the core budget; if conflicts arise, gate `make -j` via `taskset -c 4-15 make -j 8`.

### R7 — Storage policy

- Default dump scope: 1 column + 16×16 patch (combined < 15 GB / hour) — fits comfortably.
- Tier-3 full d02 must be short-pulse only (≤ 10 acoustic substeps per capture).
- Output dir: `/home/enric/src/wrf_gpu2/external/wrf_savepoint_patch/savepoints/` initially; auto-switch to `/mnt/data/wrf_savepoints/` if `/home` falls below 100 GB free.

---

## Part 6 — No-regression evidence

- `pytest --collect-only`: 579 tests collected, no errors. Test set unchanged. See `proof_no_touch.txt`.
- `git status --short` in worktree shows ONLY new files under `.agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/`.
- Operational `wrf.exe` sha256 verified unchanged at audit end: `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`.

---

## AGENT REPORT

**GO** for the M6B0 codex worker. Environment is ready to patch WRF Fortran with the following caveats: (1) `env_wrf_gpu.sh` must be sourced for every NVHPC invocation (nvfortran/MPI are not on default PATH); (2) no `gfortran` fallback exists, so codex should not assume a portable build — pin NVHPC 26.3; (3) Serialbox is **not** available and should **not** be adopted this sprint — use plain HDF5 via `h5py` (DETECTED 3.16.0) and a Python-side schema dataclass; (4) instrument via a **Fortran wrapper module called from `solve_em.F` call sites**, gated by `#ifdef WRF_SAVEPOINT`, not via in-tree edits to `module_small_step_em.F` — all 8 operator boundaries map cleanly to existing SUBROUTINEs at known line ranges; (5) the operational `wrf.exe` (`/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe`, sha `1ec3815...d0dd37`) is the canonical baseline and MUST remain byte-identical — pre/post sha checks belong in the M6B0 build script; (6) **the operational binary has both CPU and GPU operator paths (the `small_step_gpu*` companion files) — codex must pick the CPU code path for first parity and document the namelist switch in ADR-025**; (7) storage budget is comfortable for column + 16×16 patch, but Tier-3 full d02 at full acoustic rate is infeasible (517 GB/hr) — restrict to short pulses or sub-sample. No hard blockers identified. The codex worker can proceed with Stage 1.
