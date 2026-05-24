# Sprint Contract — M6B0 CPU WRF Instrumentation Environment Audit (opus tester)

## Objective

Before the M6B0 codex worker patches WRF Fortran, **probe the build environment** to surface blockers early and serialize them into a single decision memo. This sprint is a bounded *tester/probe* role for opus, running in parallel with the M6B0 codex worker. It reduces the risk that the worker spends hours fighting toolchain issues that a 60-minute environment audit could have surfaced.

## Non-Goals

- NO modifications to the operational WRF build at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/`.
- NO new WRF compile (probe only — read configs, check tool versions, dry-run extraction).
- NO ADR promotion.
- NO commitment to a file format (the codex worker decides; this sprint just lists viable options).
- NO sub-sprint dispatch.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_envaudit` on branch `tester/opus/m6b0-wrf-instrumentation-env-audit`.

Write-only:
- `.agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/env_audit_memo.md` (deliverable)
- `.agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/proof_*.txt` (raw probe outputs)

Read-only everywhere else (and on the operational WRF build directory).

## Inputs

1. `.agent/sprints/2026-05-24-m6b0-wrf-savepoint-harness/sprint-contract.md` (the M6B0 contract this sprint is supporting)
2. `.agent/references/cpu-wrf-baseline.md`
3. `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/` (operational WRF — read-only audit only)
4. `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh` (build env script)
5. `/mnt/data/canairy_meteo/runs/wrf_l3/` (Gen2 d02 backfill — for size estimates)

## Acceptance Criteria

### Part 1: Toolchain probe

`env_audit_memo.md` reports each of these as DETECTED / NOT_DETECTED with version:

- `nvfortran` (NVIDIA HPC SDK) — version, path
- `gfortran` fallback — version, path (in case nvfortran build is fragile)
- `Serialbox` — present in any standard WRF build path? Or available via pip / module?
- HDF5 + libraries with Fortran bindings — version
- NetCDF + Fortran bindings — version
- MPI runtime — version, type (mpich / openmpi / hpcx)
- Python venv for the gpuwrf project — verify `jax`, `numpy`, `xarray`, `h5py`, `netCDF4` are importable in the project venv

Capture proof: `proof_toolchain.txt` (verbatim version dumps).

### Part 2: WRF source audit (read-only)

Without modifying anything:
- Locate `module_small_step_em.F` in the operational build's source tree
- Identify line ranges for the 8 operator boundaries the M6B0 contract names (`calc_coef_w`, `advance_w`, MUTS update, etc.)
- Note any compile-time options already enabled that would interfere with instrumentation (e.g., aggressive inlining, OpenACC pragmas, IPO)

Capture proof: `proof_wrf_source_audit.txt` (file paths + line ranges + relevant Makefile flags).

### Part 3: Storage budget estimate

Using Gen2 d02 backfill in `/mnt/data/canairy_meteo/runs/wrf_l3/`:
- Identify the typical d02 domain shape (nx, ny, nz)
- For an in-memory fp64 dump of 12 operators × all 3D state fields per acoustic step (≈ 10 fields × nx × ny × nz × 8 bytes), estimate the per-step savepoint size for: 1 column, 16×16 patch, 128×128 sub-domain, full d02
- Cross-check against `/tmp` free-space, `/mnt/data` free-space, and the worktree FS

Capture proof: `proof_storage_estimate.txt`.

### Part 4: Operational-build protection sanity check

Verify and document:
- sha256 of `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe`
- The exact path the instrumented build SHOULD live at (proposed: `/home/enric/src/wrf_gpu2/external/wrf_savepoint_patch/build/wrf.exe.instrumented`)
- Confirm /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/ is the canonical operational build (cross-check with Gen2 README and `.agent/references/cpu-wrf-baseline.md`)

Capture proof: `proof_operational_protection.txt`.

### Part 5: Recommendations memo

`env_audit_memo.md` ends with explicit recommendations for the M6B0 codex worker:
- Preferred file format (HDF5 vs NetCDF vs custom NPY) based on **available tooling + worker familiarity**, with rationale
- Whether Serialbox should be adopted (it is the standard for ICON/MPAS savepointing; would standardize the schema)
- Whether the patch should be a Fortran source patch, a Fortran wrapper module, or a preprocessor-macro-driven hook (read-only assessment of which is least invasive on the operational build)
- Hard requirement enumerated: the operational `wrf.exe` sha256 MUST remain identical after M6B0; spell out the verification procedure

### Part 6: No regression

`pytest --collect-only 2>&1 | tail -3` — verify no test files were touched.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_envaudit
which nvfortran && nvfortran --version 2>&1 | tee .agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/proof_toolchain.txt
which gfortran && gfortran --version 2>&1 | tee -a .agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/proof_toolchain.txt
python -c "import jax, numpy, xarray, h5py, netCDF4; print('OK')" 2>&1 | tee -a .agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/proof_toolchain.txt
find /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/ -name "module_small_step_em.F" 2>&1 | tee .agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/proof_wrf_source_audit.txt
sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe 2>&1 | tee .agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/proof_operational_protection.txt
df -h /tmp /mnt/data 2>&1 | tee .agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/proof_storage_estimate.txt
pytest --collect-only 2>&1 | tail -3 | tee .agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/proof_no_touch.txt
```

## Performance Metrics

N/A — opus probe.

## Proof Object

- `env_audit_memo.md`
- `proof_toolchain.txt`, `proof_wrf_source_audit.txt`, `proof_storage_estimate.txt`, `proof_operational_protection.txt`, `proof_no_touch.txt`
- Branch `tester/opus/m6b0-wrf-instrumentation-env-audit`

Time budget: **45–90 min**.

## Risks

- Accidental write to operational WRF directory: hard reject — all operations must be `ls`, `cat`, `find`, `sha256sum`, NEVER `cp` / `rm` / `chmod` on `/home/enric/src/wrf_gpu/`.
- Storage estimate confabulation: use real Gen2 d02 file headers, not assumed shapes.
- CPU budget: bound to cores 0-3 via taskset wrapper.

## Handoff Requirements

When all proofs + `env_audit_memo.md` committed on branch `tester/opus/m6b0-wrf-instrumentation-env-audit`: stop. Manager merges the memo into the M6B0 codex worker's context for Stage 1.
