# Sprint Contract — M6B0-R Relink Completion: Apply solve_em.F.patch to Canonical WRF

## Objective

M6B0-R built `solve_em.F.patch`, `configure.wrf.patch`, `dyn_em/savepoint_wrapper.F90`, `namelist.savepoint`, and a small Fortran shim that proves the wrapper module compiles and emits HDF5. It did **not** apply the patches to a copy of the canonical WRF source tree and rebuild a full `wrf.exe.instrumented_relinked` that emits savepoints from within a real timestep loop.

This sprint closes that gap. The artifacts produced will tighten the M6B0-R oracle from "Python WRF-source reproduction" to "actual relinked WRF binary running on Canary d02 input." Once this lands, defect verdicts have the strongest possible WRF authority.

## Non-Goals

- NO modifications to operational `wrf.exe` at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/`. Pre/post sha256 enforced.
- NO modifications to canonical WRF source at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/`. Make a COPY first; patch the copy.
- NO modifications to `src/gpuwrf/dynamics/acoustic_wrf.py` (DEFECT-ANALYSIS lane is editing this).
- NO modifications to the comparator (`src/gpuwrf/validation/*`).
- NO redoing the M6B0-R Python-reproduction comparison — the deliverable is the relinked binary + at least one comparison from it.
- NO 1h forecast.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_relink` on branch `worker/gpt/m6b0r-relink-completion`.

Write-only:
- `external/wrf_savepoint_patch/source_copy/` (NEW; the WRF source copy that gets patched). DO NOT commit the copy itself — gitignore it.
- `external/wrf_savepoint_patch/build_relinked.sh` (NEW) — the build script that:
  1. Verifies operational sha256 (inherit from M6B0-R)
  2. Copies canonical WRF source to `source_copy/` (rsync, no symlinks)
  3. Applies `solve_em.F.patch` and `configure.wrf.patch` against `source_copy/`
  4. Sources `env_wrf_gpu.sh`
  5. Runs `./compile em_real` in `source_copy/`
  6. Verifies `source_copy/main/wrf.exe` exists with `-DWRF_SAVEPOINT` enabled
  7. Re-verifies operational sha256 unchanged
- `scripts/m6b0r_relinked_extract.py` (NEW) — orchestrator that runs the relinked WRF on the Canary d02 case from M6B0-R's golden-slice manifest
- `scripts/m6b0r_relinked_vs_shim_compare.py` (NEW) — sanity-check: relinked WRF savepoints vs the M6B0-R shim's emitted savepoints on the column tier; document any differences
- `.agent/sprints/2026-05-24-m6b0r-relink-completion/proof_*.txt`, `proof_*.json`
- `.agent/sprints/2026-05-24-m6b0r-relink-completion/worker-report.md`

Read-only:
- `src/gpuwrf/dynamics/` (DEFECT-ANALYSIS lane is editing acoustic_wrf.py)
- `src/gpuwrf/validation/` (locked)
- `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/` (inputs only — but read the savepoints/golden manifest)
- Canonical WRF source tree (NEVER write here)

## Inputs (mandatory)

1. `.agent/sprints/2026-05-24-m6b0r-relink-completion/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/worker-report.md`
3. `external/wrf_savepoint_patch/solve_em.F.patch`
4. `external/wrf_savepoint_patch/configure.wrf.patch`
5. `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90`
6. `external/wrf_savepoint_patch/namelist.savepoint`
7. `external/wrf_savepoint_patch/build.sh` (the shim build script; reference only)
8. `.agent/sprints/2026-05-24-m6b0-wrf-instrumentation-env-audit/env_audit_memo.md`
9. `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`
10. M6B0-R golden-slice manifest: `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints/golden/manifest.json`

## Acceptance Criteria

### Stage 1 — Source copy + patch apply (MANDATORY)

`build_relinked.sh`:
- Verifies operational sha256 == `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`
- `rsync -a /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/ external/wrf_savepoint_patch/source_copy/`
- Captures source git head from copy (must be `115e5756...`)
- Applies both patches with `patch -p1 < ...` from `source_copy/` root
- Re-captures git status after patch (expect modified files only, no rejected hunks)

Capture proofs: `proof_source_copy_sha256.txt` (per-file shas of the WRF tree before patch), `proof_patch_apply.txt`.

### Stage 2 — Full WRF compile with `-DWRF_SAVEPOINT` (MANDATORY)

- `source env_wrf_gpu.sh`
- Add `-DWRF_SAVEPOINT` to `CPP_OPTS` via the configure.wrf.patch
- `./compile em_real 2>&1 | tee proof_compile.txt`
- Expect 10–30 min wall time on this machine
- Verify `source_copy/main/wrf.exe` exists with sha256 different from operational
- Verify operational sha256 STILL unchanged (post-build check)

Capture proofs: `proof_compile.txt`, `proof_relinked_sha256.txt`, `proof_operational_unchanged_post_relink.txt`.

### Stage 3 — Run relinked WRF on golden slice (MANDATORY)

- Copy the Canary d02 IC + namelist set up that M6B0-R used for the golden slice
- Apply `namelist.savepoint` overrides (CPU operator path; savepoint output directory)
- Run the relinked WRF for 10 acoustic substeps (short test) on the golden domain
- Verify savepoint HDF5 files appear in the configured output dir
- Verify shape, dtype, units, stagger, RK index, acoustic substep index for each file

Capture proofs: `proof_relinked_run.txt`, `proof_relinked_savepoints_listing.txt`.

### Stage 4 — Cross-check: relinked WRF savepoints vs M6B0-R shim savepoints (MANDATORY)

`scripts/m6b0r_relinked_vs_shim_compare.py`:
- For the column tier (smallest, fastest), load M6B0-R shim savepoint for `calc_coef_w_pre` and the relinked WRF's emitted savepoint for the same operator at the same RK/acoustic indices
- Compare per-field max-abs delta
- Expectation: shim and relinked WRF should agree within fp64 ULP if both are reading the same input. Any larger discrepancy means the shim diverges from real WRF and the M6B0-R defect verdict needs re-examination.

Capture proof: `proof_shim_vs_relinked_delta.json`.

### Stage 5 — Re-run JAX-vs-relinked-WRF comparator (MANDATORY)

`scripts/m6b0r_relinked_extract.py` then re-run `scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier column --oracle relinked` (extend comparator with `--oracle` flag if needed).

If a/alpha/gamma deltas are similar to M6B0-R's Python-reproduction findings → defect is confirmed in JAX, dispatched to DEFECT-ANALYSIS lane.
If a/alpha/gamma deltas are different (smaller or larger) → the M6B0-R Python-reproduction was inaccurate; document and route to a follow-up sprint to align the reproduction.

Capture proof: `proof_jax_vs_relinked_calc_coef_w.json`.

### Stage 6 — No regression (MANDATORY)

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py -v
```
All PASS. New scripts may add tests but must not modify existing tests.

### Stage 7 — Worker report

`worker-report.md`: stages 1-6, build wall time, relinked sha, shim-vs-relinked delta, JAX-vs-relinked verdict for calc_coef_w, files changed, risks, handoff.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_relink
bash external/wrf_savepoint_patch/build_relinked.sh 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_build_relink.txt
python scripts/m6b0r_relinked_extract.py --tier column --steps 10 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_relinked_run.txt
python scripts/m6b0r_relinked_vs_shim_compare.py --tier column 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_shim_vs_relinked_delta.txt
python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier column --oracle relinked 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_jax_vs_relinked_calc_coef_w.txt
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py -v 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-relink-completion/proof_no_regression.txt
```

## Performance Metrics

- WRF rebuild wall time: report informational.
- Relinked WRF run wall time on golden slice: report informational.

## Kill Gates

- Patch apply fails with `.rej` files → stop, document, escalate; DO NOT manually rewrite the patch — fix the original `solve_em.F.patch` artifact instead.
- WRF compile fails with `-DWRF_SAVEPOINT` → if the failure is in `module_small_step_em.F` or `solve_em.F` (the instrumented files), fix the wrapper API; if elsewhere, the patch macro is leaking — fix the patch.
- Operational wrf.exe sha256 changes at any point → STOP, revert.
- Shim-vs-relinked delta is large at column tier (≥1e-6) → M6B0-R Python-reproduction was inaccurate; sprint outcome is `SHIM-DIVERGENCE-DETECTED`, escalate to manager.

## Risks

- Existing WRF Makefile may not pick up `dyn_em/savepoint_wrapper.F90` from a copy without adjusting `dyn_em/Makefile` — patch may need `Makefile` adjustment too.
- nvfortran may inline the wrapper subroutines too aggressively and surprise the HDF5 calls; add `!DEC$ NOINLINE` if needed (cite NVHPC docs).
- HDF5 file handles inside the timestep loop may exhaust descriptors at scale; for this sprint, the golden slice (small) is the safe target.
- CPU budget: cores 0-3 (taskset).

## Handoff Requirements

When all proofs + worker-report.md committed on branch `worker/gpt/m6b0r-relink-completion`: `/exit`. Manager merges and decides whether to re-baseline M6B0-R's defect verdict against the relinked oracle.

## Failure modes the manager will reject

- Modifying the canonical WRF source tree.
- Modifying operational wrf.exe.
- Committing the WRF source copy (must be gitignored).
- "Run successful" without the actual savepoint files inspected.
- Hidden namelist changes (CPU vs GPU operator path must stay CPU per ADR-025).
