# Sprint Contract — M6B0-R Fortran Hook ABI Follow-up (codex)

**Status:** Pre-drafted 2026-05-24 night. **Dispatch when:** M6B1 closes successfully AND a real-timestep-loop oracle is needed before M6B2 (instead of remaining on the Python WRF-shaped extractor).

## Objective

RELINK lane (commit `worker/gpt/m6b0r-relink-completion`) built a relinked WRF binary but the inherited `solve_em.F.patch` exposes zero-argument hooks and `savepoint_wrapper.F90` has empty hook bodies. This sprint changes the Fortran hook ABI to pass arrays + metadata, fills in the wrapper bodies for `calc_coef_w_pre/post`, and demonstrates a real timestep-loop HDF5 savepoint emission from the relinked binary.

This tightens the oracle from "Python WRF-source-shaped extractor" to "actual WRF Fortran in flight."

## Non-Goals

- NO modifications to operational `wrf.exe`. Pre/post sha256 check.
- NO modifications to canonical WRF source tree (patch a fresh copy).
- NO modifications to JAX side, comparator, or savepoint schema.
- NO multi-operator hook expansion in this sprint — `calc_coef_w` only.
- NO 1h forecast (short timestep-loop run only).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_hookabi` on branch `worker/gpt/m6b0r-fortran-hook-abi-followup`.

Write-only:
- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` — extend with real subroutine bodies that accept arrays + write HDF5
- `external/wrf_savepoint_patch/solve_em.F.patch` — replace zero-arg hook calls with real call-site instrumentation passing the operator's args
- `external/wrf_savepoint_patch/build_relinked.sh` — re-run (inherited from RELINK)
- `scripts/m6b0r_relinked_realemit_extract.py` (NEW)
- `.agent/sprints/2026-05-25-m6b0r-fortran-hook-abi-followup/` — proofs + worker-report

Read-only everywhere else.

## Acceptance Criteria

### Stage 1 — Hook ABI design (MANDATORY)

Decide and document the Fortran hook ABI: `SUBROUTINE sp_calc_coef_w_pre(ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte, mut, c1h, c2h, c1f, c2f, rdn, rdnw, top_lid)` (and corresponding `_post` with output arrays a/b/c/alpha/gamma).

### Stage 2 — Fill in wrapper bodies (MANDATORY)

Implement the Fortran subroutines in `savepoint_wrapper.F90` to write HDF5 files via the bundled HDF5 1.14.5 Fortran API (per the M6B0-R wrapper compile probe pattern).

### Stage 3 — Re-patch solve_em.F + rebuild (MANDATORY)

Update `solve_em.F.patch` to inject real-arg `CALL sp_calc_coef_w_pre(...)` immediately before the existing `CALL calc_coef_w(...)`. Same for `_post`. Re-run `build_relinked.sh`. Operational sha unchanged.

### Stage 4 — Run relinked WRF + verify real emission (MANDATORY)

Run the relinked WRF on the M6B0-R golden slice for 10 acoustic substeps. Verify HDF5 savepoint files are emitted from inside the timestep loop (not from a Python wrapper). Inspect with h5py — confirm shape, dtype, metadata.

### Stage 5 — Cross-check against Python extractor (MANDATORY)

Run `scripts/m6b0r_relinked_vs_shim_compare.py` on the real-emitted HDF5 files vs the Python-extracted ones. Expect bit-near (≤fp64 ULP) agreement at all interior cells. Discrepancies at top/boundary cells indicate residual extractor bugs or differences in input data shape.

### Stage 6 — Re-run JAX-vs-real-WRF parity (MANDATORY)

`scripts/m6b0r_jax_vs_wrf_compare.py --oracle relinked-real --operator calc_coef_w --tier all`. Acceptance: parity result unchanged from `FIRST-OPERATOR-PARITY-ACHIEVED` (commit `ac252e8`).

### Stage 7 — No regression + worker report

## Time budget: 8–16 h.
