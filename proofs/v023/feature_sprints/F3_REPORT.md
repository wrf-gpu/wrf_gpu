# F3 Report: CAM-UW Reference-Only / Fail-Closed Closure

## Objective

F3 investigated `bl_pbl_physics=9` CAM-UW PBL against a WRF-Fortran CAM-UW
column oracle, CPU-only. The target was not a JAX self-compare and not an
idealized finite-only scaffold check.

## Manager Ruling Applied

0:1 ruled after `F3_QUESTION.md` that the F3 result is honest and useful:

- The standalone WRF-Fortran CAM-UW column oracle is preserved as the future
  faithful-port gate.
- The existing CAM-UW JAX scaffold is proven RED vs that oracle.
- CAM-UW (`bl_pbl_physics=9`) is therefore `REFERENCE_ONLY` / fail-closed.
- The full faithful CAM-UW port is a separate milestone.

This report closes F3 under that scoped decision. It does not claim a working
CAM-UW port.

## Oracle Evidence

Proof object:
`proofs/v023/feature_sprints/camuw_oracle/camuw_current_jax_vs_wrf.json`

Summary:

```text
verdict=FAIL
jax_platform=cpu
jax_x64=True
worst_by_abs={'case': 1, 'field': 'pblh', 'max_abs': 1384.6212005615234, 'max_rel_to_oracle_scale': 7.46202053601862}
```

Pinned oracle run manifest:

```text
cpu_affinity=pid 629886's current affinity list: 4-31
jax_platforms=cpu
jax_platform_name=cpu
cuda_visible_devices=
omp_num_threads=2
objects=prebuilt pristine-WRF objects, default WRF REAL ABI
```

The measured failures are not roundoff-level and cannot justify operational
CAM-UW use. The worst `pblh` error is `1384.6212005615234 m`; other RED fields
include `tke`, `kvm`, `kvh`, `smaw`, and moisture increments.

## Implemented Seam

The codebase now treats CAM-UW as reference-only:

- `src/gpuwrf/io/scheme_catalog.py`: `bl_pbl_physics=9` is classified as
  `REFERENCE_ONLY`, not `IMPLEMENTED`.
- `src/gpuwrf/io/namelist_check.py`: ordinary namelist validation still accepts
  `bl_pbl_physics=9` for oracle/reference work; operational validation rejects
  it with the F3 CAM-UW reason and operational alternatives.
- `src/gpuwrf/runtime/operational_mode.py`: `9` is absent from
  `_SCAN_WIRED_OPTIONS["bl_pbl_physics"]`, with a named F3 unwired reason.
- `src/gpuwrf/coupling/physics_dispatch.py`: CAM-UW remains routable metadata
  but `gpu_runnable=False`, so suite summaries cannot mark it gate-ready.
- `src/gpuwrf/coupling/scan_adapters.py`: `PBL_SCAN_ADAPTERS` deliberately omits
  `9`; `camuw_pbl_adapter` raises `CamUwReferenceOnlyError`.
- `src/gpuwrf/physics/bl_camuw.py`: `camuw_columns` raises
  `CamUwReferenceOnlyError` before the old scaffold can run.

The default PBL path remains the existing operational MYNN path
(`DEFAULT_BL_PBL_PHYSICS=5`), and the operational PBL list is
`0/1/2/3/5/7/8/11/12/99`.

## Commands Run

Oracle proof command recorded in `F3_QUESTION.md`:

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= proofs/v023/feature_sprints/camuw_oracle/build_and_run.sh > proofs/v023/feature_sprints/camuw_oracle/camuw_taskset_run.log 2>&1
```

Reference-only seam verification:

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_v022_camuw_pbl.py tests/test_namelist_check.py tests/test_scheme_catalog_fail_closed.py tests/test_v060_physics_dispatch.py
```

Result:

```text
89 passed in 6.39s
```

The test command output is preserved at:
`proofs/v023/feature_sprints/f3_reference_only_tests.log`

## Proof Objects Produced

- `proofs/v023/feature_sprints/F3_QUESTION.md`
- `proofs/v023/feature_sprints/F3_REPORT.md`
- `proofs/v023/feature_sprints/F3_DONE`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_oracle_driver.f90`
- `proofs/v023/feature_sprints/camuw_oracle/wrf_runtime_stubs.f90`
- `proofs/v023/feature_sprints/camuw_oracle/build_and_run.sh`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_build_manifest.txt`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_case_1.json`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_case_2.json`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_case_3.json`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_current_jax_vs_wrf.json`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_taskset_run.log`
- `proofs/v023/feature_sprints/f3_reference_only_tests.log`

## Unresolved Risks

- A faithful CAM-UW JAX/GPU port is not implemented in F3.
- The preserved oracle used prebuilt pristine-WRF objects with the default WRF
  REAL ABI. It is sufficient to disprove the old scaffold, but the future
  faithful-port milestone should add a fresh FP64 WRF savepoint/oracle ladder.
- The CAM-UW carry contract (`tke_pbl`, `kvm3d`, `kvh3d`, `tauresx`, `tauresy`)
  is documented for the future port but not operationally threaded.

## Next Decision

Schedule CAM-UW as its own faithful-port milestone. Until that lands,
`bl_pbl_physics=9` is only for oracle/reference comparison and operational runs
must fail closed.
