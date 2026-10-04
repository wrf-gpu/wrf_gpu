# F3 QUESTION: CAM-UW PBL WRF-Fortran Column Oracle Not Passed

## Objective

F3 objective was CPU-only column-oracle bit identity / tight parity for `bl_pbl_physics=9` CAM-UW PBL against WRF-Fortran CAM-UW, not a JAX self-compare and not an idealized finite-only scaffold check.

## Verdict

`F3_DONE` is not written. `F3_REPORT.md` is not written. The current JAX CAM-UW path does not satisfy the F3 acceptance contract.

The proof artifact `camuw_current_jax_vs_wrf.json` has:

```text
verdict=FAIL
jax_platform=cpu
jax_x64=True
worst_by_abs={'case': 1, 'field': 'pblh', 'max_abs': 1384.6212005615234, 'max_rel_to_oracle_scale': 7.46202053601862}
```

## Proof Objects

- `proofs/v023/feature_sprints/camuw_oracle/camuw_oracle_driver.f90`
- `proofs/v023/feature_sprints/camuw_oracle/wrf_runtime_stubs.f90`
- `proofs/v023/feature_sprints/camuw_oracle/build_and_run.sh`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_build_manifest.txt`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_case_1.json`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_case_2.json`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_case_3.json`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_current_jax_vs_wrf.json`
- `proofs/v023/feature_sprints/camuw_oracle/camuw_taskset_run.log`

The manifest records:

```text
cpu_affinity=pid 629886's current affinity list: 4-31
jax_platforms=cpu
jax_platform_name=cpu
cuda_visible_devices=
omp_num_threads=2
objects=prebuilt pristine-WRF objects, default WRF REAL ABI
```

## Commands Run

Final pinned oracle command:

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= proofs/v023/feature_sprints/camuw_oracle/build_and_run.sh > proofs/v023/feature_sprints/camuw_oracle/camuw_taskset_run.log 2>&1
```

Expected final exit after fail-closed harness correction: non-zero, because `compare_current_jax.py` writes the FAIL JSON and exits non-zero when parity is not met.

JSON summary extraction:

```bash
taskset -c 4-31 python3 - <<'PY'
import json
p='proofs/v023/feature_sprints/camuw_oracle/camuw_current_jax_vs_wrf.json'
d=json.load(open(p))
print(d['verdict'])
print(d['jax_platform'], d['jax_x64'])
print(d['worst_by_abs'])
PY
```

Earlier exploratory scaffold pytest command was interrupted after one passing test before this final pinned proof run:

```bash
env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_v022_camuw_pbl.py
```

That pytest is not a WRF-Fortran CAM-UW parity proof and was not used for the verdict.

## Numerical Failures

Against the standalone WRF-Fortran CAM-UW oracle linked from pristine WRF objects, the current JAX scaffold differs by:

- Case 1 `unstable_moist_cloud`: `pblh` max abs `1384.6212005615234 m`, `tke` max abs `0.48273907527317655`, `kvm` max abs `451.287483412013`, `kvh` max abs `609.8479505567743`, `smaw` max abs `4.964`.
- Case 2 `stable_nocturnal`: `pblh` max abs `411.77415466308594 m`, `tke` max abs `0.19823383132761668`.
- Case 3 `neutral_low_cloud`: `pblh` max abs `614.4441986083984 m`, `tke` max abs `0.22242188136306412`, `kvm` max abs `37.063344900736`.

These are not roundoff-level errors and are not close to column bit identity.

## Diagnosis

The existing `src/gpuwrf/physics/bl_camuw.py` is a v0.22 CAM-style scaffold. It is useful for finite, shape, and routing tests, but it is not a faithful transcription of WRF CAM-UW.

The WRF CAM-UW driver wraps CAM routines and state that are not represented by the scaffold: `compute_eddy_diff`, `compute_vdiff`, pseudo-conservative moisture handling, CAM cloud-number constituent registration, residual stress state (`tauresx`, `tauresy`), previous-step `kvm3d`/`kvh3d`, `tke_pbl`, `smaw`/`turbtype`, and shallow-convection coupling state. Matching the oracle is a real port/transcription task, not a constant-tuning task.

This harness is sufficient to disprove the current JAX endpoint. It is not a success oracle for F3 because it uses prebuilt pristine-WRF objects with the default WRF REAL ABI rather than a dedicated fresh full-FP64 WRF savepoint build.

## Question / Next Decision Needed

Recommended decision: do not mark F3 complete. Assign a larger dedicated CAM-UW port sprint to Fable/manager, including a frozen CAM-UW carry-state interface and a fresh FP64 WRF-source oracle/savepoint ladder. Alternative: explicitly re-scope `bl_pbl_physics=9` as reference-only / fail-closed until that port exists.
