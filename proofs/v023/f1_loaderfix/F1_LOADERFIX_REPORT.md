# F1 Loaderfix Report

## Verdict

SUCCESS. The F1 B>1 loader failure was bounded to static-aux canonicalization for identical lanes. The fix re-enables `GPUWRF_BATCH_ENSEMBLE>1` while keeping the default unset/B=1 path byte-identical to the v0.22.2 reference.

The warm Tenerife B-sweep shows useful batching value: B=1 reaches 4.51M weighted cells-in-flight/sec, B=4 reaches 5.63M, and B=5 plateaus at 5.62M. Per manager comparison, B=4 is 97% of the 5.81M large-cell ceiling. The original 1.5x linear throughput target was not reached; the value is GPU saturation on the starved fixture, not linear case scaling.

## Root Cause

Diagnosis verdict: BOUNDED.

Identical same-geometry lanes were loaded independently and produced equal runtime values but different JAX treedefs for `d01`/`d02` namelist static aux. The mismatch came from per-load static holder object identity in fields such as `noahmp_static`. `_assert_homogeneous_batch()` correctly failed closed with:

`batch lane 1 d01: namelist/static treedef differs`

The assertion was not weakened. Instead, B>1 batch loading now value-canonicalizes selected static namelist fields so equal static aux values share the same object before the homogeneous-batch treedef check.

## Fix

Changed `src/gpuwrf/integration/nested_pipeline.py`:

- accepts positive integer `GPUWRF_BATCH_ENSEMBLE` again;
- canonicalizes batch static aux by value only inside the B>1 batch-load path;
- keeps `_assert_homogeneous_batch()` fail-closed for real lane differences;
- adds B>1 batched operational domain-tree execution;
- keeps unset/B=1 delegated to the existing singleton operational runner path.

Regression tests:

- `tests/test_operational_namelist_cache_key.py` covers value-canonical static aux treedef equality and B env parsing.
- `tests/test_v023_batched_minigrid.py` covers the updated B=2 CPU real-dycore gate.

## Default Byte Identity

Default path validation against the v0.22.2 reference is green.

Proof objects:

- `proofs/v023/f1_loaderfix/default_gate_final/v023_f1loaderfix_default_final_state_digest.json`
- `proofs/v023/f1_loaderfix/default_gate_final/v023_f1loaderfix_default_final_summary.json`
- `proofs/v023/f1_loaderfix/DEFAULT_FINAL_REFCACHE_FIELD_COMPARE.json`

Result:

- digest: `8b5b4405699d0f67ac46a4c8977d201482cdbce3635417cbee944e67387345ac`
- leaves: 178
- bytes: 367594272
- field compare: `exact=true`
- mismatched leaves: `0/178`
- `max_abs_global=0.0`

Gate command:

```bash
scripts/with_gpu_lock.sh --timeout 7200 --label v023-f1fix -- env -u GPUWRF_BATCH_ENSEMBLE -u GPUWRF_SINGLE_SCAN PYTHONPATH=src GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF JAX_PLATFORMS=cuda JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=<USER_HOME>/src/wrf_gpu2_wt/v023-refcheck/proofs/v023/refcheck/cache GPUWRF_CACHE=<USER_HOME>/src/wrf_gpu2_wt/v023-refcheck/proofs/v023/refcheck/cache XLA_PYTHON_CLIENT_PREALLOCATE=false GPUWRF_NESTED_AOT=1 GPUWRF_NESTED_PARALLEL_COMPILE=0 GPUWRF_NESTED_DEFUSE_COMPILE=0 CANARY_DUMP_STATE=1 CANARY_FIXED_REPEATS=2 CANARY_MAXDOM=3 CANARY_SINGLE_SHAPE_TIMING=1 CANARY_FIXED_ROOT_STEPS=2 CANARY_INPUT_DIR=<DATA_ROOT>/wrf_downscale/runs/20250121/cpu CANARY_ARTIFACT_DIR=proofs/v023/f1_loaderfix/default_gate_final CANARY_TAG=v023_f1loaderfix_default_final python -u proofs/v023/gpu_gates/canary_state_capture.py
python proofs/v023/gpu_gates/canary_state_compare.py --ref-npz <USER_HOME>/src/wrf_gpu2_wt/v023-refcheck/proofs/v023/gpu_gates/canary_state/v0222_cap_state.npz --ref-meta <USER_HOME>/src/wrf_gpu2_wt/v023-refcheck/proofs/v023/gpu_gates/canary_state/v0222_cap_state_meta.json --cand-npz proofs/v023/f1_loaderfix/default_gate_final/v023_f1loaderfix_default_final_state.npz --cand-meta proofs/v023/f1_loaderfix/default_gate_final/v023_f1loaderfix_default_final_state_meta.json --output proofs/v023/f1_loaderfix/DEFAULT_FINAL_REFCACHE_FIELD_COMPARE.json
```

## B Sweep

Raw JSONL deliverable: `proofs/v023/f1_loaderfix/f1_bench.json`.

Convention:

`cells/sec = B * (d01_3Dcells*d01_steps_per_measure + d02_3Dcells*d02_steps_per_measure) / warm_wall_s`

Fixture:

- Tenerife real 3 km / 1 km: `<DATA_ROOT>/wrf_downscale/f1_retest/tenerife_20250121_3km1km/real_run`
- d01 3D cells: 158400
- d02 3D cells: 309672
- d01 measured root steps: 400
- d02 measured child steps: 1200

| B | warm_wall_s | peak_vram_mib | fits | total cells/sec | per-lane cells/sec | vs B=1 |
|---:|---:|---:|:---:|---:|---:|---:|
| 1 | 96.3957 | 9425 | true | 4.512M | 4.512M | 1.000x |
| 2 | 188.4887 | 10801 | true | 4.615M | 2.308M | 1.023x |
| 3 | 249.1505 | 15374 | true | 5.237M | 1.746M | 1.161x |
| 4 | 308.8051 | 18320 | true | 5.634M | 1.409M | 1.249x |
| 5 | 386.8391 | 14989 | true | 5.622M | 1.124M | 1.246x |

Best observed point: B=4, 5.63M weighted cells-in-flight/sec. B=5 is a plateau, not a further win. All recorded rows stayed under the 26 GiB headroom target.

## Commands Run

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true pytest -q tests/test_operational_namelist_cache_key.py tests/test_v023_batched_minigrid.py -q
python -m py_compile src/gpuwrf/integration/nested_pipeline.py proofs/v023/batched_minigrid/f1_cpu_b2_gate.py proofs/v023/gpu_gates/g3_batch_sweep_driver.py tests/test_v023_batched_minigrid.py tests/test_operational_namelist_cache_key.py
```

Results:

- CPU tests: 13 passed.
- Python compile check: passed.

## Residual Risk

No known correctness blocker remains for F1 B>1 loading. The B-sweep was closed after manager acceptance with B=1..5; no B=6 completed row is included in the final JSONL.
