# FitSwiss 390 GPU Benchmark Investigation

Date: 2026-06-15
Branch/worktree inspected: `.wt-rc` (`worker/perf/v017-rc`)

## Outcome

`PIPELINE_BLOCKED` was not a validation-gate failure and not the CPU-wrfout replay
gotcha. The preserved pipeline proof says the run failed before any history file
with:

`JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 16.65GiB.`

The `6.88 GiB` peak in `fitswiss_gpu_benchmark.json` is only the peak of
successful allocations before the failed request. The raw JAX memory limit was
`25248645120` bytes (`23.51 GiB`), i.e. the default ~75% BFC pool cap on the
32 GB card. At failure, JAX already held about `6.88 GiB`; the new `16.65 GiB`
request put the needed pool just over the cap. The physical card was not proven
full; the JAX allocator pool was capped too low.

Primary blocked artifact:

- `<DATA_ROOT>/wrf_gpu_validation/v017_fitswiss_gpu_bench/proofs/pipeline_run_20260521.json`
- Backed up into `.wt-rc/proofs/v017/fitswiss_original_pipeline_blocked.json`

## Fix Applied

Patched `.wt-rc/scripts/v017_fitswiss_gpu_bench.py`:

- corrected the fitting-Swiss init path to `<DATA_ROOT>/wrf_gpu_validation/v017_fitswiss_gpu_init`;
- corrected grid metadata to `390 x 390`, `389 x 389` mass columns;
- set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_CLIENT_MEM_FRACTION=0.90`
  before any JAX import;
- cleaned stale output/proof dirs before a rerun;
- recorded allocator settings and both requested CPU denominator bases in the
  benchmark JSON;
- made the script return nonzero if the pipeline blocks or writes fewer than 13
  hourly outputs.

This is the intended rerun command:

```bash
cd <USER_HOME>/src/wrf_gpu2
bash scripts/with_gpu_lock.sh --label fitswiss-gpu-fix -- \
  bash -lc 'source <USER_HOME>/miniconda3/etc/profile.d/conda.sh; conda activate base; \
  cd <USER_HOME>/src/wrf_gpu2/.wt-rc; \
  export PYTHONPATH=src; export GPUWRF_MYNN_BOULAC_ONZ=1; export OMP_NUM_THREADS=4; \
  taskset -c 0-3 python -u scripts/v017_fitswiss_gpu_bench.py'
```

## Rerun Status

Not run in this sandbox. Hard blockers observed here:

- `nvidia-smi` fails: `couldn't communicate with the NVIDIA driver`;
- `/dev/nvidia*` is absent;
- `<DATA_ROOT>` is mounted read-only in this sandbox, but the benchmark writes
  `<DATA_ROOT>/wrf_gpu_validation/v017_fitswiss_gpu_bench`;
- the GPU lock probe timed out while the holder file named stale PID `3088566`;
  that PID no longer exists, but a nonblocking Python `flock` call still reports
  the lock blocked.

Therefore warm 13 h GPU wallclock, peak VRAM, and CPU/GPU factor are not measured
in this run.

## CPU Reference Defect

The stated CPU reference is also not a completed 13 h validation reference:

- `<DATA_ROOT>/wrf_gpu_validation/v017_fitswiss_390_20260614T223801Z/run_cpu/cpu_reference.out`
  reports `wrf.exe rc=1`;
- it explicitly says `SUCCESS COMPLETE WRF` was not found;
- `cpu_timing.json` has `mainloop_steps=1072`, but 13 h at 18 s requires `2600`
  steps;
- only six CPU wrfout files exist: `h0` through `h5`.

This blocks the binding h1-h13 incremental CPU-vs-GPU validation. A complete CPU
truth directory with `wrfout_d01_2023-01-15_00:00:00` through
`wrfout_d01_2023-01-15_13:00:00` is required before the validation gate can be
called green.

## Validation Status

No GPU checkpoint validation was run because the fixed GPU benchmark could not be
started in this sandbox. Even after a successful GPU run, only h1-h5 could be
validated against the currently supplied CPU directory; h6-h13 would remain
blocked until CPU truth is regenerated or supplied.

## Commands Run

- `python -m json.tool <DATA_ROOT>/wrf_gpu_validation/v017_fitswiss_gpu_bench/fitswiss_gpu_benchmark.json`
- `python -m json.tool <DATA_ROOT>/wrf_gpu_validation/v017_fitswiss_gpu_bench/proofs/pipeline_run_20260521.json`
- `sed -n ... src/gpuwrf/integration/daily_pipeline.py`
- `python -m py_compile scripts/v017_fitswiss_gpu_bench.py scripts/v017_fitswiss_gpu_fitprobe.py scripts/compare_wrfout_grid.py`
- `bash scripts/with_gpu_lock.sh --timeout 2 --label fitswiss-stale-lock-check -- bash -lc 'echo lock-ok'`
- `nvidia-smi --query-gpu=...`
- `ls -l /dev/nvidia*`
- CPU wrfout and timing inventory checks against the supplied `run_cpu` directory

## Next Required Action

Run the fixed driver on a host/session with GPU device access and writable
`<DATA_ROOT>`, then validate each produced GPU hour against a complete 13 h CPU
truth directory. Do not use the current `cpu_timing.json` as a clean 13 h
denominator without correcting the interrupted CPU run.
