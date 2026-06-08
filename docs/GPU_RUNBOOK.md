# GPU Runbook

This is the supported way to run long GPU validations from this repository. Do
not depend on a helper under `/tmp`; those disappear and were the source of the
failed launch attempt on 2026-06-08.

## Golden Path

From the repository root:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- \
  python -m gpuwrf.cli run \
    --input-dir my_case \
    --output-dir runs/my_forecast \
    --domain d02 \
    --hours 24 \
    --scratch-dir /fast/nvme/gpuwrf_scratch
```

The wrapper:

- holds `/tmp/wrf_gpu_validation_gpu.lock` with `flock` so only one GPU
  validation owns the card;
- sets `PYTHONPATH=src`, `JAX_ENABLE_X64=true`, and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`;
- runs at low CPU/IO priority and pins CPU helper work to the selected cores;
- exits `75` if another GPU run already owns the lock.

The first JAX/XLA invocation can spend several minutes compiling with little or
no log output. Check GPU memory/process state before assuming it is hung.

## Powered TOST n=15

Foreground:

```bash
PYTHON=/home/enric/miniconda3/bin/python \
  scripts/run_powered_tost_n15.sh --resume
```

Detached, with durable log/rc/runinfo:

```bash
PYTHON=/home/enric/miniconda3/bin/python \
  scripts/run_powered_tost_n15.sh --detach --resume
```

Default detached artifacts:

- `/mnt/data/wrf_gpu_validation/v0130_marathon/n15_current.log`
- `/mnt/data/wrf_gpu_validation/v0130_marathon/n15_current.rc`
- `/mnt/data/wrf_gpu_validation/v0130_marathon/n15_current.runinfo`

Monitor:

```bash
tail -f /mnt/data/wrf_gpu_validation/v0130_marathon/n15_current.log
cat /mnt/data/wrf_gpu_validation/v0130_marathon/n15_current.runinfo
cat /mnt/data/wrf_gpu_validation/v0130_marathon/n15_current.rc
nvidia-smi
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | rg 'run_powered_tost|run_one_case'
```

The rc file is removed at launch and is written only when the detached run
ends. No rc file while the process is alive is normal.

## Resume And Hibernation

CUDA contexts do not survive suspend. If the workstation hibernates during a
GPU run, treat only the in-flight case as lost:

1. Confirm the current GPU process is stale: GPU utilization stays at 0%, no log
   progress, and no new proof object appears after resume.
2. Kill only the stale `run_one_case` / TOST process tree.
3. Relaunch with `scripts/run_powered_tost_n15.sh --detach --resume`.

Completed TOST cases are durable in
`proofs/v0120/powered_tost_n15/case_<RUN_ID>.json`; `--resume` skips them.
Per-case pipeline proofs are under
`proofs/v0120/powered_tost_n15/pipeline_proofs/<RUN_ID>/`.

## Quick Checks

```bash
nvidia-smi
python -c 'import jax; print(jax.devices())'
scripts/run_powered_tost_n15.sh --dry-run --resume
```

`--dry-run` prepares the merged root and prints the case plan without launching
GPU forecasts.

## Common Failures

- `GPU lock busy` / rc `75`: another GPU campaign owns
  `/tmp/wrf_gpu_validation_gpu.lock`; inspect with `ps` and wait or stop the
  owning run deliberately.
- `command not found` for a `/tmp/wrf_gpu_run_lowprio.sh` path: use
  `scripts/run_gpu_lowprio.sh` or `scripts/run_powered_tost_n15.sh`.
- Long startup with no hourly `wrfout`: usually cold XLA compile. Watch
  `nvidia-smi` memory and CPU activity.
- `CUDA_ERROR_OUT_OF_MEMORY`: verify the run is on the memory-fixed branch and
  that `XLA_PYTHON_CLIENT_PREALLOCATE=false` is visible in the log.
