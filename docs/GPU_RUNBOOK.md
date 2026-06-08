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

## L2 d02 TOST/Debug Cases

The L2 corpus cases used by powered TOST are **max_dom=2 live-nested cases**:
d01 has `wrfbdy_d01`, while d02 is a nest and correctly has no `wrfbdy_d02`.
For these cases, do not use the old single-domain `scripts/m7_l2_d02_replay.py`
path for a fresh GPU forecast; it routes d02 as standalone and fails fast with
`standalone native-init requires wrfbdy_d02`.

Use the TOST-fixed live-nested per-case runner for short debug smokes:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- \
  python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
    --run-root /tmp/v0120_merged_run_root \
    --cpu-truth-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output \
    --run-id 20260501_18z_l2_72h_20260519T173026Z \
    --hours 1 \
    --output-root /tmp/v014_post_static_writer_smoke \
    --proof-dir proofs/v014/post_static_writer_smoke/live_nested_h1
```

Then compare the written d02 wrfout against CPU-WRF truth:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
    --gpu-dir /tmp/v014_post_static_writer_smoke/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
    --domain d02 \
    --min-lead 1 --max-lead 1 \
    --out-json proofs/v014/post_static_writer_grid_compare.json \
    --out-md proofs/v014/post_static_writer_grid_compare.md
```

For the full n=15 campaign, still use `scripts/run_powered_tost_n15.sh`; it
prepares the same merged root and runs the same live-nested per-case path.

## Common Failures

- `GPU lock busy` / rc `75`: another GPU campaign owns
  `/tmp/wrf_gpu_validation_gpu.lock`; inspect with `ps` and wait or stop the
  owning run deliberately.
- `command not found` for a `/tmp/wrf_gpu_run_lowprio.sh` path: use
  `scripts/run_gpu_lowprio.sh` or `scripts/run_powered_tost_n15.sh`.
- `standalone native-init requires wrfbdy_d02` on an L2 d02 case: wrong runner
  for a nested case. Use `run_one_case_v0120.py` or `python -m gpuwrf.cli run
  --max-dom 2` so d02 gets live parent boundaries.
- Long startup with no hourly `wrfout`: usually cold XLA compile. Watch
  `nvidia-smi` memory and CPU activity.
- `CUDA_ERROR_OUT_OF_MEMORY`: verify the run is on the memory-fixed branch and
  that `XLA_PYTHON_CLIENT_PREALLOCATE=false` is visible in the log.
