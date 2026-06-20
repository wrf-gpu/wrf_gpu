# GPU Runbook

This is the supported way to run long GPU validations from this repository. Do
not depend on a helper under `/tmp`; those disappear and were the source of the
failed launch attempt on 2026-06-08.

> **Paths in this runbook are the maintainer's reference layout.** Concrete
> `<DATA_ROOT>/...` directories (run roots, the Canary CPU-WRF truth, the AEMET
> station dataset) are the paths on the reference workstation. **Substitute your
> own** writable run root and your own case/truth directories — every command
> here takes the paths as explicit arguments or honours an environment override
> (`GPUWRF_TOST_RUN_DIR`, `GPUWRF_AEMET_ROOT`, `PYTHON`, `--input-dir`,
> `--cpu-truth-root`, `--out-dir`). Nothing here requires the literal paths.

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

With resource logging:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 \
  --resource-log-dir <DATA_ROOT>/wrf_gpu_validation/my_run/resources \
  --resource-label my_run \
  --resource-interval 5 \
  -- \
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
- optionally writes `*_gpu_usage.csv`, `*_process_usage.csv`, and
  `*_system_memory.csv` via `scripts/monitor_resource_usage.sh`;
- exits `75` if another GPU run already owns the lock.

The first JAX/XLA invocation can spend several minutes compiling with little or
no log output. Check GPU memory/process state before assuming it is hung.

## Powered TOST n=15

Foreground (set `PYTHON` only if your GPU-enabled interpreter is not the default
`python3` on `PATH`):

```bash
scripts/run_powered_tost_n15.sh --resume
# or, to pin a specific interpreter:
PYTHON=/path/to/your/python scripts/run_powered_tost_n15.sh --resume
```

Detached, with durable log/rc/runinfo:

```bash
scripts/run_powered_tost_n15.sh --detach --resume
```

Default detached artifacts:

- `<DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current.log`
- `<DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current.rc`
- `<DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current.runinfo`
- `<DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current_resources/n15_current_gpu_usage.csv`
- `<DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current_resources/n15_current_process_usage.csv`
- `<DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current_resources/n15_current_system_memory.csv`

Monitor:

```bash
tail -f <DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current.log
cat <DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current.runinfo
cat <DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current.rc
tail -f <DATA_ROOT>/wrf_gpu_validation/v0130_marathon/n15_current_resources/n15_current_gpu_usage.csv
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

## v0.14 Field-Parity Gates

The v0.14 release gate is field parity/stability, not station-only TOST. Do not
start long GPU gates until the short h1 field falsifier is green on the final
candidate branch and exact-branch memory preflight is green.

The mandatory Canary gate uses this existing CPU-WRF truth:

- run id: `20260501_18z_l2_72h_20260519T173026Z`
- CPU truth:
  `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- retained inputs:
  `<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z`
- domain: `d02`

Short h1 Canary falsifier with resource CSVs:

```bash
RUN_ROOT=<DATA_ROOT>/wrf_gpu_validation/v014_short_field_falsifier_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_ROOT"/{gpu_output,proofs,resources}
scripts/run_gpu_lowprio.sh --cores 0-23 \
  --resource-log-dir "$RUN_ROOT/resources" \
  --resource-label v014_canary_h1_field_falsifier \
  --resource-interval 5 \
  -- \
  python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
    --run-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2 \
    --cpu-truth-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output \
    --run-id 20260501_18z_l2_72h_20260519T173026Z \
    --hours 1 \
    --output-root "$RUN_ROOT/gpu_output" \
    --proof-dir "$RUN_ROOT/proofs"
```

Detached h1 Canary falsifier (recommended for manager/agent shells that may
exit before the run finishes):

```bash
RUN_ROOT=<DATA_ROOT>/wrf_gpu_validation/v014_short_field_falsifier_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_ROOT"/{gpu_output,proofs,resources}
cat > "$RUN_ROOT/runinfo.txt" <<EOF
run_root=$RUN_ROOT
run_id=20260501_18z_l2_72h_20260519T173026Z
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
launch=nohup setsid
status=launched
EOF
nohup setsid bash -lc '
  set +e
  cd <USER_HOME>/src/wrf_gpu2
  RUN_ROOT="'"$RUN_ROOT"'"
  scripts/run_gpu_lowprio.sh --cores 0-23 \
    --resource-log-dir "$RUN_ROOT/resources" \
    --resource-label v014_canary_h1_field_falsifier \
    --resource-interval 5 \
    -- \
    python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
      --run-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2 \
      --cpu-truth-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output \
      --run-id 20260501_18z_l2_72h_20260519T173026Z \
      --hours 1 \
      --output-root "$RUN_ROOT/gpu_output" \
      --proof-dir "$RUN_ROOT/proofs"
  rc=$?
  echo "$rc" > "$RUN_ROOT/h1_gpu.rc"
  printf "finished_utc=%s\nrc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >> "$RUN_ROOT/runinfo.txt"
  exit "$rc"
' > "$RUN_ROOT/h1_gpu.log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/h1_gpu.pid"
echo "$RUN_ROOT"
```

Do not rely on `(...) >log 2>&1 &` from a short-lived agent/tool shell: the
child process can receive SIGHUP before it writes an rc file or resource rows.
Use `nohup setsid` for detached validation jobs.

Compare the h1 output:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
    --gpu-dir "$RUN_ROOT/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z" \
    --domain d02 \
    --init 2026-05-01T18:00:00+00:00 \
    --min-lead 1 --max-lead 1 \
    --out-json "$RUN_ROOT/short_field_h1_grid_compare.json" \
    --out-md "$RUN_ROOT/short_field_h1_grid_compare.md"
```

If the h1 falsifier is classified as launch-safe, run the full Canary 72 h gate
with an immutable detached run root under `<DATA_ROOT>/wrf_gpu_validation/`.
This is the exact v0.14 Canary d02 release-gate command shape:

```bash
RUN_ROOT=<DATA_ROOT>/wrf_gpu_validation/v014_canary_d02_72h_$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID=20260501_18z_l2_72h_20260519T173026Z
CPU_DIR=<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/$RUN_ID
GPU_DIR="$RUN_ROOT/gpu_output/l2_d02_$RUN_ID"
mkdir -p "$RUN_ROOT"/{gpu_output,proofs,resources,grid_delta_atlas,grid_delta_atlas_assets}
cat > "$RUN_ROOT/runinfo.txt" <<EOF
run_root=$RUN_ROOT
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
kind=v014_canary_d02_72h_field_parity_gate
run_id=$RUN_ID
cpu_truth=$CPU_DIR
input_root=<DATA_ROOT>/canairy_meteo/runs/wrf_l2
launch=nohup setsid
status=launched
EOF
nohup setsid bash -lc '
  set +e
  cd <USER_HOME>/src/wrf_gpu2
  RUN_ROOT="'"$RUN_ROOT"'"
  RUN_ID="'"$RUN_ID"'"
  CPU_DIR=<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/$RUN_ID
  GPU_DIR="$RUN_ROOT/gpu_output/l2_d02_$RUN_ID"
  scripts/run_gpu_lowprio.sh --cores 0-23 \
    --resource-log-dir "$RUN_ROOT/resources" \
    --resource-label v014_canary_d02_72h \
    --resource-interval 5 \
    -- \
    python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
      --run-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2 \
      --cpu-truth-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output \
      --run-id "$RUN_ID" \
      --hours 72 \
      --output-root "$RUN_ROOT/gpu_output" \
      --proof-dir "$RUN_ROOT/proofs" \
    > "$RUN_ROOT/canary_d02_72h_gpu.log" 2>&1
  gpu_rc=$?
  echo "$gpu_rc" > "$RUN_ROOT/canary_d02_72h_gpu.rc"
  printf "gpu_finished_utc=%s\ngpu_rc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gpu_rc" >> "$RUN_ROOT/runinfo.txt"
  if [ "$gpu_rc" -ne 0 ]; then exit "$gpu_rc"; fi

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
    python scripts/compare_wrfout_grid.py \
      --cpu-dir "$CPU_DIR" \
      --gpu-dir "$GPU_DIR" \
      --domain d02 \
      --init 2026-05-01T18:00:00+00:00 \
      --min-lead 1 --max-lead 72 \
      --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
      --out-json "$RUN_ROOT/canary_d02_72h_grid_compare.json" \
      --out-md "$RUN_ROOT/canary_d02_72h_grid_compare.md" \
    > "$RUN_ROOT/canary_d02_72h_compare.log" 2>&1
  cmp_rc=$?
  echo "$cmp_rc" > "$RUN_ROOT/canary_d02_72h_compare.rc"
  printf "compare_finished_utc=%s\ncompare_rc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cmp_rc" >> "$RUN_ROOT/runinfo.txt"

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
    python scripts/build_grid_delta_atlas.py \
      --cpu-dir "$CPU_DIR" \
      --gpu-dir "$GPU_DIR" \
      --case-id canary_d02_20260501_18z \
      --domain d02 \
      --init 2026-05-01T18:00:00+00:00 \
      --min-lead 1 --max-lead 72 \
      --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
      --proof-dir "$RUN_ROOT/grid_delta_atlas" \
      --asset-dir "$RUN_ROOT/grid_delta_atlas_assets" \
    > "$RUN_ROOT/canary_d02_72h_atlas.log" 2>&1
  atlas_rc=$?
  echo "$atlas_rc" > "$RUN_ROOT/canary_d02_72h_atlas.rc"
  printf "atlas_finished_utc=%s\natlas_rc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$atlas_rc" >> "$RUN_ROOT/runinfo.txt"
  if [ "$cmp_rc" -ne 0 ]; then exit "$cmp_rc"; fi
  exit "$atlas_rc"
' > "$RUN_ROOT/canary_d02_72h_chain.log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/canary_d02_72h_chain.pid"
echo "$RUN_ROOT"
```

Monitor the detached Canary gate:

```bash
RUN_ROOT=<DATA_ROOT>/wrf_gpu_validation/v014_canary_d02_72h_<timestamp>
cat "$RUN_ROOT"/canary_d02_72h_*.rc 2>/dev/null || true
cat "$RUN_ROOT/runinfo.txt"
find "$RUN_ROOT/gpu_output" -type f -name 'wrfout_d0*' | wc -l
tail -f "$RUN_ROOT/resources/v014_canary_d02_72h_gpu_usage.csv"
tail -80 "$RUN_ROOT/canary_d02_72h_gpu.log"
nvidia-smi
```

The Switzerland/Gotthard 72 h CPU truth is built as a CPU-WRF run, not a GPU
job. After the Switzerland case builder has populated `wrfinput_d01`,
`wrfbdy_d01`, and `namelist.input` under a `run_cpu` directory, track CPU memory
with the monitor directly around the launched WRF process:

```bash
OUT=<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_$(date -u +%Y%m%dT%H%M%SZ)
CPU_DIR=$OUT/run_cpu
mkdir -p "$OUT/resources"
# Launch the CPU-WRF wrapper in the background, then monitor its PID tree.
RUNROOT="$OUT" CPU_DIR="$CPU_DIR" NRANKS=24 \
  bash scripts/run_switzerland_cpu_reference_mpi.sh > "$OUT/switzerland_72h_cpu.log" 2>&1 &
PID=$!
scripts/monitor_resource_usage.sh \
  --out-dir "$OUT/resources" \
  --label switzerland_72h_cpu \
  --interval 5 \
  --no-gpu \
  --pid "$PID" \
  --match-regex 'wrf.exe|mpirun'
wait "$PID"; echo $? > "$OUT/switzerland_72h_cpu.rc"
```

For GPU runs, prefer `scripts/run_gpu_lowprio.sh --resource-log-dir ...`; it
writes:

- `<label>_gpu_usage.csv`
- `<label>_process_usage.csv`
- `<label>_system_memory.csv`

Matched Switzerland/Gotthard 72 h GPU gate after the CPU truth above is
available and the GPU lock is free:

```bash
RUN_ROOT=<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_$(date -u +%Y%m%dT%H%M%SZ)
CPU_DIR=<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu
GPU_OUT="$RUN_ROOT/gpu_output"
SCRATCH="$RUN_ROOT/scratch"
mkdir -p "$RUN_ROOT"/{gpu_output,proofs,resources,grid_delta_atlas,grid_delta_atlas_assets,scratch}
cat > "$RUN_ROOT/runinfo.txt" <<EOF
run_root=$RUN_ROOT
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
kind=v014_switzerland_d01_72h_field_parity_gate
cpu_truth=$CPU_DIR
domain=d01
launch=nohup setsid
status=launched
EOF
nohup setsid bash -lc '
  set +e
  cd <USER_HOME>/src/wrf_gpu2
  RUN_ROOT="'"$RUN_ROOT"'"
  CPU_DIR="'"$CPU_DIR"'"
  GPU_OUT="$RUN_ROOT/gpu_output"
  SCRATCH="$RUN_ROOT/scratch"
  scripts/run_gpu_lowprio.sh --cores 0-23 \
    --resource-log-dir "$RUN_ROOT/resources" \
    --resource-label v014_switzerland_d01_72h \
    --resource-interval 5 \
    -- \
    python -m gpuwrf.cli run \
      --input-dir "$CPU_DIR" \
      --output-dir "$GPU_OUT" \
      --scratch-dir "$SCRATCH" \
      --domain d01 \
      --hours 72 \
      --proof-dir "$RUN_ROOT/proofs" \
    > "$RUN_ROOT/switzerland_d01_72h_gpu.log" 2>&1
  gpu_rc=$?
  echo "$gpu_rc" > "$RUN_ROOT/switzerland_d01_72h_gpu.rc"
  printf "gpu_finished_utc=%s\ngpu_rc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gpu_rc" >> "$RUN_ROOT/runinfo.txt"
  if [ "$gpu_rc" -ne 0 ]; then exit "$gpu_rc"; fi

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
    python scripts/compare_wrfout_grid.py \
      --cpu-dir "$CPU_DIR" \
      --gpu-dir "$GPU_OUT" \
      --domain d01 \
      --init 2023-01-15T00:00:00+00:00 \
      --min-lead 1 --max-lead 72 \
      --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
      --out-json "$RUN_ROOT/switzerland_d01_72h_grid_compare.json" \
      --out-md "$RUN_ROOT/switzerland_d01_72h_grid_compare.md" \
    > "$RUN_ROOT/switzerland_d01_72h_compare.log" 2>&1
  cmp_rc=$?
  echo "$cmp_rc" > "$RUN_ROOT/switzerland_d01_72h_compare.rc"
  printf "compare_finished_utc=%s\ncompare_rc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cmp_rc" >> "$RUN_ROOT/runinfo.txt"

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
    python scripts/build_grid_delta_atlas.py \
      --cpu-dir "$CPU_DIR" \
      --gpu-dir "$GPU_OUT" \
      --case-id switzerland_d01_20230115_00z \
      --domain d01 \
      --init 2023-01-15T00:00:00+00:00 \
      --min-lead 1 --max-lead 72 \
      --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
      --proof-dir "$RUN_ROOT/grid_delta_atlas" \
      --asset-dir "$RUN_ROOT/grid_delta_atlas_assets" \
    > "$RUN_ROOT/switzerland_d01_72h_atlas.log" 2>&1
  atlas_rc=$?
  echo "$atlas_rc" > "$RUN_ROOT/switzerland_d01_72h_atlas.rc"
  printf "atlas_finished_utc=%s\natlas_rc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$atlas_rc" >> "$RUN_ROOT/runinfo.txt"
  if [ "$cmp_rc" -ne 0 ]; then exit "$cmp_rc"; fi
  exit "$atlas_rc"
' > "$RUN_ROOT/switzerland_d01_72h_chain.log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/switzerland_d01_72h_chain.pid"
echo "$RUN_ROOT"
```

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
    --cpu-truth-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output \
    --run-id 20260501_18z_l2_72h_20260519T173026Z \
    --hours 1 \
    --output-root /tmp/v014_post_static_writer_smoke \
    --proof-dir proofs/v014/post_static_writer_smoke/live_nested_h1
```

Then compare the written d02 wrfout against CPU-WRF truth:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
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
