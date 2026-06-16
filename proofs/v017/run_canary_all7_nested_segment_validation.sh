#!/usr/bin/env bash
# Manager-only GPU validation for the v0.17 nested _advance_chunk compile fix.
# Do not run from code-only worker lanes.
set -euo pipefail

ROOT="${ROOT:-/home/user/src/wrf_gpu2/.wt-nested-seg}"
INPUT_DIR="${INPUT_DIR:-/mnt/data/wrf_downscale/canary_all7/run}"
HOURS="${HOURS:-1}"
MAX_DOM="${MAX_DOM:-9}"
RR="${RR:-/mnt/data/wrf_gpu_validation/v017_canary_all7_nested_segment_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$RR/gpu_output" "$RR/proofs" "$RR/resources"

cat > "$RR/runinfo.txt" <<EOF
kind=v017_nested_segment_compile_validation
branch=worker/gpt/v017-nested-segment
root=$ROOT
input_dir=$INPUT_DIR
hours=$HOURS
max_dom=$MAX_DOM
expectation=bounded jit__advance_chunk compiles (~one per distinct domain shape/static namelist), then first-hour wrfout files
cpu_cores=0-3
EOF

cd "$ROOT"

set +e
env \
  OMP_NUM_THREADS=4 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_ENABLE_X64=true \
  GPUWRF_MYNN_BOULAC_ONZ=1 \
  GPUWRF_MOIST_CQW=1 \
  GPUWRF_KEEP_SCRATCH=1 \
  GPUWRF_RESOURCE_LABEL=v017_canary_all7_nested_segment \
  scripts/run_gpu_lowprio.sh \
    --cores 0-3 \
    --resource-log-dir "$RR/resources" \
    --resource-label v017_canary_all7_nested_segment \
    --resource-interval 5 \
    -- python -m gpuwrf run \
      --input-dir "$INPUT_DIR" \
      --output-dir "$RR/gpu_output" \
      --proof-dir "$RR/proofs" \
      --max-dom "$MAX_DOM" \
      --hours "$HOURS" \
  > "$RR/all7_nested_segment.log" 2>&1
rc=$?
set -e
echo "$rc" > "$RR/all7_nested_segment.rc"

compile_count="$(grep -c "Compiling module jit__advance_chunk" "$RR/all7_nested_segment.log" || true)"
slow_count="$(grep -c "The operation took" "$RR/all7_nested_segment.log" || true)"
wrfout_count="$(find "$RR/gpu_output" -maxdepth 1 -type f -name 'wrfout_d0*_*' | wc -l | tr -d ' ')"

cat > "$RR/summary.txt" <<EOF
rc=$rc
jit__advance_chunk_compile_alarms=$compile_count
slow_operation_completed=$slow_count
wrfout_count=$wrfout_count
expected_wrfout_count=$((HOURS * MAX_DOM))
log=$RR/all7_nested_segment.log
proof_json=$RR/proofs/nested_pipeline_run.json
EOF

cat "$RR/summary.txt"
exit "$rc"
