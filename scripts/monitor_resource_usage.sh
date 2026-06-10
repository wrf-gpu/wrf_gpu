#!/usr/bin/env bash
# monitor_resource_usage.sh -- lightweight CSV resource monitor for validation runs.
#
# Writes up to three CSV files:
#   <label>_gpu_usage.csv       GPU memory/util/power from nvidia-smi
#   <label>_process_usage.csv   process RSS/CPU rows for a PID tree and/or regex
#   <label>_system_memory.csv   host memory snapshot from /proc/meminfo
#
# Typical:
#   scripts/monitor_resource_usage.sh --out-dir run/resources --label swiss_gpu --pid "$PID"
#   scripts/monitor_resource_usage.sh --out-dir run/resources --label swiss_cpu --no-gpu \
#     --pid "$PID" --match-regex 'wrf.exe|mpirun'

set -euo pipefail
trap '' HUP

OUT_DIR=
LABEL=resource
INTERVAL=5
TARGET_PID=
MATCH_REGEX=
NO_GPU=0

usage() {
  printf '%s\n' \
    "Usage: scripts/monitor_resource_usage.sh --out-dir DIR [--label NAME] [--interval SEC] [--pid PID] [--match-regex REGEX] [--no-gpu]" \
    "" \
    "Stops when --pid exits. Without --pid it runs until killed." >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      [[ $# -ge 2 ]] || { echo "missing value for --out-dir" >&2; exit 2; }
      OUT_DIR="$2"; shift 2 ;;
    --label)
      [[ $# -ge 2 ]] || { echo "missing value for --label" >&2; exit 2; }
      LABEL="$2"; shift 2 ;;
    --interval)
      [[ $# -ge 2 ]] || { echo "missing value for --interval" >&2; exit 2; }
      INTERVAL="$2"; shift 2 ;;
    --pid)
      [[ $# -ge 2 ]] || { echo "missing value for --pid" >&2; exit 2; }
      TARGET_PID="$2"; shift 2 ;;
    --match-regex)
      [[ $# -ge 2 ]] || { echo "missing value for --match-regex" >&2; exit 2; }
      MATCH_REGEX="$2"; shift 2 ;;
    --no-gpu)
      NO_GPU=1; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$OUT_DIR" ]] || { usage; exit 2; }
mkdir -p "$OUT_DIR"

GPU_CSV="$OUT_DIR/${LABEL}_gpu_usage.csv"
PROC_CSV="$OUT_DIR/${LABEL}_process_usage.csv"
MEM_CSV="$OUT_DIR/${LABEL}_system_memory.csv"
RUNINFO="$OUT_DIR/${LABEL}_monitor.runinfo"

printf 'timestamp_epoch,timestamp_iso,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct,power_draw_w\n' > "$GPU_CSV"
printf 'timestamp_epoch,timestamp_iso,pid,ppid,stat,psr,pcpu,pmem,rss_kib,vsz_kib,args\n' > "$PROC_CSV"
printf 'timestamp_epoch,timestamp_iso,mem_total_kib,mem_available_kib,swap_total_kib,swap_free_kib\n' > "$MEM_CSV"

{
  printf 'started=%s\n' "$(date -Is)"
  printf 'label=%s\n' "$LABEL"
  printf 'interval=%s\n' "$INTERVAL"
  printf 'target_pid=%s\n' "${TARGET_PID:-}"
  printf 'match_regex=%s\n' "${MATCH_REGEX:-}"
  printf 'no_gpu=%s\n' "$NO_GPU"
  printf 'gpu_csv=%s\n' "$GPU_CSV"
  printf 'process_csv=%s\n' "$PROC_CSV"
  printf 'system_memory_csv=%s\n' "$MEM_CSV"
} > "$RUNINFO"

descendants_of() {
  local root="$1"
  local queue=("$root")
  local seen=" $root "
  local i=0
  while [[ $i -lt ${#queue[@]} ]]; do
    local p="${queue[$i]}"
    i=$((i + 1))
    local children
    children="$(pgrep -P "$p" 2>/dev/null || true)"
    local c
    for c in $children; do
      if [[ "$seen" != *" $c "* ]]; then
        queue+=("$c")
        seen+=" $c "
      fi
    done
  done
  printf '%s\n' "${queue[@]}"
}

sample_once() {
  local epoch iso
  epoch="$(date +%s)"
  iso="$(date -Is)"

  if [[ "$NO_GPU" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, -v e="$epoch" -v t="$iso" '
          {
            for (i=1; i<=NF; i++) { gsub(/^ +| +$/, "", $i) }
            printf "%s,%s,%s,%s,%s,%s,%s\n", e,t,$1,$2,$3,$4,$5
          }' >> "$GPU_CSV" || true
  fi

  awk -v e="$epoch" -v t="$iso" '
    BEGIN { mt=""; ma=""; st=""; sf="" }
    /^MemTotal:/ { mt=$2 }
    /^MemAvailable:/ { ma=$2 }
    /^SwapTotal:/ { st=$2 }
    /^SwapFree:/ { sf=$2 }
    END { printf "%s,%s,%s,%s,%s,%s\n", e,t,mt,ma,st,sf }
  ' /proc/meminfo >> "$MEM_CSV"

  local pid_set=""
  if [[ -n "$TARGET_PID" ]]; then
    pid_set="$(descendants_of "$TARGET_PID" | tr '\n' ' ')"
  fi

  ps -eo pid=,ppid=,stat=,psr=,pcpu=,pmem=,rss=,vsz=,args= \
    | awk -v e="$epoch" -v t="$iso" -v pids=" $pid_set " -v regex="$MATCH_REGEX" '
        {
          pid=$1; ppid=$2; stat=$3; psr=$4; pcpu=$5; pmem=$6; rss=$7; vsz=$8
          args=$0
          sub(/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+/, "", args)
          keep=0
          if (pids != "  " && index(pids, " " pid " ") > 0) keep=1
          if (regex != "" && args ~ regex) keep=1
          if (keep) {
            gsub(/"/, "\"\"", args)
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\"%s\"\n", e,t,pid,ppid,stat,psr,pcpu,pmem,rss,vsz,args
          }
        }' >> "$PROC_CSV"
}

while true; do
  sample_once
  if [[ -n "$TARGET_PID" ]] && ! kill -0 "$TARGET_PID" 2>/dev/null; then
    break
  fi
  sleep "$INTERVAL"
done

printf 'ended=%s\n' "$(date -Is)" >> "$RUNINFO"
