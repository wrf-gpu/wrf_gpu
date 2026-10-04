#!/usr/bin/env bash
# Inside a GPU lock: N concurrent WN3 forecasts + 1 s power dmon (-s pucm) + host-RAM watchdog; optional rolling compression (ROLLING=1).
# Args: ROOT arm_dir hours case1 [case2 ...]
set -uo pipefail
export ROOT=$1; arm_dir=$2; hours=$3; shift 3
W=<USER_HOME>/wrf_gpu2_lanes/wn3/W7
mkdir -p "$arm_dir"
nvidia-smi dmon -s pucm -d 1 -o T > "$arm_dir/dmon.log" 2>&1 & dmon=$!
t0=$(date +%s.%N); echo "{\"start_unix\": $t0, \"cases\": \"$*\", \"hours\": $hours, \"root\": \"$ROOT\", \"mem_fraction\": \"$(cat $ROOT/mem_fraction 2>/dev/null)\", \"cpus\": \"$(taskset -pc $$ | awk -F': ' '{print $2}')\"}" > "$arm_dir/start.json"
pids=()
for c in "$@"; do
  bash $W/arm.sh <USER_HOME>/wrf_gpu2_lanes/wn3/cases/$c "$arm_dir/$c" "$hours" "wn3_${c}_W7_$(basename $arm_dir)" > "$arm_dir/$c.log" 2>&1 &
  pids+=($!)
done
comp=""
if [[ "${ROLLING:-0}" == 1 ]]; then
  taskset -c 8,9,24,25 nice -n 19 env JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3 $W/rolling_compress.py "$arm_dir" --workers 2 > "$arm_dir/rolling_compress.log" 2>&1 & comp=$!
fi
( while true; do
    avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    rss=0; for p in "${pids[@]}"; do r=$(awk '/VmRSS/ {print $2}' /proc/$p/status 2>/dev/null || echo 0); rss=$((rss + ${r:-0})); done
    echo "$(date +%s) $avail $rss" >> "$arm_dir/hostmem.log"
    if (( avail < 6291456 )); then echo "WATCHDOG: MemAvailable ${avail} kB < 6 GiB, killing own pids ${pids[*]}" >> "$arm_dir/hostmem.log"; kill -TERM "${pids[@]}" 2>/dev/null; sleep 20; kill -KILL "${pids[@]}" 2>/dev/null; break; fi
    alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=1; done; (( alive )) || break
    sleep 1
  done ) & wd=$!
rcs=()
for p in "${pids[@]}"; do wait $p; rcs+=($?); done
t1=$(date +%s.%N); kill $wd $dmon 2>/dev/null
echo "{\"end_unix\": $t1, \"wall_s\": $(echo "$t1 - $t0" | bc), \"rcs\": \"${rcs[*]}\"}" > "$arm_dir/end.json"
[[ -n "$comp" ]] && wait $comp
cat "$arm_dir/end.json"
for r in "${rcs[@]}"; do [[ $r == 0 ]] || exit 1; done
