#!/usr/bin/env bash
# Runs INSIDE one GPU lock. R07 WN3 restart: interrupted (SIGKILL after verified checkpoint at root_step=CUT) + fresh-process resume
# into the SAME output stream; the control is the uninterrupted run with identical ROOT/env/hours (compared afterwards).
# Args: ROOT stream_dir case hours cut_root_step
set -euo pipefail
export ROOT=$1; stream=$2; case=$3; hours=$4; cut=$5
W4=<USER_HOME>/wrf_gpu2_lanes/wn3/W7
gens=$stream.generations; mkdir -p "$gens"
ck=(--checkpoint-dir "$gens" --checkpoint-interval-steps 67 --checkpoint-max-generations 2
    --checkpoint-max-bytes 8589934592 --checkpoint-reserve-bytes 10737418240)
to_json() { <USER_HOME>/miniconda3/bin/python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"; }
# 1) interrupted
WN3_CLI_EXTRA=$(to_json "${ck[@]}") bash $W4/arm.sh <USER_HOME>/wrf_gpu2_lanes/wn3/cases/$case "$stream" "$hours" "wn3_${case}_R07_interrupted" \
  > "$stream.interrupted.log" 2>&1 & child=$!
t0=$(date +%s.%N)
until grep -q "\[checkpoint\] VERIFIED .* root_step=$cut\$" "$stream.interrupted.log" 2>/dev/null; do
  if ! kill -0 $child 2>/dev/null; then echo "interrupted child exited before verified root_step=$cut" >&2; exit 3; fi
  sleep 1
done
py=$child  # arm.sh execs python: the background pid IS the forecast process
kill -9 $py; wait $child || true
gen=$(grep "\[checkpoint\] VERIFIED .* root_step=$cut\$" "$stream.interrupted.log" | tail -1 | awk '{print $3}')
echo "{\"killed_pid\": $py, \"signal\": \"SIGKILL\", \"cut_root_step\": $cut, \"generation\": \"$gen\", \"t_kill_s\": $(echo "$(date +%s.%N) - $t0" | bc)}" > "$stream.kill.json"
mv "$stream/receipt.json" "$stream.receipt_interrupted.json" 2>/dev/null || true
mv "$stream/cli_stdout.json" "$stream.cli_stdout_interrupted.json" 2>/dev/null || true
# 2) fresh-process resume, same stream
WN3_CLI_EXTRA=$(to_json "${ck[@]}" --resume-checkpoint "$gen") bash $W4/arm.sh <USER_HOME>/wrf_gpu2_lanes/wn3/cases/$case "$stream" "$hours" "wn3_${case}_R07_resume" \
  > "$stream.resume.log" 2>&1
echo "{\"resume_rc\": 0, \"total_s\": $(echo "$(date +%s.%N) - $t0" | bc)}" > "$stream.done.json"
