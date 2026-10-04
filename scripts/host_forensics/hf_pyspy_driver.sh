#!/usr/bin/env bash
# cProfile + py-spy(native) of the warm loop — run under one GPU lock.
set -uo pipefail
PY=<USER_HOME>/miniconda3/bin/python3
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=<USER_HOME>/src/wrf_gpu2/.agent/sprints/2026-09-18-v0250-host-forensics/artifacts
mkdir -p "$OUT"
export GPUWRF_WRF_ROOT=<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF

echo "=== cprofile stage ==="
$PY "$HERE/hf_timeline.py" --stage cprofile > "$OUT/cprofile_stage.log" 2>&1
echo "cprofile rc=$?"

echo "=== py-spy native sample (75s) ==="
rm -f /tmp/hf_pyspy_stop
timeout -s INT 80 <USER_HOME>/miniconda3/bin/py-spy record \
  --native --rate 200 --format raw -o "$OUT/pyspy_native_raw.txt" -- \
  $PY "$HERE/hf_timeline.py" --stage loopchild &
PYSPY_PID=$!
# child loops until STOP file; sample for ~75s then stop it
sleep 72
touch /tmp/hf_pyspy_stop
wait $PYSPY_PID
echo "pyspy rc=$?"
wc -l "$OUT/pyspy_native_raw.txt" 2>/dev/null
echo "PYSPY_DONE"
