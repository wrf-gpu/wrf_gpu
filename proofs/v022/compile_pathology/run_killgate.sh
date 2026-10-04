#!/usr/bin/env bash
# Drives both phases of the BouLac O(nz) compile-pathology kill-gate, each in an
# ISOLATED process (so peak RSS + cold compile are clean), then the tiered
# comparison. Intended to be invoked UNDER the GPU lock:
#   scripts/with_gpu_lock.sh --label boulac-killgate -- proofs/v022/compile_pathology/run_killgate.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=true
export GPUWRF_MYNN_BOULAC_FP32=0

PY=python
HARNESS=proofs/v022/compile_pathology/boulac_onz_killgate.py
CMP=proofs/v022/compile_pathology/compare_tiered.py
HOURS="${KILLGATE_HOURS:-1}"
CTO="${KILLGATE_COMPILE_TIMEOUT:-1800}"
# Hard outer wall (SIGKILL) per phase: compile cap + warm-run/dump headroom.
# Authoritative against an XLA compile that ignores the in-process SIGALRM.
HARD=$(( CTO + 1200 ))

run_phase () {  # $1 algo
  local algo="$1"
  # SIGTERM at HARD (lets the in-process handler write a JSON), SIGKILL 60s later.
  timeout -k 60 "$HARD" "$PY" "$HARNESS" --algo "$algo" --hours "$HOURS" --compile-timeout "$CTO"
  local rc=$?
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    echo "phase $algo OUTER-TIMED-OUT rc=$rc (hard ${HARD}s) -> writing STILL-PATHOLOGICAL"
    "$PY" - "$algo" "$CTO" "$HARD" <<'PYEOF'
import json,sys
from pathlib import Path
algo,cto,hard=sys.argv[1],sys.argv[2],sys.argv[3]
HERE=Path("proofs/v022/compile_pathology")
HERE.mkdir(parents=True,exist_ok=True)
(HERE/f"killgate_{algo}.json").write_text(json.dumps({
 "algo":algo,"verdict":"STILL-PATHOLOGICAL","cold_compile_timed_out":True,
 "compile_timeout_s":int(cto),"hard_wall_s":int(hard),
 "note":f"outer SIGKILL wall {hard}s hit; cold compile did not complete -> XLA slow-compile pathology reproduced (or absolute compile wall exceeds the cap)",
},indent=2)+"\n")
print(json.dumps({"algo":algo,"verdict":"STILL-PATHOLOGICAL","outer_timeout":True}))
PYEOF
  fi
  return $rc
}

echo "=== PHASE 1: dense (frozen v0.14 default) — baseline compile/runtime + reference dump ==="
run_phase dense
rc_dense=$?
echo "phase1 rc=$rc_dense"

echo "=== PHASE 2: onz (O(nz) as DEFAULT in full operational jit) — pathology candidate ==="
run_phase onz
rc_onz=$?
echo "phase2 rc=$rc_onz"

echo "=== PHASE 3: tiered identity (onz vs dense == frozen v0.14) ==="
"$PY" "$CMP"
rc_cmp=$?
echo "phase3 rc=$rc_cmp"

echo "=== ARTIFACTS ==="
ls -la proofs/v022/compile_pathology/*.json 2>/dev/null
