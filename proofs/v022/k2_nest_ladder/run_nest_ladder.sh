#!/usr/bin/env bash
# K2 nest dt/n_sound ladder for the 3-domain Canary proxy.
#
# Run under the GPU lock, for example:
#   scripts/with_gpu_lock.sh --timeout 21600 --label k2-nest-ladder -- \
#     proofs/v022/k2_nest_ladder/run_nest_ladder.sh
#
# The source Canary namelist currently stores time_step=18. The brief's 3-domain
# ladder is d01/d02/d03 = 54/18/6, 60/20/6.67, 72/24/8, 90/30/10, so this runner
# scales the source root time_step explicitly to those root values.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"

ROOT="$(pwd)"
OUT_ROOT="${K2_NEST_OUT_ROOT:-proofs/v022/k2_nest_ladder}"
INPUT="${K2_NEST_INPUT_DIR:-<DATA_ROOT>/wrf_downscale/runs/20250121/cpu}"
HOURS="${K2_NEST_HOURS:-3}"
MAXDOM="${K2_NEST_MAXDOM:-3}"
RUNG_TIMEOUT="${K2_NEST_RUNG_TIMEOUT:-5400}"
MANAGER_PANE="${K2_NEST_MANAGER_PANE:-0:1.0}"
HISTORY_INTERVAL_MIN="${K2_NEST_HISTORY_INTERVAL_MIN:-18}"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/runs"

notify () {
  local msg="$1"
  if command -v tmux >/dev/null 2>&1; then
    tmux send-keys -t "$MANAGER_PANE" "[codex 0:3] $msg" C-m || true
  fi
}

build_input () {
  local tag="$1" root_dt="$2"
  local dst="$OUT_ROOT/runs/$tag/input"
  rm -rf "$dst"
  mkdir -p "$dst"
  for f in "$INPUT"/wrfinput_d* "$INPUT"/wrfbdy_d*; do
    [ -e "$f" ] && ln -sf "$f" "$dst/$(basename "$f")"
  done
  local factor
  factor="$(python - "$INPUT/namelist.input" "$root_dt" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"(?im)^\s*time_step\s*=\s*(\d+)", text)
if not m:
    raise SystemExit("no time_step in source namelist")
print(float(sys.argv[2]) / float(m.group(1)))
PY
)"
  env PYTHONPATH=src K2_FACTOR="$factor" K2_MAXDOM="$MAXDOM" \
    python proofs/v022/k2_dt_ladder/scale_namelist.py \
      "$INPUT/namelist.input" "$dst/namelist.input" \
      > "$OUT_ROOT/logs/${tag}_scale.log" 2>&1
  python - "$dst/namelist.input" "$HISTORY_INTERVAL_MIN" "$MAXDOM" <<'PY'
import re, sys
from pathlib import Path
path = Path(sys.argv[1])
interval = int(float(sys.argv[2]))
maxdom = int(sys.argv[3])
text = path.read_text()
values = ", ".join([str(interval)] * maxdom) + ","
text = re.sub(
    r"(?im)^(\s*history_interval\s*=\s*).*$",
    rf"\g<1>{values}",
    text,
)
zeros = ", ".join(["0"] * maxdom) + ","
text = re.sub(
    r"(?im)^(\s*history_interval_s\s*=\s*).*$",
    rf"\g<1>{zeros}",
    text,
)
path.write_text(text)
PY
}

run_one () {
  local tag="$1" root_dt="$2" nsound="$3"
  local run_dir="$OUT_ROOT/runs/$tag"
  local input="$run_dir/input"
  local out="$run_dir/out"
  local proof="$run_dir/proof"
  local scratch="$run_dir/scratch"
  local cache="/tmp/gpuwrf_k2_nest_${tag}_dt${root_dt}_ns${nsound}"
  local log="$OUT_ROOT/logs/${tag}.log"
  local result="$run_dir/result.json"
  local rc=0

  rm -rf "$out" "$proof" "$scratch" "$cache"
  mkdir -p "$out" "$proof" "$scratch" "$cache"

  notify "K2 nest ladder ${tag} start: requested d01 root_dt=${root_dt}s n_sound=${nsound}, maxdom=${MAXDOM}, input=${INPUT}."
  echo "=== ${tag}: root_dt=${root_dt} n_sound=${nsound} hours=${HOURS} maxdom=${MAXDOM} ==="
  timeout -k 60 "$RUNG_TIMEOUT" env \
    OMP_NUM_THREADS=8 \
    PYTHONPATH=src \
    JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_COMPILATION_CACHE_DIR="$cache" \
    GPUWRF_JAX_CACHE_DIR="$cache" \
    GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo \
    GPUWRF_ACOUSTIC_SUBSTEPS="$nsound" \
    GPUWRF_NESTED_AOT=1 \
    GPUWRF_NESTED_PARALLEL_COMPILE=0 \
    python -m gpuwrf.cli run \
      --namelist "$input/namelist.input" \
      --input-dir "$input" \
      --output-dir "$out" \
      --proof-dir "$proof" \
      --scratch-dir "$scratch" \
      --domain d01 \
      --max-dom "$MAXDOM" \
      --hours "$HOURS" \
      > "$log" 2>&1
  rc=$?
  echo "$rc" > "$OUT_ROOT/logs/${tag}.rc"

  local baseline_arg=()
  if [[ "$tag" != "N1" ]]; then
    baseline_arg=(--baseline-dir "$OUT_ROOT/runs/N1")
  fi

  env PYTHONPATH=src JAX_ENABLE_X64=true \
    python proofs/v022/k2_nest_ladder/summarize_nest_rung.py \
      --tag "$tag" \
      --root-dt "$root_dt" \
      --n-sound "$nsound" \
      --run-dir "$run_dir" \
      --hours "$HOURS" \
      --maxdom "$MAXDOM" \
      --rc "$rc" \
      --log "$log" \
      "${baseline_arg[@]}" \
      --out "$result" \
      > "$OUT_ROOT/logs/${tag}_summary.log" 2>&1
  local summary_rc=$?

  python - "$result" "$tag" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
tag = sys.argv[2]
if not p.exists():
    print(f"K2 nest ladder {tag} summary missing")
    raise SystemExit(0)
r = json.loads(p.read_text())
parts = [
    f"K2 nest ladder {tag} {r.get('verdict')}",
    f"root_dt={r.get('root_dt_s_requested')} n_sound={r.get('n_sound')}",
    f"cold_s/fc-h={r.get('cold_inclusive_s_per_fc_h')}",
    f"worst_C_total={r.get('worst_C_total'):.3f}" if isinstance(r.get('worst_C_total'), (int, float)) else "worst_C_total=?",
    f"worst_Cz={r.get('worst_Cz'):.3f}" if isinstance(r.get('worst_Cz'), (int, float)) else "worst_Cz=?",
    f"cz_gt_1={r.get('cz_gt_1')}",
    f"gates={r.get('gates')}",
]
for dom, entry in (r.get("domains") or {}).items():
    w = ((entry.get("courant") or {}).get("worst") or {})
    dt = entry.get("dt_s")
    if w:
        parts.append(f"{dom}:dt={dt},Ctot={w.get('C_total'):.3f},Cz={w.get('Cz'):.3f}")
print("; ".join(parts))
PY
  local one_line
  one_line="$(python - "$result" "$tag" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
tag = sys.argv[2]
if not p.exists():
    print(f"K2 nest ladder {tag}: summary missing")
    raise SystemExit(0)
r = json.loads(p.read_text())
domains = []
for dom, entry in (r.get("domains") or {}).items():
    w = ((entry.get("courant") or {}).get("worst") or {})
    if w:
        domains.append(f"{dom} Ctot={w.get('C_total'):.2f} Cz={w.get('Cz'):.2f}")
gate = r.get("gates") or {}
print(
    f"K2 nest {tag} {r.get('verdict')}: root_dt={r.get('root_dt_s_requested')} "
    f"n_sound={r.get('n_sound')} cold_s/fc-h={r.get('cold_inclusive_s_per_fc_h')} "
    f"worst_Ctot={r.get('worst_C_total'):.2f} worst_Cz={r.get('worst_Cz'):.2f} "
    f"cz_gt_1={r.get('cz_gt_1')} run_ok={gate.get('run_ok')} "
    f"tol={gate.get('within_operational_band_vs_N1')} {'; '.join(domains)}"
)
PY
)"
  notify "$one_line"

  if [[ $rc -ne 0 || $summary_rc -ne 0 ]]; then
    echo "=== ${tag}: stopping ladder after run_rc=${rc} summary_rc=${summary_rc} ==="
    return 1
  fi

  local passes
  passes="$(python - "$result" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("1" if r.get("passes_all") else "0")
PY
)"
  if [[ "$passes" != "1" ]]; then
    echo "=== ${tag}: stopping ladder after failed gate ==="
    return 1
  fi
  return 0
}

cat > "$OUT_ROOT/ladder_config.json" <<EOF
{
  "schema": "v022_k2_nest_ladder_config",
  "input_dir": "$INPUT",
  "hours": $HOURS,
  "maxdom": $MAXDOM,
  "history_interval_min": $HISTORY_INTERVAL_MIN,
  "rung_timeout_s": $RUNG_TIMEOUT,
  "rungs": [
    {"tag": "N1", "root_dt_s": 54, "n_sound": 10},
    {"tag": "N2", "root_dt_s": 60, "n_sound": 9},
    {"tag": "N3", "root_dt_s": 72, "n_sound": 8},
    {"tag": "N4", "root_dt_s": 90, "n_sound": 7}
  ]
}
EOF

notify "K2 nest ladder launch: input=${INPUT}, hours=${HOURS}, maxdom=${MAXDOM}, rungs=N1/N2/N3/N4."

build_input N1 54
build_input N2 60
build_input N3 72
build_input N4 90

run_one N1 54 10 || exit 0
run_one N2 60 9 || exit 0
run_one N3 72 8 || exit 0
run_one N4 90 7 || exit 0

echo "=== K2 NEST LADDER COMPLETE ==="
