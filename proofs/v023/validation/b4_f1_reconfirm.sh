#!/usr/bin/env bash
# B4: F1 fresh paired reconfirm (optional).
#
# Uses the v023 F1 batch-sweep driver if present in HEAD. The driver is expected
# to write JSON/JSONL with B-sweep throughput and VRAM. This script is optional
# but turnkey once the F1 harness is present.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  F1_DRIVER    default: proofs/v023/gpu_gates/g3_batch_sweep_driver.py
  F1_INPUT     default: <DATA_ROOT>/wrf_downscale/f1_retest/tenerife_20250121_3km1km/real_run
  F1_B_VALUES  default: "1 2 3 4 5"

Acceptance: reproduces the accepted B-sweep band, including B=1 around 4.51M
and best around 5.63M weighted cells/s.
EOF
  v023_usage_common
}

for arg in "$@"; do
  if v023_parse_common_flag "$arg"; then continue; fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then usage; exit 0; fi
  echo "unknown argument: $arg" >&2; usage >&2; exit 2
done

GATE="b4_f1_reconfirm"
v023_gate_paths "$GATE"
F1_DRIVER="${F1_DRIVER:-proofs/v023/gpu_gates/g3_batch_sweep_driver.py}"
F1_INPUT="${F1_INPUT:-<DATA_ROOT>/wrf_downscale/f1_retest/tenerife_20250121_3km1km/real_run}"
F1_B_VALUES="${F1_B_VALUES:-1 2 3 4 5}"
OUT="$V023_RUN_DIR/f1_bench.jsonl"

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "Would run $F1_DRIVER for B values: $F1_B_VALUES"
  exit 0
fi

if [[ ! -f "$V023_ROOT/$F1_DRIVER" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "BLOCKED" "MISSING_F1_DRIVER" \
    "Expected F1 driver at $F1_DRIVER after F1 integration."
  exit 2
fi

"$V023_GPU_LOCK" --timeout "$V023_LOCK_TIMEOUT" --label "v023-b4-f1" -- bash -lc "
set -euo pipefail
cd '$V023_ROOT'
: > '$OUT'
for b in $F1_B_VALUES; do
  taskset -c '$V023_TASKSET' env \
    PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    GPUWRF_BATCH_ENSEMBLE=\"\$b\" GPUWRF_NESTED_AOT=0 \
    G3_INPUT_DIR='$F1_INPUT' G3_OUTPUT='$V023_RUN_DIR/b'\${b}'.json' \
    G3_MAXDOM='${F1_MAXDOM:-2}' G3_ROOT_STEPS='${F1_ROOT_STEPS:-400}' G3_REPEATS='${F1_REPEATS:-2}' \
    python '$F1_DRIVER' \
    >> '$V023_RUN_DIR/b\${b}.log' 2>&1
  cat '$V023_RUN_DIR/b\${b}.json' >> '$OUT'
  printf '\n' >> '$OUT'
done
"
rc=$?

python3 - "$V023_VERDICT_JSON" "$OUT" "$rc" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, rows_path, rc = sys.argv[1:4]
rows = []
if Path(rows_path).is_file():
    rows = [json.loads(line) for line in Path(rows_path).read_text().splitlines() if line.strip()]
values = [
    r.get("warm_cells_per_sec")
    or r.get("total_cells_per_sec")
    or (r.get("cell_metrics") or {}).get("warm_cells_per_sec")
    or (r.get("cell_metrics") or {}).get("total_cells_per_sec")
    for r in rows
]
best = max([float(v) for v in values if isinstance(v, (int, float))], default=None)
payload = {
    "schema": "v023-b4-f1-reconfirm-v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "gate": "B4",
    "returncode": int(rc),
    "rows_jsonl": rows_path,
    "row_count": len(rows),
    "best_cells_per_sec": best,
    "expected_best_cells_per_sec": 5.63e6,
    "verdict": "PASS" if int(rc) == 0 and best is not None and best >= 5.4e6 else "FAIL",
    "rows": rows,
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"gate": "B4", "verdict": payload["verdict"], "output": out}, sort_keys=True))
sys.exit(0 if payload["verdict"] == "PASS" else 1)
PY
