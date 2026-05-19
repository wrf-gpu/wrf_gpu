#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
VENV="$ROOT/data/scratch/m2-triton-venv"
SCRATCH="$ROOT/data/scratch/m2-triton"
PROF="$ROOT/data/profiler_artifacts/triton"
ART="$ROOT/artifacts/m2/triton"
CACHE="$ROOT/data/scratch/m2-triton-cache"
STENCIL_MANIFEST="$ROOT/fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml"
COLUMN_MANIFEST="$ROOT/fixtures/manifests/analytic-column-thermo-v1.yaml"
STENCIL_FIXTURE="$ROOT/fixtures/samples/analytic-stencil-3d-advdiff-v1.npz"
COLUMN_FIXTURE="$ROOT/fixtures/samples/analytic-column-thermo-v1.npz"

mkdir -p "$SCRATCH" "$PROF" "$ART" "$CACHE"

if [[ ! -x "$VENV/bin/python" ]]; then
  python -m venv "$VENV"
fi

if ! "$VENV/bin/python" - <<'PY' >/dev/null 2>&1
import numpy
import torch
import triton
ok = triton.__version__ == "3.7.0" and torch.__version__.split("+", 1)[0] == "2.12.0"
raise SystemExit(0 if ok else 1)
PY
then
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install 'numpy>=1.24' 'triton==3.7.0' 'torch==2.12.0'
fi

PY="$VENV/bin/python"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$CACHE"

"$PY" -c "import triton, torch; assert torch.cuda.is_available(); print(triton.__version__, torch.version.cuda)"

run_ncu() {
  local problem=$1
  local export_base="$PROF/${problem}"
  rm -f "${export_base}.ncu-rep"
  set +e
  ncu --set=full --target-processes=all --force-overwrite --export "$export_base" \
    "$PY" -m gpuwrf.backends.triton.bench --problem "$problem" --skip-artifacts \
      --scratch "$SCRATCH" --artifact-dir "$ART" --profiler-dir "$PROF" \
    > "$PROF/${problem}_ncu_stdout.txt" 2> "$PROF/${problem}_ncu_stderr.txt"
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$PROF/${problem}_ncu_exit.txt"
}

if command -v ncu >/dev/null 2>&1; then
  run_ncu stencil
  run_ncu column
else
  printf 'ncu not found on PATH\n' > "$PROF/stencil_ncu_stderr.txt"
  printf 'ncu not found on PATH\n' > "$PROF/column_ncu_stderr.txt"
  printf '127\n' > "$PROF/stencil_ncu_exit.txt"
  printf '127\n' > "$PROF/column_ncu_exit.txt"
  : > "$PROF/stencil_ncu_stdout.txt"
  : > "$PROF/column_ncu_stdout.txt"
fi

rm -rf "$CACHE"
mkdir -p "$CACHE"

"$PY" -m gpuwrf.backends.triton.bench \
  --problem both \
  --stencil-fixture "$STENCIL_FIXTURE" \
  --column-fixture "$COLUMN_FIXTURE" \
  --scratch "$SCRATCH" \
  --artifact-dir "$ART" \
  --profiler-dir "$PROF"

python -m gpuwrf.validation.compare_fixture \
  --manifest "$STENCIL_MANIFEST" \
  --candidate "$SCRATCH/stencil_out.npz" \
  --reference "$STENCIL_FIXTURE" \
  --out "$SCRATCH/stencil_correctness.json"

python -m gpuwrf.validation.compare_fixture \
  --manifest "$COLUMN_MANIFEST" \
  --candidate "$SCRATCH/column_out.npz" \
  --reference "$COLUMN_FIXTURE" \
  --out "$SCRATCH/column_correctness.json"

python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
scratch = root / "data" / "scratch" / "m2-triton"
art = root / "artifacts" / "m2" / "triton"

stencil = json.loads((scratch / "stencil_correctness.json").read_text())
column = json.loads((scratch / "column_correctness.json").read_text())
(art / "correctness.json").write_text(
    json.dumps(
        {
            "backend": "triton",
            "pass": bool(stencil["pass"] and column["pass"]),
            "stencil": stencil,
            "column": column,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

"$PY" -m pip freeze > "$SCRATCH/pip_freeze.txt"
