#!/usr/bin/env bash
set -euo pipefail

if [[ -f /home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh ]]; then
  # shellcheck disable=SC1091
  source /home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
fi

ROOT=$(git rev-parse --show-toplevel)
VENV="$ROOT/data/scratch/m2-cupy-venv"
SCRATCH="$ROOT/data/scratch/m2-cupy"
PROF="$ROOT/data/profiler_artifacts/cupy_or_numba"
ART="$ROOT/artifacts/m2/cupy_or_numba"
STENCIL_MANIFEST="$ROOT/fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml"
COLUMN_MANIFEST="$ROOT/fixtures/manifests/analytic-column-thermo-v1.yaml"
STENCIL_FIXTURE="$ROOT/fixtures/samples/analytic-stencil-3d-advdiff-v1.npz"
COLUMN_FIXTURE="$ROOT/fixtures/samples/analytic-column-thermo-v1.npz"

mkdir -p "$SCRATCH" "$PROF" "$ART"

if [[ ! -x "$VENV/bin/python" ]]; then
  python -m venv "$VENV"
fi

if ! "$VENV/bin/python" - <<'PY' >/dev/null 2>&1
import cupy
raise SystemExit(0 if cupy.__version__ == "14.0.1" else 1)
PY
then
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install cupy-cuda13x==14.0.1
fi

PY="$VENV/bin/python"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"

run_ncu() {
  local problem=$1
  local export_base="$PROF/${problem}"
  rm -f "${export_base}.ncu-rep"
  set +e
  ncu --set=full --target-processes=all --force-overwrite --export "$export_base" \
    "$PY" -m gpuwrf.backends.cupy.bench --problem "$problem" --skip-artifacts \
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

"$PY" -m gpuwrf.backends.cupy.bench \
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
scratch = root / "data" / "scratch" / "m2-cupy"
art = root / "artifacts" / "m2" / "cupy_or_numba"

stencil = json.loads((scratch / "stencil_correctness.json").read_text())
column = json.loads((scratch / "column_correctness.json").read_text())
(art / "correctness.json").write_text(
    json.dumps(
        {
            "backend": "cupy",
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
