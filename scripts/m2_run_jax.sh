#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
VENV="$ROOT/data/scratch/m2-jax-venv"
SCRATCH="$ROOT/data/scratch/m2-jax"
PROF="$ROOT/data/profiler_artifacts/jax"
ART="$ROOT/artifacts/m2/jax"
STENCIL_MANIFEST="$ROOT/fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml"
COLUMN_MANIFEST="$ROOT/fixtures/manifests/analytic-column-thermo-v1.yaml"
STENCIL_FIXTURE="$ROOT/fixtures/samples/analytic-stencil-3d-advdiff-v1.npz"
COLUMN_FIXTURE="$ROOT/fixtures/samples/analytic-column-thermo-v1.npz"

mkdir -p "$SCRATCH" "$PROF" "$ART"

if [[ ! -x "$VENV/bin/python" ]]; then
  python -m venv "$VENV"
fi

if ! "$VENV/bin/python" - <<'PY' >/dev/null 2>&1
import jax
raise SystemExit(0 if jax.__version__ == "0.10.0" else 1)
PY
then
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install 'jax[cuda13]==0.10.0'
fi

PY="$VENV/bin/python"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_FLAGS="--xla_dump_to=$PROF/xla_dump_work --xla_dump_hlo_as_text${XLA_FLAGS:+ $XLA_FLAGS}"

"$PY" -c "import jax; print(jax.default_backend(), jax.devices())"
"$PY" - <<'PY'
import jax
assert jax.default_backend() == "gpu", (jax.default_backend(), jax.devices())
assert any("CudaDevice(id=0" in repr(device) or getattr(device, "id", None) == 0 for device in jax.devices()), jax.devices()
PY

run_ncu() {
  local problem=$1
  local export_base="$PROF/${problem}"
  rm -f "${export_base}.ncu-rep"
  set +e
  ncu --set=full --target-processes=all --force-overwrite --export "$export_base" \
    "$PY" -m gpuwrf.backends.jax.bench --problem "$problem" --skip-artifacts \
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

"$PY" -m gpuwrf.backends.jax.bench \
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
scratch = root / "data" / "scratch" / "m2-jax"
art = root / "artifacts" / "m2" / "jax"

stencil = json.loads((scratch / "stencil_correctness.json").read_text())
column = json.loads((scratch / "column_correctness.json").read_text())
(art / "correctness.json").write_text(
    json.dumps(
        {
            "backend": "jax",
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
