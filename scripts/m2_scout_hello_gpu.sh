#!/usr/bin/env bash
set -u

ROOT=$(git rev-parse --show-toplevel)
VENV="$ROOT/data/scratch/m2-scout-venv"
HELLO_ROOT="$ROOT/artifacts/m2/scout/hello_gpu"
MATRIX="$ROOT/artifacts/m2/scout/toolchain_support_matrix.json"
CANDIDATES=(jax triton gt4py kokkos cupy_or_numba cuda_tile)

is_blocked() {
  local name=$1
  [ -f "$MATRIX" ] || return 1
  "$VENV/bin/python" - "$MATRIX" "$name" <<'PY'
import json
import sys

matrix = json.load(open(sys.argv[1]))
for candidate in matrix["candidates"]:
    if candidate["name"] == sys.argv[2]:
        raise SystemExit(0 if candidate["verdict"] == "blocked" else 1)
raise SystemExit(1)
PY
}

run_candidate() {
  local name=$1
  local dir="$HELLO_ROOT/$name"
  local tmp="$dir/output.tmp"
  local exit_file="$dir/exit.txt"
  mkdir -p "$dir"
  : >"$tmp"

  (
    cd "$dir" || exit 1
    case "$name" in
      jax|triton|gt4py|cupy_or_numba)
        source "$VENV/bin/activate"
        python hello.py
        ;;
      kokkos|cuda_tile)
        bash build.sh >build.log 2>&1
        if [ "$name" = kokkos ]; then
          "$ROOT/data/scratch/m2-scout-build/hello_gpu/kokkos/hello_kokkos"
        else
          "$ROOT/data/scratch/m2-scout-build/hello_gpu/cuda_tile/hello_cuda_tile"
        fi
        ;;
      *)
        echo "unknown candidate $name" >&2
        exit 2
        ;;
    esac
  ) >"$tmp" 2>&1
  local code=$?
  mv "$tmp" "$dir/output.txt"
  printf '%s\n' "$code" >"$exit_file"
  return "$code"
}

pass=0
fail=0
for candidate in "${CANDIDATES[@]}"; do
  if is_blocked "$candidate"; then
    echo "$candidate: skipped blocked"
    continue
  fi
  if run_candidate "$candidate"; then
    pass=$((pass + 1))
    echo "$candidate: pass"
  else
    fail=$((fail + 1))
    echo "$candidate: fail"
  fi
done

echo "m2 scout hello-gpu: $pass pass, $fail fail"
[ "$fail" -eq 0 ]
