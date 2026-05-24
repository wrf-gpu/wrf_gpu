#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATCH_ROOT="$ROOT/external/wrf_savepoint_patch"
OUT="$PATCH_ROOT/build"
MAIN="$OUT/main"
CANONICAL="/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF"
ENV_SCRIPT="/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh"
STABLE="/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe"
EXPECTED_STABLE_SHA="1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37"
EXPECTED_WRF_HEAD="115e5756f98ee2370d62b6709baac6417d8f7338"

mkdir -p "$MAIN" "$OUT/proofs"

stable_before="$(sha256sum "$STABLE" | awk '{print $1}')"
if [[ "$stable_before" != "$EXPECTED_STABLE_SHA" ]]; then
  echo "FATAL: operational wrf.exe sha changed BEFORE M6B0-R build: $stable_before" >&2
  exit 1
fi

wrf_head="$(git -C "$CANONICAL" rev-parse HEAD)"
if [[ "$wrf_head" != "$EXPECTED_WRF_HEAD" ]]; then
  echo "FATAL: canonical WRF source drifted: $wrf_head" >&2
  exit 1
fi

# shellcheck source=/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
source "$ENV_SCRIPT"

cat > "$OUT/preflight.json" <<JSON
{
  "operational_wrf": "$STABLE",
  "operational_sha256_before": "$stable_before",
  "expected_operational_sha256": "$EXPECTED_STABLE_SHA",
  "canonical_wrf": "$CANONICAL",
  "canonical_wrf_head": "$wrf_head",
  "expected_wrf_head": "$EXPECTED_WRF_HEAD",
  "env_script": "$ENV_SCRIPT",
  "nvfortran": "$(command -v nvfortran)",
  "h5fc": "$(command -v h5fc)"
}
JSON

# This sprint keeps the operational WRF immutable and records the source patch
# artifacts needed for the relinked WRF lane. The executable below is a CPU
# savepoint emission shim used by the extraction stage, not the protected WRF.
(
  cd "$OUT"
  h5fc -O2 -cpp -DWRF_SAVEPOINT \
    "$PATCH_ROOT/dyn_em/savepoint_wrapper.F90" \
    -o "$MAIN/wrf.exe.instrumented"
)

stable_after="$(sha256sum "$STABLE" | awk '{print $1}')"
if [[ "$stable_after" != "$EXPECTED_STABLE_SHA" ]]; then
  echo "FATAL: operational wrf.exe was modified during M6B0-R build: $stable_after" >&2
  exit 2
fi

instrumented_hash="$(sha256sum "$MAIN/wrf.exe.instrumented" | awk '{print $1}')"

cat > "$OUT/build_registry.json" <<JSON
{
  "instrumented_wrf": "$MAIN/wrf.exe.instrumented",
  "instrumented_sha256": "$instrumented_hash",
  "strategy": "Fortran wrapper module gated by WRF_SAVEPOINT; CPU savepoint emission shim for M6B0-R extraction",
  "patches": [
    "$PATCH_ROOT/solve_em.F.patch",
    "$PATCH_ROOT/configure.wrf.patch"
  ],
  "operational_sha256_after": "$stable_after",
  "cpu_path_namelist": "$PATCH_ROOT/namelist.savepoint"
}
JSON

echo "M6B0-R preflight OK"
echo "stable_wrf=$STABLE"
echo "stable_sha256_before=$stable_before"
echo "stable_sha256_after=$stable_after"
echo "canonical_wrf=$CANONICAL"
echo "canonical_wrf_head=$wrf_head"
echo "env_script=$ENV_SCRIPT"
echo "nvfortran=$(command -v nvfortran)"
echo "h5fc=$(command -v h5fc)"
echo "instrumented_wrf=$MAIN/wrf.exe.instrumented"
echo "instrumented_sha256=$instrumented_hash"
