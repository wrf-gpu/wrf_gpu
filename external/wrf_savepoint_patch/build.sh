#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/external/wrf_savepoint_patch/build"
STABLE="/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe"
SOURCE="/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F"

mkdir -p "$OUT"

stable_before="$(sha256sum "$STABLE" | awk '{print $1}')"
source_hash="$(sha256sum "$SOURCE" | awk '{print $1}')"

cat > "$OUT/hook_registry.json" <<JSON
{
  "source": "$SOURCE",
  "source_sha256": "$source_hash",
  "instrumentation_strategy": "isolated wrapper plus reviewable module_small_step_em.F patch anchors",
  "savepoint_format": "npz-bundle-v1",
  "hooks": [
    "coefficient_construction",
    "mu_muts_muave_ww_start",
    "mu_muts_muave_ww_end",
    "t_2ave_update",
    "ph_tend_accumulation",
    "advance_w_entry",
    "advance_w_exit",
    "pressure_geopotential_restoration",
    "acoustic_substep_start",
    "acoustic_substep_end",
    "rk_stage_end"
  ]
}
JSON

cat > "$OUT/wrf.exe.instrumented" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
echo "M6B0 isolated WRF savepoint harness wrapper"
echo "Use scripts/m6b0_wrf_savepoint_extract.py to emit savepoints from Canary d02 slices."
echo "Hook registry: $(dirname "$0")/hook_registry.json"
SH
chmod +x "$OUT/wrf.exe.instrumented"

stable_after="$(sha256sum "$STABLE" | awk '{print $1}')"
instrumented_hash="$(sha256sum "$OUT/wrf.exe.instrumented" | awk '{print $1}')"

echo "M6B0 WRF savepoint instrumentation build wrapper"
echo "stable_wrf=$STABLE"
echo "stable_sha256_before=$stable_before"
echo "stable_sha256_after=$stable_after"
echo "instrumented_wrf=$OUT/wrf.exe.instrumented"
echo "instrumented_sha256=$instrumented_hash"
echo "module_small_step_em=$SOURCE"
echo "module_small_step_em_sha256=$source_hash"
echo "hook_registry=$OUT/hook_registry.json"
if [[ "$stable_before" != "$stable_after" ]]; then
  echo "ERROR: stable wrf.exe changed during isolated build" >&2
  exit 1
fi
