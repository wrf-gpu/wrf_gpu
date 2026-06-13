#!/usr/bin/env bash
# v0.15 final-gate Switzerland d01 72h post-processing (CPU-only; no GPU lock).
# Runs: grid compare -> Grid-Delta Atlas -> identity-proof dashboards.
# Replicates the v0.14 recipe (proofs/v014/switzerland_72h_thetafix_gate.md,
# docs/IDENTITY_PROOF.md) against the v0.15 paired run + the FROZEN manifest.
set -euo pipefail

RR="$1"                       # GPU run root
CPU_DIR=/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu
GPU_DIR="${RR}/gpu_output"
MANIFEST=proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json
INIT=2023-01-15T00:00:00+00:00

echo "[switz-post] compare ..."
taskset -c 0-3 python3 scripts/compare_wrfout_grid.py \
  --cpu-dir "$CPU_DIR" --gpu-dir "$GPU_DIR" \
  --domain d01 --init "$INIT" --min-lead 1 --max-lead 72 \
  --tolerance-json "$MANIFEST" \
  --out-json "${RR}/switzerland_d01_72h_grid_compare.json" \
  --out-md   "${RR}/switzerland_d01_72h_grid_compare.md" \
  > "${RR}/switzerland_d01_72h_compare.log" 2>&1
echo $? > "${RR}/switzerland_d01_72h_compare.rc"

echo "[switz-post] atlas ..."
taskset -c 0-3 python3 scripts/build_grid_delta_atlas.py \
  --cpu-dir "$CPU_DIR" --gpu-dir "$GPU_DIR" \
  --case-id switzerland_d01_20230115_00z \
  --domain d01 --init "$INIT" --min-lead 1 --max-lead 72 \
  --tolerance-json "$MANIFEST" \
  --proof-dir "${RR}/grid_delta_atlas" \
  --asset-dir "${RR}/grid_delta_atlas_assets" \
  > "${RR}/switzerland_d01_72h_atlas.log" 2>&1
echo $? > "${RR}/switzerland_d01_72h_atlas.rc"

echo "[switz-post] identity-proof dashboards ..."
taskset -c 0-3 python3 scripts/build_identity_proof_plots.py \
  --cpu-dir "$CPU_DIR" --gpu-dir "$GPU_DIR" \
  --domain d01 --init "$INIT" \
  --case-id switzerland_d01_72h \
  --region-label "Switzerland d01 72h (2023-01-15 00Z, v0.15)" \
  --tolerance-json "$MANIFEST" \
  --proof-dir proofs/v015/identity_proof/switzerland_d01 \
  --asset-dir docs/assets/v015/identity_proof/switzerland_d01 \
  > "${RR}/switzerland_d01_72h_identity.log" 2>&1
echo $? > "${RR}/switzerland_d01_72h_identity.rc"

echo "[switz-post] DONE compare_rc=$(cat ${RR}/switzerland_d01_72h_compare.rc) atlas_rc=$(cat ${RR}/switzerland_d01_72h_atlas.rc) identity_rc=$(cat ${RR}/switzerland_d01_72h_identity.rc)"
