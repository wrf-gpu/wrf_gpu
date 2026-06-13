#!/usr/bin/env bash
# v0.15 final-gate Canary L2 d02 72h post-processing (CPU-only; no GPU lock).
# Runs: grid compare -> Grid-Delta Atlas -> identity-proof dashboards.
# Replicates the v0.14 recipe (proofs/v014/canary_d02_72h_field_gate_summary.md,
# docs/IDENTITY_PROOF.md) against the v0.15 paired run + the FROZEN manifest.
set -euo pipefail

RR="$1"                       # GPU run root
RUN_ID=20260501_18z_l2_72h_20260519T173026Z
CPU_DIR=/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/${RUN_ID}
GPU_DIR="${RR}/gpu_output/l2_d02_${RUN_ID}"
MANIFEST=proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json
INIT=2026-05-01T18:00:00+00:00

echo "[canary-post] compare ..."
taskset -c 0-3 python3 scripts/compare_wrfout_grid.py \
  --cpu-dir "$CPU_DIR" --gpu-dir "$GPU_DIR" \
  --domain d02 --init "$INIT" --min-lead 1 --max-lead 72 \
  --tolerance-json "$MANIFEST" \
  --out-json "${RR}/canary_d02_72h_grid_compare.json" \
  --out-md   "${RR}/canary_d02_72h_grid_compare.md" \
  > "${RR}/canary_d02_72h_compare.log" 2>&1
echo $? > "${RR}/canary_d02_72h_compare.rc"

echo "[canary-post] atlas ..."
taskset -c 0-3 python3 scripts/build_grid_delta_atlas.py \
  --cpu-dir "$CPU_DIR" --gpu-dir "$GPU_DIR" \
  --case-id canary_d02_20260501_18z \
  --domain d02 --init "$INIT" --min-lead 1 --max-lead 72 \
  --tolerance-json "$MANIFEST" \
  --proof-dir "${RR}/grid_delta_atlas" \
  --asset-dir "${RR}/grid_delta_atlas_assets" \
  > "${RR}/canary_d02_72h_atlas.log" 2>&1
echo $? > "${RR}/canary_d02_72h_atlas.rc"

echo "[canary-post] identity-proof dashboards ..."
taskset -c 0-3 python3 scripts/build_identity_proof_plots.py \
  --cpu-dir "$CPU_DIR" --gpu-dir "$GPU_DIR" \
  --domain d02 --init "$INIT" \
  --case-id canary_l2_d02_72h \
  --region-label "Canary L2 d02 72h (2026-05-01 18Z, v0.15)" \
  --tolerance-json "$MANIFEST" \
  --proof-dir proofs/v015/identity_proof/canary_l2_d02 \
  --asset-dir docs/assets/v015/identity_proof/canary_l2_d02 \
  > "${RR}/canary_d02_72h_identity.log" 2>&1
echo $? > "${RR}/canary_d02_72h_identity.rc"

echo "[canary-post] DONE compare_rc=$(cat ${RR}/canary_d02_72h_compare.rc) atlas_rc=$(cat ${RR}/canary_d02_72h_atlas.rc) identity_rc=$(cat ${RR}/canary_d02_72h_identity.rc)"
