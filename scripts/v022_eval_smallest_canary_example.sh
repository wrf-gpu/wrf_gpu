#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${V022_EVAL_OUT:-$ROOT/proofs/v022/operational_relaxed/smallest_canary_example}"
LEADS=(${V022_EVAL_LEADS:-24 72 120})
FIELDS=(${V022_EVAL_FIELDS:-T2 U10 V10})

if [[ -n "${V022_EVAL_CANDIDATE_DIR:-}" && -n "${V022_EVAL_CPU_WRF_DIR:-}" ]]; then
  STRICT_ARGS=()
  RESTART_ARGS=()
  GUARD_ARGS=()
  INIT_ARGS=()
  if [[ -n "${V022_EVAL_STRICT_DIR:-}" ]]; then
    STRICT_ARGS+=(--strict-dir "$V022_EVAL_STRICT_DIR")
  fi
  if [[ -n "${V022_EVAL_RESTART_DIR:-}" ]]; then
    RESTART_ARGS+=(--restart-dir "$V022_EVAL_RESTART_DIR")
  fi
  if [[ -n "${V022_EVAL_CANDIDATE_GUARDS:-}" ]]; then
    GUARD_ARGS+=(--candidate-guards "$V022_EVAL_CANDIDATE_GUARDS")
  fi
  if [[ -n "${V022_EVAL_STRICT_GUARDS:-}" ]]; then
    GUARD_ARGS+=(--strict-guards "$V022_EVAL_STRICT_GUARDS")
  fi
  if [[ -n "${V022_EVAL_INIT:-}" ]]; then
    INIT_ARGS+=(--init "$V022_EVAL_INIT")
  fi
  python "$ROOT/scripts/v022_operational_relaxed_gate.py" \
    --candidate-dir "$V022_EVAL_CANDIDATE_DIR" \
    --cpu-wrf-dir "$V022_EVAL_CPU_WRF_DIR" \
    "${STRICT_ARGS[@]}" \
    "${RESTART_ARGS[@]}" \
    "${GUARD_ARGS[@]}" \
    --out "$OUT" \
    --case-id "${V022_EVAL_CASE_ID:-CANARY-L2-D02-SMALLEST-TILE}" \
    --lever-id "${V022_EVAL_LEVER_ID:-v022_eval_harness_example}" \
    --gate-mode "${V022_EVAL_GATE_MODE:-cpu-wrf-backlog}" \
    --domains "${V022_EVAL_DOMAIN:-d02}" \
    --leads "${LEADS[@]}" \
    --fields "${FIELDS[@]}" \
    "${INIT_ARGS[@]}"
else
  SYNTH_ROOT="${V022_EVAL_SYNTHETIC_ROOT:-/tmp/gpuwrf_v022_eval_smallest_canary}"
  rm -rf "$SYNTH_ROOT"
  python "$ROOT/scripts/v022_operational_relaxed_gate.py" \
    --synthetic-example-root "$SYNTH_ROOT" \
    --out "$OUT" \
    --case-id "CANARY-L2-D02-SMALLEST-TILE-SYNTHETIC" \
    --lever-id "v022_eval_harness_example" \
    --gate-mode "operational-relaxed" \
    --domains d02 \
    --leads "${LEADS[@]}" \
    --fields "${FIELDS[@]}"
fi
