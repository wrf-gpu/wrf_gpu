#!/usr/bin/env bash
# ab_nested_event_tail_guard.sh — READY-TO-FIRE GPU A/B for the v0.20
# "bound the per-step host events/outputs lists" host-RAM guard.
#
#   GROUP   : G_domain_tree
#   LEVER   : bounded per-segment events/outputs tail (host-RAM guard)
#   GATE    : bit_identical (numerics-free; host-only bookkeeping change)
#
# This lever is a STABILITY / host-RAM hardening, NOT a throughput speedup.
# So the A/B proves TWO things on the live nest:
#   (A) THROUGHPUT IS UNCHANGED — warm nest wall_clock_forecast_only_s is within
#       noise WITH vs WITHOUT the guard (the change touches no dispatched op).
#   (B) HOST RAM IS BOUNDED — peak python RSS does not grow with forecast length
#       under the guard, whereas the legacy unbounded tail (GPUWRF_NESTED_EVENT_TAIL=0)
#       lets the host events list grow O(forecast_length).
#
# It is intentionally NOT run here (GPU is held by the live fp32 1km demo).
# Fire it later, under the shared GPU lock, exactly as printed at the bottom.
#
# ---------------------------------------------------------------------------
# Usage (fire later, when the GPU is free):
#   scripts/with_gpu_lock.sh --label evtail-ab -- scripts/ab_nested_event_tail_guard.sh
#
# Tunables (env; all have defaults):
#   INPUT_DIR  nested CPU-WRF/Gen2 case dir (met_em + namelist)   [required-ish]
#   NAMELIST   explicit namelist.input                            [INPUT_DIR/namelist.input]
#   MAX_DOM    nested domain count (>1 to exercise the nest)      [9]
#   HOURS      forecast hours — use a LONG run to expose growth   [24]
#   OUTROOT    output/proof root                                  [/tmp/evtail_ab]
#   PY         python interpreter                                 [python]
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="${INPUT_DIR:-<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z}"
NAMELIST="${NAMELIST:-${INPUT_DIR}/namelist.input}"
MAX_DOM="${MAX_DOM:-9}"
HOURS="${HOURS:-24}"
OUTROOT="${OUTROOT:-/tmp/evtail_ab}"
PY="${PY:-python}"

mkdir -p "$OUTROOT"
echo "== A/B: nested event-tail host-RAM guard =="
echo "   input-dir : $INPUT_DIR"
echo "   namelist  : $NAMELIST"
echo "   max-dom   : $MAX_DOM   hours: $HOURS"
echo "   outroot   : $OUTROOT"
echo

# Sample peak RSS of the forecast process while it runs (host-RAM growth probe).
_run_one () {
  local label="$1" tail_env="$2" outdir="$OUTROOT/$label"
  mkdir -p "$outdir"
  echo "-- run [$label]  GPUWRF_NESTED_EVENT_TAIL=$tail_env --"
  # /usr/bin/time -v reports "Maximum resident set size" = peak host RSS (KB).
  GPUWRF_NESTED_EVENT_TAIL="$tail_env" \
    /usr/bin/time -v "$PY" -m gpuwrf.cli run \
      --input-dir "$INPUT_DIR" \
      --namelist "$NAMELIST" \
      --max-dom "$MAX_DOM" \
      --hours "$HOURS" \
      --output-dir "$outdir" \
      > "$outdir/stdout.log" 2> "$outdir/time.log" || {
        echo "   run [$label] FAILED — see $outdir/time.log"; return 1; }
  # Pull the forecast-only wall time from the proof JSON
  # (cli.py writes <output-dir>/proofs/nested_pipeline_run.json).
  local proof
  proof="$(ls -t "$outdir"/proofs/nested_pipeline_run.json "$outdir"/proofs/*.json "$outdir"/*.json 2>/dev/null | head -1 || true)"
  local fc_s peak_kb
  fc_s="$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('wall_clock_forecast_only_s','?'))" "$proof" 2>/dev/null || echo '?')"
  peak_kb="$(grep -i 'Maximum resident set size' "$outdir/time.log" | grep -oE '[0-9]+' | head -1 || echo '?')"
  echo "   forecast_only_s = $fc_s   peak_host_RSS_KB = $peak_kb   proof = $proof"
  echo "$label $fc_s $peak_kb" >> "$OUTROOT/summary.txt"
}

: > "$OUTROOT/summary.txt"
# WITH guard = the v0.20 default (bounded tail, e.g. 4096). Cold compile lands
# here; run it FIRST so the second run is warm (same XLA cache) -> fair wall A/B.
_run_one with_guard ""        # empty -> default cap 4096 (bounded)
_run_one no_guard   "0"       # 0 -> unbounded legacy tail (host list grows)

echo
echo "== SUMMARY (label  forecast_only_s  peak_host_RSS_KB) =="
cat "$OUTROOT/summary.txt"
echo
echo "PASS CRITERIA:"
echo "  (A) forecast_only_s(with_guard) ~= forecast_only_s(no_guard)  [within ~2% noise — numerics-free]"
echo "  (B) peak_host_RSS_KB(with_guard) <= peak_host_RSS_KB(no_guard) [guard caps host events RAM;"
echo "      the gap WIDENS as HOURS grows — re-run with HOURS=120 to make it stark]"
