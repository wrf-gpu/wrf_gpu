#!/usr/bin/env bash
# Tester validation harness for sprint 2026-05-28-gpu-wrf-history-research.
#
# Mechanical checks against the sprint contract's AC1-AC7. Designed to be re-run
# by any tester pass once the worker has produced the deliverables. Returns
# non-zero on any failure so a CI-style invocation can gate the worker report.
#
# Usage:
#   bash .agent/sprints/2026-05-28-gpu-wrf-history-research/tests/validate_deliverables.sh
#
# Exit codes:
#   0  all checks passed
#   1  one or more deliverables missing
#   2  deliverable present but fails content gates
#   3  worker-report missing or below threshold
#
# Pinned to CPU cores 0-3 per project rule.

set -u
SPRINT_DIR="/tmp/wrf_gpu2_history/.agent/sprints/2026-05-28-gpu-wrf-history-research"
fail=0
content_fail=0

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }

check_file() {
  local rel="$1" min_bytes="$2" must_contain="$3"
  local path="$SPRINT_DIR/$rel"
  if [[ ! -f "$path" ]]; then
    red "MISS  $rel (file absent)"
    fail=$((fail+1))
    return
  fi
  local size
  size=$(stat -c %s "$path" 2>/dev/null || echo 0)
  if (( size < min_bytes )); then
    red "SMALL $rel (${size}B < ${min_bytes}B)"
    content_fail=$((content_fail+1))
    return
  fi
  if [[ -n "$must_contain" ]] && ! grep -q -E "$must_contain" "$path"; then
    red "MISSING-TOKEN $rel (regex: $must_contain)"
    content_fail=$((content_fail+1))
    return
  fi
  green "OK    $rel (${size}B)"
}

echo "== AC1-AC6: worker deliverables =="
# AC1: history narrative, ~1500 words. 1500 words ~= 9000 bytes minimum.
check_file "gpu_wrf_port_history.md"       8000  "WRF|NCAR|GPU|OpenACC|AceCAST"
# AC2: catalogue table, >=8 rows. Min size ~3 KB for a populated markdown table.
check_file "gpu_wrf_port_catalogue.md"     3000  "\| *Year *\||AceCAST|OpenACC"
# AC3: novelty bounds with three claim options.
check_file "novelty_bounds.md"             2500  "[Oo]ption *1|[Aa]ggressive|[Dd]efensible"
# AC4: four-section why-hard analysis.
check_file "why_it_is_hard.md"             3000  "math|physics|coding|organisational"
# AC6: multi-agent framing memo.
check_file "multi_agent_framing.md"        2000  "frontrunner|critic|feedback"
# AC5: bibtex stub list (may be inline in another file but a separate one is cleanest).
check_file "citations_to_add.md"           500   "@(article|misc|inproceedings|manual|techreport|software)"

echo
echo "== AC7: tester-facing surface =="
WR="$SPRINT_DIR/worker-report.md"
if [[ ! -f "$WR" ]]; then
  red "MISS  worker-report.md (worker has not delivered)"
  exit 3
fi
wr_size=$(stat -c %s "$WR")
if (( wr_size < 1000 )); then
  red "SMALL worker-report.md (${wr_size}B < 1000B)"
  content_fail=$((content_fail+1))
fi
if ! grep -q -E "^(Decision|## Decision|Summary):" "$WR"; then
  red "MISSING-TOKEN worker-report.md (no Decision: line)"
  content_fail=$((content_fail+1))
fi

echo
echo "== Hard-rule checks: no fabricated citations =="
# Every BibTeX-shaped citation key cited in deliverables must appear in either:
#   - publication/draft/references.bib (already curated), or
#   - .agent/sprints/.../citations_to_add.md (new proposals)
BIB="/tmp/wrf_gpu2_history/publication/draft/references.bib"
CITES_NEW="$SPRINT_DIR/citations_to_add.md"
known=$(mktemp)
if [[ -f "$BIB" ]]; then grep -hoE '^@[a-zA-Z]+\{[^,]+' "$BIB" | sed 's/^@[a-zA-Z]*{//' >>"$known"; fi
if [[ -f "$CITES_NEW" ]]; then grep -hoE '^@[a-zA-Z]+\{[^,]+' "$CITES_NEW" | sed 's/^@[a-zA-Z]*{//' >>"$known"; fi
# Find every \cite{...} or [[cite:KEY]] reference in deliverable bodies.
referenced=$(mktemp)
for f in "$SPRINT_DIR"/gpu_wrf_port_history.md "$SPRINT_DIR"/gpu_wrf_port_catalogue.md \
         "$SPRINT_DIR"/novelty_bounds.md "$SPRINT_DIR"/why_it_is_hard.md \
         "$SPRINT_DIR"/multi_agent_framing.md; do
  [[ -f "$f" ]] || continue
  grep -hoE '\\cite\{[^}]+\}|\[cite: *[^]]+\]' "$f" 2>/dev/null \
    | sed -E 's/\\cite\{//; s/\[cite: *//; s/[\}\]]//g; s/,/\n/g' \
    | tr -d ' ' >>"$referenced"
done
sort -u "$referenced" >"$referenced.s" && mv "$referenced.s" "$referenced"
missing_keys=$(comm -23 "$referenced" <(sort -u "$known")) || true
if [[ -n "$missing_keys" ]]; then
  red "UNVERIFIED CITATION KEYS (not in references.bib or citations_to_add.md):"
  echo "$missing_keys" | sed 's/^/    - /'
  content_fail=$((content_fail+1))
fi
rm -f "$known" "$referenced"

echo
echo "== Catalogue row count (AC2 >=8) =="
CAT="$SPRINT_DIR/gpu_wrf_port_catalogue.md"
if [[ -f "$CAT" ]]; then
  rows=$(awk '/^\|/{c++} END{print c+0}' "$CAT")
  # Subtract the header row and the alignment row.
  data_rows=$((rows - 2))
  if (( data_rows >= 8 )); then
    green "OK    catalogue rows: ${data_rows}"
  else
    red "FEW   catalogue rows: ${data_rows} < 8"
    content_fail=$((content_fail+1))
  fi
fi

echo
echo "== Honesty check: stronger-than-evidence claim words =="
# The user's verbal claim "no full GPU port even commercially available" is
# strictly stronger than the brief supports (AceCAST is a commercial product).
# A defensible novelty_bounds.md must explicitly acknowledge AceCAST and
# differentiate the claim wording. Flag if novelty_bounds.md uses "first GPU
# port of WRF" without qualification.
NB="$SPRINT_DIR/novelty_bounds.md"
if [[ -f "$NB" ]]; then
  if grep -qE -i 'first +(full +)?(open-source )?GPU +port +of +WRF' "$NB" \
     && ! grep -qi 'AceCAST' "$NB"; then
    red "CLAIM-OVERREACH novelty_bounds.md asserts 'first GPU port of WRF' without acknowledging AceCAST"
    content_fail=$((content_fail+1))
  fi
fi

echo
if (( fail == 0 && content_fail == 0 )); then
  green "== ALL CHECKS PASSED =="
  exit 0
elif (( fail > 0 )); then
  red "== ${fail} MISSING DELIVERABLE(S), ${content_fail} CONTENT FAILURE(S) =="
  exit 1
else
  yellow "== ALL FILES PRESENT, ${content_fail} CONTENT FAILURE(S) =="
  exit 2
fi
