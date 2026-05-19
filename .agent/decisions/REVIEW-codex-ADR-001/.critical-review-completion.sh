#!/usr/bin/env bash
# Auto-generated completion helper for critical-review of REVIEW-codex-ADR-001.
set +e
EC="$(cat "/home/enric/src/wrf_gpu2/.agent/decisions/REVIEW-codex-ADR-001/.critical-review-exit" 2>/dev/null || echo unknown)"
REP="/home/enric/src/wrf_gpu2/.agent/decisions/REVIEW-codex-ADR-001/critical-review.md"
SIZE=0; DEC=""
if [[ -f "$REP" ]]; then
  SIZE=$(stat -c %s "$REP" 2>/dev/null || echo 0)
  DEC="$(grep -m1 -E '^(Decision:|## Decision|Summary:)' "$REP" 2>/dev/null | head -1 | tr -d '\n' | cut -c1-160)"
fi
# Build the message manager receives. Keep it under 400 chars so it types fast.
MSG="AGENT REPORT [critical-review via codex / REVIEW-codex-ADR-001] exit=${EC} report=${REP##/home/enric/src/wrf_gpu2/} size=${SIZE}B ${DEC}. Per active milestone runbook: read disk for full content, then take next decision-tree step."
# Type the message into the manager window with a visible pause before Enter.
tmux send-keys -t "1:0" "$MSG"
sleep 5
tmux send-keys -t "1:0" Enter
sleep 1
# Suicide: kill this tmux window. (Runs last; whatever follows in the parent shell is irrelevant.)
tmux kill-window -t "1:REVIEW-codex-ADR-001-critical-review" 2>/dev/null || true
