#!/usr/bin/env bash
set -uo pipefail
REPO=/home/enric/src/wrf_gpu2
cd "$REPO"
taskset -c 0-3 agy \
  --dangerously-skip-permissions \
  --add-dir "$REPO" \
  --add-dir /home/enric/src/wrf_pristine \
  --print-timeout 40m \
  --print "Read .agent/reviews/2026-05-30-agy-v10-task.md and perform EXACTLY that READ-ONLY surface-physics analysis comparing WRF sf_sfclayrev vs sf_mynn momentum/V10 paths. Cite file:line. Write findings to the exact output path named in the task. End with AGY_V10_ANALYSIS_COMPLETE + your single top recommendation."
ec=$?
tmux send-keys -t 0:0 "AGENT REPORT: agy-v10-analysis exit=$ec" Enter
sleep 1
tmux kill-window -t agy-v10 2>/dev/null
