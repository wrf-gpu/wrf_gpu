#!/usr/bin/env bash
set -uo pipefail
REPO=/home/enric/src/wrf_gpu2
cd "$REPO"
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
taskset -c 0-3 codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  -C "$REPO" \
  -m gpt-5.5 \
  -c model_reasoning_effort=xhigh \
  "Read .agent/sprints/2026-05-29-sprintU-operationalize-dycore/gpt-final-confirm-prompt.md and perform EXACTLY that final decisive confirmation. Cite file:line. Write findings to the exact output path named. End with SPRINTU_FINAL_COMPLETE and CLOSE-CONFIRMED or CLOSE-REJECTED-pending-<specific item>. Do not reject without a concrete fixable item."
ec=$?
tmux send-keys -t 0:0 "AGENT REPORT: gpt-final-confirm exit=$ec" Enter
sleep 1
tmux kill-window -t gpt-final 2>/dev/null
