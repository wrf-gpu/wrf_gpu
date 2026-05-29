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
  "Read .agent/sprints/2026-05-29-sprintU-operationalize-dycore/gpt-reconfirm-prompt.md and perform EXACTLY that focused re-confirm of the fp64 dycore-close remediation. Be adversarial, cite file:line, run any verification you need (pin python to cores 0-3 with taskset, cap XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 to avoid GPU contention with concurrent WRF jobs). Write findings to the exact output path named in the prompt. End with SPRINTU_RECONFIRM_COMPLETE and CLOSE-CONFIRMED or CLOSE-REJECTED-pending-<items>."
ec=$?
tmux send-keys -t 0:0 "AGENT REPORT: gpt-reconfirm exit=$ec" Enter
sleep 1
tmux kill-window -t gpt-reconfirm 2>/dev/null
