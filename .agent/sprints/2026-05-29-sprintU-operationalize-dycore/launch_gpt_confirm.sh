#!/usr/bin/env bash
set -uo pipefail
REPO=/home/enric/src/wrf_gpu2
cd "$REPO"
# Pin to cores 0-3 (CPU core budget: WRF reserves 4-31). codex itself is light;
# any python test it runs stays on our cores.
export OMP_NUM_THREADS=4
taskset -c 0-3 codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  -C "$REPO" \
  -m gpt-5.5 \
  -c model_reasoning_effort=xhigh \
  "Read .agent/sprints/2026-05-29-sprintU-operationalize-dycore/gpt-confirm-close-prompt.md and perform EXACTLY that CONFIRM-CLOSE review of the Sprint U dycore remediation. Be adversarial, cite file:line, run any verification you need (tests are runnable; pin python to cores 0-3 with taskset). Write findings to the exact output path named in the prompt. End with SPRINTU_CONFIRM_COMPLETE and CLOSE-CONFIRMED or CLOSE-REJECTED-pending-<items>."
ec=$?
tmux send-keys -t 0:0 "AGENT REPORT: gpt-confirm-close exit=$ec" Enter
sleep 1
tmux kill-window -t gpt-confirm 2>/dev/null
