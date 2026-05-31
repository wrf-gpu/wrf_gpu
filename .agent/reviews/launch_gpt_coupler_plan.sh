#!/usr/bin/env bash
set -uo pipefail
REPO=/home/enric/src/wrf_gpu2
cd "$REPO"
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
taskset -c 0-3 codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  -C "$REPO" \
  -m gpt-5.5 \
  -c model_reasoning_effort=xhigh \
  "Read .agent/reviews/2026-05-30-gpt-coupler-plan-task.md and perform EXACTLY that independent adversarial review: PART A physics-coupler bug/inefficiency review + PART B remaining-technical-hurdles plan review (skip validation/publishing). Cite file:line. Write findings to the exact output path named. End with GPT_COUPLER_PLAN_REVIEW_COMPLETE + the top-3 must-fix technical items ranked."
ec=$?
tmux send-keys -t 0:0 "AGENT REPORT: gpt-coupler-plan-review exit=$ec" Enter
sleep 1
tmux kill-window -t gpt-coupler 2>/dev/null
