#!/usr/bin/env bash
# Manager's universal sprint-role dispatcher (fire-and-forget + send-keys completion).
#
# Usage:
#   scripts/dispatch_role.sh <role> <sprint-folder> [--reasoning <high|xhigh>] [--retry-cap N]
#
# Roles:
#   worker            — implements the sprint per its contract, writes worker-report.md
#   tester            — independently validates, writes tester-report.md
#   reviewer          — critiques, writes reviewer-report.md with "Decision: ..." token
#   critical-review   — manager's second-opinion path; <sprint-folder> may be any decision-proposal folder
#
# Behavior:
#   - Opens a tmux window in the manager's tmux session, named "<sprint>-<role>".
#   - Runs codex exec non-interactively (Gen2 protocol: --dangerously-bypass-approvals-and-sandbox,
#     gpt-5.5, configurable reasoning effort).
#   - **Returns immediately to the caller** (does not block). Manager yields after dispatch.
#   - When the agent finishes, the agent's tmux window:
#       (a) writes a done marker + exit code to disk
#       (b) tmux send-keys a short summary into the manager's window (5s pause then Enter)
#       (c) kills its own window
#   - The manager receives the summary as a "user message" on its next turn and continues the loop.
#   - All output is teed to logs/<sprint>-<role>-<timestamp>.log.
#   - Enforces a per-sprint per-role retry cap (default 5).
#
# Only one agent should run at a time per sprint (worker → tester → reviewer is sequential by lifecycle).
# Multiple sprints in parallel are NOT recommended in M1 because send-keys can interleave into the
# manager's prompt area. The manager's runbook keeps M1 single-threaded.

set -euo pipefail

usage() { sed -n '2,28p' "$0" >&2; exit "${1:-2}"; }

ROLE="${1:-}"; SPRINT="${2:-}"; shift 2 || usage 2
REASONING="high"; RETRY_CAP=5
while (( $# > 0 )); do
  case "$1" in
    --reasoning) REASONING="$2"; shift 2 ;;
    --retry-cap) RETRY_CAP="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; usage 2 ;;
  esac
done

case "$ROLE" in worker|tester|reviewer|critical-review) ;; *) echo "bad role: $ROLE" >&2; usage 2 ;; esac
[[ -d "$SPRINT" ]] || { echo "sprint folder not found: $SPRINT" >&2; exit 2; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SPRINT_ABS="$(cd "$SPRINT" && pwd)"
SPRINT_NAME="$(basename "$SPRINT_ABS")"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
WIN="${SPRINT_NAME:0:40}-${ROLE}"
LOG_DIR="$REPO/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SPRINT_NAME}-${ROLE}-${TS}.log"
DONE_MARK="$SPRINT_ABS/.${ROLE}-done"
EXIT_FILE="$SPRINT_ABS/.${ROLE}-exit"
RETRY_FILE="$SPRINT_ABS/.${ROLE}-retry-count"
PROMPT_DIR="$SPRINT_ABS/role-prompts"; mkdir -p "$PROMPT_DIR"
PROMPT="$PROMPT_DIR/${ROLE}.md"

# Retry cap.
COUNT=0; [[ -f "$RETRY_FILE" ]] && COUNT=$(cat "$RETRY_FILE")
if (( COUNT >= RETRY_CAP )); then
  echo "{\"ok\":false,\"error\":\"retry cap reached\",\"role\":\"$ROLE\",\"sprint\":\"$SPRINT_NAME\",\"count\":$COUNT}"
  exit 3
fi
echo $((COUNT+1)) > "$RETRY_FILE"

# Resolve manager session + window for send-keys completion.
MGR_SESS="$(tmux display-message -p '#S' 2>/dev/null || echo 1)"
MGR_WIN="$(tmux display-message -p '#I' 2>/dev/null || echo 0)"
MGR_TARGET="${MGR_SESS}:${MGR_WIN}"
tmux has-session -t "$MGR_SESS" 2>/dev/null || { echo "no tmux session $MGR_SESS" >&2; exit 2; }

# Role-specific report path + branch + instructions.
ROLE_REPORT_FILE="$SPRINT_ABS/${ROLE//-/_}-report.md"
case "$ROLE" in
  worker)
    ROLE_REPORT_FILE="$SPRINT_ABS/worker-report.md"
    BRANCH="worker/gpt/${SPRINT_NAME#????-??-??-}"
    ROLE_INSTRUCTIONS="$(cat <<EOF
You are the sprint **worker** (codex gpt-5.5). Implement exactly what \`sprint-contract.md\` specifies, no more, no less.

Required output:
- Write code only to paths listed under "File Ownership" in the contract.
- Run every validation command listed in the contract; capture stdout/stderr.
- Write \`worker-report.md\` (replacing the template) with: summary, files changed (paths), commands run + their output, proof objects (paths), risks, handoff. Must include the literal token \`Summary:\` and be >=400 bytes.
- Push your work on a feature branch named \`$BRANCH\`. If the branch exists, append to it.
- Do not touch reviewer-report.md, tester-report.md, manager-closeout.md, or memory-patch.md — those are other roles.
- Do not modify governance files (PROJECT_CONSTITUTION.md, PROJECT_SCOPE.md, AGENTS.md, CLAUDE.md, ARCHITECTURE_PRINCIPLES.md, VALIDATION_STRATEGY.md, PRECISION_POLICY.md, PERFORMANCE_TARGETS.md, RISK_REGISTER.md, INTERFACE_CONTRACTS.md, PROJECT_PLAN.md, MILESTONES.md, the \`.agent/rules/*\`, \`.agent/roles/*\`, or any goal file).
EOF
)"
    ;;
  tester)
    ROLE_REPORT_FILE="$SPRINT_ABS/tester-report.md"
    BRANCH="tester/sonnet/${SPRINT_NAME#????-??-??-}"
    ROLE_INSTRUCTIONS="$(cat <<EOF
You are the sprint **tester** (codex gpt-5.5 acting as sonnet-test-engineer). The worker has already implemented the sprint and written \`worker-report.md\`. Your job:

- Re-run every validation command in the contract from a clean shell; record results.
- Add edge-case tests under \`tests/\` that the worker may have missed.
- Try to break the implementation: malformed inputs, boundary cases, missing files, schema violations.
- Write \`tester-report.md\` (replacing the template) with: tests added or run, results, fixtures used, gaps, Decision. Must include the literal token \`Decision:\` and be >=400 bytes.
- You may edit only \`tests/\`, the report file, and your own branch \`$BRANCH\`.
- Do not edit code under \`src/\`, \`scripts/\` (other than test helpers under \`tests/\`), or any governance file.
EOF
)"
    ;;
  reviewer)
    ROLE_REPORT_FILE="$SPRINT_ABS/reviewer-report.md"
    BRANCH="reviewer/opus/${SPRINT_NAME#????-??-??-}"
    ROLE_INSTRUCTIONS="$(cat <<EOF
You are the sprint **reviewer** (codex gpt-5.5 acting as opus-reviewer). The worker has implemented; the tester has tested. You read everything and pass independent judgment.

- Read worker-report.md, tester-report.md, the full sprint diff, the contract, the constitution, AGENTS.md, the relevant goal file, and PROJECT_PLAN.md.
- Run independent spot-checks of validation commands.
- Write \`reviewer-report.md\` (replacing the template) with: findings (severity-ranked: blocker/major/minor/note, each citing file:line), contract compliance, correctness risks, performance risks, required fixes, Decision. Must include the literal token \`Decision:\` followed by exactly one of \`Accept\` | \`Accept with required fixes\` | \`Reject\`, and be >=400 bytes.
- Read-only access to source files; you may only write \`reviewer-report.md\` and your own branch \`$BRANCH\`.
- No rewriting the worker's code during review.
EOF
)"
    ;;
  critical-review)
    ROLE_REPORT_FILE="$SPRINT_ABS/critical-review.md"
    BRANCH=""
    ROLE_INSTRUCTIONS="$(cat <<EOF
You are an **independent senior reviewer** asked by the manager for a second opinion on a decision. Read the decision proposal under this folder (file \`proposal.md\`), then the cited governance files and any cited evidence, then write \`critical-review.md\` with:

- Decision (Accept | Accept with required fixes | Reject)
- Top three structural concerns
- Findings (numbered, severity-ranked, file:line cited)
- Dissent
- Closing recommendation

You may only write \`critical-review.md\`. Read-only everywhere else. Do not commit anything.
EOF
)"
    ;;
esac

cat > "$PROMPT" <<EOF
# Role: $ROLE   Sprint: $SPRINT_NAME   Launched: $TS

## Read order (mandatory, in order)

1. \`PROJECT_CONSTITUTION.md\`
2. \`AGENTS.md\`
3. \`CLAUDE.md\`
4. \`PROJECT_PLAN.md\`
5. \`.agent/milestones/ROADMAP.md\`
6. \`.agent/goals/M1-DONE.md\` (the active goal; do not change it)
7. \`$SPRINT_ABS/sprint-contract.md\`
8. The relevant skill under \`.agent/skills/\` for your role:
   - worker → writing-gpu-kernels, writing-execplans
   - tester → validating-physics
   - reviewer → conducting-blind-review
   - critical-review → resolving-cross-model-disagreements

## Role-specific instructions

$ROLE_INSTRUCTIONS

## Universal hard rules

- Do not edit any file outside the role's allowed scope.
- Do not modify governance files or goal files.
- Do not commit binary fixture data to git. Use \`data/\` (symlink to \`/mnt/data/wrf_gpu2/\`).
- All work happens on the role's branch ($BRANCH if applicable). The manager integrates branches.
- Your report file must be >=400 bytes and include the role-specific decision token.
- Exit cleanly when your deliverable is on disk. Do not loop.
EOF

# Build the post-completion summary helper as a separate script the tmux window will source.
# The helper builds a one-paragraph status, send-keys it to the manager window, then kills its own window.
COMPLETION_HELPER="$SPRINT_ABS/.${ROLE}-completion.sh"
cat > "$COMPLETION_HELPER" <<COMPLETION_EOF
#!/usr/bin/env bash
# Auto-generated completion helper for ${ROLE} of ${SPRINT_NAME}.
set +e
EC="\$(cat "$EXIT_FILE" 2>/dev/null || echo unknown)"
REP="$ROLE_REPORT_FILE"
SIZE=0; DEC=""
if [[ -f "\$REP" ]]; then
  SIZE=\$(stat -c %s "\$REP" 2>/dev/null || echo 0)
  DEC="\$(grep -m1 -E '^(Decision:|## Decision|Summary:)' "\$REP" 2>/dev/null | head -1 | tr -d '\n' | cut -c1-160)"
fi
# Build the message manager receives. Keep it under 400 chars so it types fast.
MSG="AGENT REPORT [$ROLE via $AI_CLI / $SPRINT_NAME] exit=\${EC} report=\${REP##$REPO/} size=\${SIZE}B \${DEC}. Per active milestone runbook: read disk for full content, then take next decision-tree step."
# Type the message into the manager window with a visible pause before Enter.
tmux send-keys -t "$MGR_TARGET" "\$MSG"
sleep 5
tmux send-keys -t "$MGR_TARGET" Enter
sleep 1
# Suicide: kill this tmux window. (Runs last; whatever follows in the parent shell is irrelevant.)
tmux kill-window -t "$MGR_SESS:$WIN" 2>/dev/null || true
COMPLETION_EOF
chmod +x "$COMPLETION_HELPER"

# Cross-model AI assignment per role (codified 2026-05-19 per user directive):
#   worker          → codex gpt-5.5      — implementation (codex is good at code)
#   tester          → claude opus 4.7    — independent second-AI verification (different blind spots)
#   reviewer        → codex gpt-5.5      — binding judgment (worker AI + reviewer AI same = OK because tester is the other AI)
#   critical-review → codex gpt-5.5      — manager's second-opinion path
# Override --reasoning maps differently per CLI:
#   codex: model_reasoning_effort=<high|xhigh>
#   claude: --effort <high|xhigh>
case "$ROLE" in
  tester)
    AI_CLI="claude"
    # Claude Code: -p (print/non-interactive), --model opus (alias = latest opus = 4.7),
    # --effort <level>, --permission-mode bypassPermissions, prompt via stdin.
    LAUNCH_CMD="claude -p --model opus --effort \"$REASONING\" --permission-mode bypassPermissions --no-session-persistence --append-system-prompt \"You are acting as the sonnet-test-engineer ROLE for this project, running as Claude Opus 4.7. Strictly follow the role-specific instructions in the prompt. Do not loop interactively. Exit cleanly when your deliverable file is on disk.\" --add-dir \"$REPO\" --add-dir /mnt/data/wrf_gpu2 < \"$PROMPT\""
    ;;
  worker|reviewer|critical-review)
    AI_CLI="codex"
    LAUNCH_CMD="codex exec --dangerously-bypass-approvals-and-sandbox -m gpt-5.5 -c model_reasoning_effort=\"$REASONING\" --color always --output-last-message \"$SPRINT_ABS/.${ROLE}-last.txt\" -C \"$REPO\" < \"$PROMPT\""
    ;;
esac

# Launch in tmux: AI runs non-interactively, on exit the completion helper send-keys back to the manager.
# `tmux new-window -d` returns immediately; the manager does not block.
tmux new-window -d -t "${MGR_SESS}:" -n "$WIN" \
  "bash -lc 'set -o pipefail; echo \"[dispatch] role=$ROLE ai=$AI_CLI sprint=$SPRINT_NAME reasoning=$REASONING attempt=$((COUNT+1))/$RETRY_CAP at \$(date -u)\" | tee \"$LOG\"; $LAUNCH_CMD 2>&1 | tee -a \"$LOG\"; ec=\${PIPESTATUS[0]}; echo \"\$ec\" > \"$EXIT_FILE\"; touch \"$DONE_MARK\"; echo \"[dispatch] $AI_CLI exited \$ec at \$(date -u)\" | tee -a \"$LOG\"; bash \"$COMPLETION_HELPER\"'"

# Manager's view: confirmation that the agent was launched (this is the JSON the manager reads).
printf '{"ok":true,"role":"%s","ai":"%s","sprint":"%s","tmux_target":"%s:%s","attempt":%s,"retry_cap":%s,"log":"%s","report":"%s","note":"fire-and-forget; agent will send-keys report on completion"}\n' \
  "$ROLE" "$AI_CLI" "$SPRINT_NAME" "$MGR_SESS" "$WIN" "$((COUNT+1))" "$RETRY_CAP" "$LOG" "$ROLE_REPORT_FILE"
exit 0
