#!/usr/bin/env bash
# Manager's universal sprint-role dispatcher.
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
#   - Blocks the caller until the codex process exits.
#   - Kills the tmux window on delivery (per project memory rule).
#   - Enforces a per-sprint per-role retry cap (default 5).
#   - All output is teed to logs/<sprint>-<role>-<timestamp>.log.
#
# Exit code = codex exit code. The manager loop calls this script from its turn and
# inspects the role-specific report file to decide what to do next.

set -euo pipefail

usage() { sed -n '2,22p' "$0" >&2; exit "${1:-2}"; }

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

# Resolve tmux session — manager's current session, fall back to session 1.
SESS="$(tmux display-message -p '#S' 2>/dev/null || echo 1)"
tmux has-session -t "$SESS" 2>/dev/null || { echo "no tmux session $SESS" >&2; exit 2; }

# Assemble role prompt. Each role appends a role-specific block on top of a shared header.
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
- Write \`worker-report.md\` (replacing the template) with: summary, files changed (paths), commands run + their output, proof objects (paths), risks, handoff. Must include the literal token \`Summary:\` and be ≥400 bytes.
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
- Write \`tester-report.md\` (replacing the template) with: tests added or run, results, fixtures used, gaps, Decision. Must include the literal token \`Decision:\` and be ≥400 bytes.
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
- Write \`reviewer-report.md\` (replacing the template) with: findings (severity-ranked: blocker/major/minor/note, each citing file:line), contract compliance, correctness risks, performance risks, required fixes, Decision. Must include the literal token \`Decision:\` followed by exactly one of \`Accept\` | \`Accept with required fixes\` | \`Reject\`, and be ≥400 bytes.
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
- Your report file must be ≥400 bytes and include the role-specific decision token.
- Exit cleanly when your deliverable is on disk. Do not loop.
EOF

# Launch in tmux. The window blocks on codex exit, writes done marker + exit code, then we kill it.
tmux new-window -d -t "${SESS}:" -n "$WIN" \
  "bash -lc 'set -o pipefail; echo \"[dispatch] role=$ROLE sprint=$SPRINT_NAME reasoning=$REASONING attempt=$((COUNT+1))/$RETRY_CAP at \$(date -u)\" | tee \"$LOG\"; codex exec --dangerously-bypass-approvals-and-sandbox -m gpt-5.5 -c model_reasoning_effort=\"$REASONING\" --color always --output-last-message \"$SPRINT_ABS/.${ROLE}-last.txt\" -C \"$REPO\" < \"$PROMPT\" 2>&1 | tee -a \"$LOG\"; ec=\${PIPESTATUS[0]}; echo \"\$ec\" > \"$EXIT_FILE\"; touch \"$DONE_MARK\"; echo \"[dispatch] codex exited \$ec at \$(date -u)\" | tee -a \"$LOG\"; sleep 2'"

# Block until codex finishes (no polling beyond a slow filesystem wait).
until [[ -f "$DONE_MARK" ]]; do sleep 10; done
EXIT_CODE="$(cat "$EXIT_FILE")"

# Kill the tmux window per the close-on-delivery rule.
tmux kill-window -t "${SESS}:$WIN" 2>/dev/null || true

# Verify the role's report has the required decision token + size.
VALID="true"; REASON=""
if [[ -f "$ROLE_REPORT_FILE" ]]; then
  SIZE=$(stat -c %s "$ROLE_REPORT_FILE")
  if (( SIZE < 400 )); then VALID="false"; REASON="report too short ($SIZE bytes)"; fi
  case "$ROLE" in
    worker)
      grep -q "Summary:" "$ROLE_REPORT_FILE" || { VALID="false"; REASON="worker-report.md missing 'Summary:' token"; } ;;
    tester|reviewer|critical-review)
      grep -q "Decision:" "$ROLE_REPORT_FILE" || { VALID="false"; REASON="$ROLE report missing 'Decision:' token"; } ;;
  esac
else
  VALID="false"; REASON="role report file not produced: $ROLE_REPORT_FILE"
fi

printf '{"ok":%s,"role":"%s","sprint":"%s","codex_exit":%s,"attempt":%s,"retry_cap":%s,"log":"%s","report":"%s","reason":"%s"}\n' \
  "$VALID" "$ROLE" "$SPRINT_NAME" "$EXIT_CODE" "$((COUNT+1))" "$RETRY_CAP" "$LOG" "$ROLE_REPORT_FILE" "$REASON"

[[ "$VALID" == "true" ]] && exit "$EXIT_CODE" || exit 4
