# Dispatching Agents — Canonical tmux Pattern (with mandatory completion handler)

**Hard rule** (added 2026-05-21 ~01:15 after a flow-break incident): every tmux-launched agent dispatch MUST include a completion handler that tap-types an `AGENT REPORT` message into the manager pane on exit. Without this, the manager doesn't get auto-notified and watchman cycles waste turns rediscovering finished sprints.

## Why this matters

When manager dispatches `tmux new-window` with just `codex …` or `claude …` and no completion handler:
- Agent eventually `/exit`s the AI tool
- The shell process terminates
- tmux window may stay alive at a dead shell, or close silently
- Manager has no signal until next watchman tick + manual peek

With the completion handler:
- Agent `/exit`s
- Wrapper bash captures exit code
- `tmux send-keys -t "$MGR_TARGET" "$MSG"` types the report into manager's Claude input
- Manager sees `AGENT REPORT [role / sprint / AI] exit=N report=PATH` as if user typed it → automatic awareness
- Window self-destructs cleanly

## Canonical dispatch (use this every time, no exceptions for code/governance sprints)

```bash
WT=/path/to/worktree
SP="$WT/.agent/sprints/<sprint-folder>"
PROMPT="$SP/role-prompts/<role>.md"
LOG="$SP/<role>.log"
EXIT_FILE="$SP/.<role>-exit"
DONE_MARK="$SP/.<role>-done"
WIN=<role>-<sprint-short-tag>
MGR_SESS=$(tmux display-message -p '#S')
MGR_WIN=$(tmux display-message -p '#I')
MGR_TARGET="${MGR_SESS}:${MGR_WIN}"

# Pre-create empty done marker so manager can check existence as proxy
rm -f "$EXIT_FILE" "$DONE_MARK"

tmux new-window -d -t "${MGR_SESS}:" -n "$WIN" \
  "bash -lc 'echo \"[dispatch-${WIN}] started at \$(date -u)\"; cd $WT; \
   <YOUR_AI_INVOCATION_HERE>; \
   ec=\$?; echo \"\$ec\" > \"$EXIT_FILE\"; touch \"$DONE_MARK\"; \
   MSG=\"AGENT REPORT [<role> / <sprint> / <AI>] exit=\$ec report=$SP/<role>-report.md\"; \
   tmux send-keys -t \"$MGR_TARGET\" \"\$MSG\"; \
   sleep 5; tmux send-keys -t \"$MGR_TARGET\" Enter; \
   sleep 1; tmux kill-window -t \"${MGR_SESS}:${WIN}\" 2>/dev/null || true'"

sleep 8     # Codex/Claude banner + initial prompt rendering takes 4-7s — wait longer to avoid pasting into raw shell
tmux pipe-pane -t "${MGR_SESS}:${WIN}" -O "cat >> \"$LOG\"" 2>/dev/null || true
tmux load-buffer -b "${WIN}_seed" "$PROMPT"
sleep 2
tmux paste-buffer -d -p -b "${WIN}_seed" -t "${MGR_SESS}:${WIN}"
sleep 3     # Paste-buffer is async; give it time to land before Enter
tmux send-keys -t "${MGR_SESS}:${WIN}" Enter

# VERIFY: after dispatch, capture the pane and confirm the AI got the prompt (look for "Working" or active state)
# tmux capture-pane -t "${MGR_SESS}:${WIN}" -p -S -10 | tail -5
# If pane shows raw shell or codex/claude welcome banner WITHOUT "Working" or prompt acknowledgment,
# the dispatch timing failed: re-paste with: tmux paste-buffer -d -p -b "${WIN}_seed" -t "${MGR_SESS}:${WIN}"; tmux send-keys ... Enter
# Observed 2026-05-21 ~03:18: dispatch landed BEFORE codex banner rendered → prompt appeared in scrollback, codex idle. Lost ~60min. Hence sleep 8 not 4.
```

Substitute `<YOUR_AI_INVOCATION_HERE>` with:
- Codex: `codex --dangerously-bypass-approvals-and-sandbox -m gpt-5.5 -c model_reasoning_effort=xhigh`
- Claude Opus: `claude --dangerously-skip-permissions --model claude-opus-4-7`
- Claude Sonnet: `claude --dangerously-skip-permissions --model claude-sonnet-4-6`
- Gemini (reactive only per current policy): use Pattern D inline `agy -p` from manager pane; tmux dispatch problematic due to OAuth-per-pty quirk

## Per the sprint-lifecycle hard rule

Every code/governance sprint requires Opus 4.7 reviewer pass before close. Dispatch the reviewer with the canonical pattern above + the reviewer role prompt. Without the completion handler the manager won't know when the reviewer is done.

## HARD RULE: Watchdog auto-notify (added 2026-05-21 ~11:40 after 4 stuck-at-/exit incidents)

Both codex and claude CLIs have the SAME failure mode: when the role prompt says "When done, /exit", the AI writes "/exit" as PLAIN TEXT inside its report and never executes it as a slash command. The pane sits at the input prompt with `/exit` visible but un-fired forever. Manager has to manually tap Enter, wasting an entire watchman cycle per stuck agent.

**Fix**: every launcher MUST include a backgrounded watchdog that:
1. Polls for the report file (`worker-report.md` or `reviewer-report.md`) to appear
2. Waits for it to be "stable" (no modification in last 60s)
3. Taps `Enter` into the pane to flush any queued `/exit`
4. Sends `/exit Enter` as a fresh slash-command
5. Force-kills the window if it's still alive
6. Fires `AGENT REPORT` to the manager pane UNCONDITIONALLY (if not already done by the foreground)

Skeleton (drop into every launcher script):

```bash
# --- BACKGROUND WATCHDOG ---
( while [ ! -f "$REPORT" ]; do sleep 10; done
  while true; do
    age=$(( $(date +%s) - $(stat -c %Y "$REPORT" 2>/dev/null || echo 0) ))
    [ "$age" -ge 60 ] && break
    sleep 15
  done
  tmux send-keys -t "1:$WIN" Enter 2>/dev/null || true; sleep 5
  tmux send-keys -t "1:$WIN" "/exit" Enter 2>/dev/null || true; sleep 5
  tmux send-keys -t "1:$WIN" Enter 2>/dev/null || true; sleep 3
  tmux kill-window -t "1:$WIN" 2>/dev/null || true
  if [ ! -f "$DONE_MARK" ]; then
    echo "watchdog" > "$EXIT_FILE"; touch "$DONE_MARK"
    MSG="AGENT REPORT [$ROLE_TAG / $SPRINT_TAG / $AI_NAME] exit=watchdog report=$REPORT"
    tmux send-keys -t "1:0" "$MSG" 2>/dev/null
    sleep 3
    tmux send-keys -t "1:0" Enter 2>/dev/null  # first Enter — establishes input complete
    sleep 2
    tmux send-keys -t "1:0" Enter 2>/dev/null  # second Enter — actually submits (Claude Code needs this)
    sleep 2
    tmux send-keys -t "1:0" Enter 2>/dev/null  # safety third Enter
  fi
) &
WATCHDOG_PID=$!

# --- FOREGROUND AI ---
cd "$WT"
<AI_CMD>
ec=$?
if [ ! -f "$DONE_MARK" ]; then
  echo "$ec" > "$EXIT_FILE"; touch "$DONE_MARK"
  MSG="AGENT REPORT [$ROLE_TAG / $SPRINT_TAG / $AI_NAME] exit=$ec report=$REPORT"
  tmux send-keys -t "1:0" "$MSG" 2>/dev/null
  sleep 3
  tmux send-keys -t "1:0" Enter 2>/dev/null
  sleep 2
  tmux send-keys -t "1:0" Enter 2>/dev/null
  sleep 2
  tmux send-keys -t "1:0" Enter 2>/dev/null
fi
kill "$WATCHDOG_PID" 2>/dev/null || true
sleep 1
tmux kill-window -t "1:$WIN" 2>/dev/null || true
```

The watchdog and foreground race; whoever notices first fires AGENT REPORT (the `$DONE_MARK` sentinel prevents double-fire). Net effect: manager gets notified within ~75s of report-file stability regardless of whether the AI exited cleanly.

**Role prompt clarification**: also update every role prompt to say "type `/exit` as a slash-command (not as text in the report); the wrapper has a watchdog that will force-fire AGENT REPORT after 60s of report-file stability if you forget."

**Multi-Enter pattern** (added 2026-05-21 12:05 after user-flagged "had to press enter after the last report" incident): Claude Code's input prompt requires **three Enter keystrokes with 2-3s delays** to reliably submit a pasted AGENT REPORT message. Single Enter sometimes silently fails (input shows the text but doesn't submit). Always pattern:

```bash
tmux send-keys -t "1:0" "$MSG"   # paste text
sleep 3                          # let the buffer settle
tmux send-keys -t "1:0" Enter    # establish input complete
sleep 2
tmux send-keys -t "1:0" Enter    # actual submit (often the load-bearing one)
sleep 2
tmux send-keys -t "1:0" Enter    # safety
```

Sending the Enter immediately after `send-keys "$MSG"` or with only 1s delay loses the input. The 3-2-2s spacing is empirically required.

Reference launchers using this pattern: `/tmp/launch_rev_s3y.sh`, `/tmp/launch_rev_m6s1.sh` (2026-05-21).

## When the bash-script-heuristic blocks the dispatch (legacy)

Earlier in this project the sandbox refused long bash-string dispatch citing "agent-inferred parameters for a high-stakes autonomous worker spawn." Mitigations:

1. Write the role prompt + a short launcher script to disk via the Write tool (verifiable by the sandbox)
2. Issue the tmux dispatch with the launcher path, not inlined
3. User one-time permission approval if needed

## Agent-pool balance (added 2026-05-21 14:40 per user directive)

**Rule**: ≤3 codex (gpt-5.5) parallel workers at any time — rate-limit risk. When a 4th sprint slot opens, dispatch **Opus** for any of these roles:

- Reviewer (already standard)
- Bug fixer (specific named bug)
- Bug hunter (mysterious residual diagnosis)
- Tool builder (sidecar script, harness extension, helper utility)
- Test runner (pytest, gates, profile audits)
- Skill/memory/ADR patch authoring
- Manager intermediate consensus / second opinions

Opus has separate quota pool, is faster (~2-4×) for short focused tasks, and has equal or better quality on reviewer-style work. Mixing the workforce keeps throughput high without either pool exhausting.

Opus reviewer dispatches do NOT count against the codex 3-cap.

Heavy code-write sprints (Thompson kernel rewrite, RRTMG transcription, M6-S2 driver): still codex, counts against the cap.

Investigative / diagnostic / validation / governance / tooling: prefer opus.

## Cross-links

- `.agent/rules/sprint-lifecycle.md` — double-AI principle
- `.agent/references/dispatching-gemini.md` — Gemini-specific quirks
- `.agent/SPRINT-TRACKER.md` — live dashboard updated each watchman tick
