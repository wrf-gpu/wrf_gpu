#!/usr/bin/env bash
# tmux_submit.sh <pane> <message...>
#
# Reliably submit a message to a Claude / codex TUI pane (manager<->worker, both directions).
#
# HISTORY OF THE TWO BUGS THIS FIXES (principal had to press Enter manually, 2026-06-21):
#   v1 used "agent looks RUNNING" as proof of submit. WRONG: claude shows "esc to interrupt" as a
#      PERMANENT legend (false positive after 1 Enter while text is staged), and sending to an
#      already-busy agent detects the PRIOR task's activity, not yours -> your message sits staged.
#   v2 used "my message signature left the input". WRONG for codex: codex ECHOES the submitted
#      message into the transcript, so the signature is still on screen after a successful submit
#      -> false WARN.
#
# THE RELIABLE SIGNAL (v3): a PARENTHESIZED LIVE TIMER — "(15s • esc to interrupt)" (codex) or
# "(26m 12s · thinking…)" (claude) — appears ONLY while the agent is actively processing. It is
# absent when idle (the permanent legend has no timer) and absent on a finished "Worked for 1m 45s"
# line (no paren). So: if the agent was IDLE before, success = it transitions to a live timer after
# our Enter (i.e. it started processing OUR message). If it was already BUSY, we cannot credit the
# timer to our message -> we report it as QUEUED (it will run when the agent next idles; verify).
#
# OPERATING RULE (managing-sprints skill): prefer sending only to an IDLE agent. This helper
# enforces it softly: a send to a busy agent returns rc 1 with a QUEUED warning, not a false ok.
#
# Usage:  scripts/tmux_submit.sh 0:1 "Read /tmp/brief.txt and execute it. Report to pane 0:1."
# Exit 0 = agent started processing our message. Exit 1 = queued/unconfirmed (verify).
set -uo pipefail

pane="${1:-}"; shift || true; msg="${*:-}"
[ -n "$pane" ] && [ -n "$msg" ] || { echo "usage: tmux_submit.sh <pane> <message...>" >&2; exit 2; }

# A live "(<timer>)" in the bottom region == the agent is actively processing right now.
is_working() {
  tmux capture-pane -t "$pane" -p 2>/dev/null | tail -n 8 \
    | grep -qE '\([0-9]+m?[ ]?[0-9]*s[ )•·]'
}

busy_before=no; is_working && busy_before=yes

# 1) paste the text ONLY (no Enter), let the TUI ingest the paste.
tmux send-keys -t "$pane" -- "$msg"
sleep 2

# 2) send Enter (delayed, repeated) until the agent is processing OUR message.
for i in $(seq 1 12); do
  tmux send-keys -t "$pane" Enter
  sleep 1.3
  if is_working; then
    if [ "$busy_before" = no ]; then
      echo "tmux_submit: SUBMITTED to $pane (agent processing) after $i Enter(s)"
      exit 0
    fi
    # busy_before=yes: the timer may be the PRIOR task; cannot credit it to our message.
  fi
done

if [ "$busy_before" = yes ]; then
  echo "tmux_submit: WARN — $pane was BUSY when sent; message is likely QUEUED and will run when" >&2
  echo "tmux_submit:        the agent next idles. Re-verify (or re-send) once it is idle." >&2
else
  echo "tmux_submit: WARN — could not confirm $pane started processing after 12 Enters; verify." >&2
fi
tmux capture-pane -t "$pane" -p 2>/dev/null | grep -vE '^\s*$' | tail -3 >&2
exit 1
