#!/usr/bin/env bash
# tmux_submit.sh [--skip-busy | --wait-idle SECONDS] <pane> <message...>
#
# Reliably submit a message to a Claude / codex TUI pane (manager<->worker, both directions).
#
# HISTORY OF THE BUGS THIS FIXES (principal had to press Enter manually):
#   v1 used "agent looks RUNNING" as proof of submit. WRONG: claude shows "esc to interrupt" as a
#      PERMANENT legend (false positive after 1 Enter while text is staged), and sending to an
#      already-busy agent detects the PRIOR task's activity, not yours -> your message sits staged.
#   v2 used "my message signature left the input". WRONG for codex: codex ECHOES the submitted
#      message into the transcript, so the signature is still on screen after a successful submit
#      -> false WARN.
#   v3 pasted into a BUSY pane before warning. A timer therefore spliced its heartbeat into a
#      principal's partially typed message. It also used 1.3 s Enter gaps, which remained flaky.
#
# V6 FAILS CLOSED: never paste unless the pane is idle AND its current input line is empty or is an
# ANSI-dim TUI suggestion (a placeholder, not editable human input). After pasting, wait 3 s, send
# Enter separately with 2 s gaps, and confirm either a live processing timer or an empty current
# input line. --skip-busy is for heartbeats: if the manager is already active or a human is typing,
# skip without touching the pane. --wait-idle is for background completion handlers that may wait
# for a clean prompt. Claude may render an otherwise-empty prompt with U+00A0 after the glyph;
# normalise that one spacing character before the exact empty-input check.
#
# Usage:
#   scripts/tmux_submit.sh 0:1 "Read /tmp/brief.txt and execute it."
#   scripts/tmux_submit.sh --skip-busy 0:1 "20-minute manager wakeup ..."
#   scripts/tmux_submit.sh --wait-idle 600 0:1 "AGENT REPORT ..."
#
# Exit 0 = submitted, or intentionally skipped by --skip-busy. Exit 1 = refused/timeout/unconfirmed.
set -uo pipefail

skip_busy=no
wait_idle=0
while [ "${1:-}" != "" ]; do
  case "$1" in
    --skip-busy)
      skip_busy=yes
      shift
      ;;
    --wait-idle)
      [ "${2:-}" != "" ] || {
        echo "tmux_submit: --wait-idle requires seconds" >&2
        exit 2
      }
      wait_idle="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "tmux_submit: unknown option: $1" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

[[ "$wait_idle" =~ ^[0-9]+$ ]] || {
  echo "tmux_submit: --wait-idle must be a non-negative integer" >&2
  exit 2
}

pane="${1:-}"; shift || true; msg="${*:-}"
[ -n "$pane" ] && [ -n "$msg" ] || {
  echo "usage: tmux_submit.sh [--skip-busy | --wait-idle SECONDS] <pane> <message...>" >&2
  exit 2
}

# A live "(<timer>)" in the bottom region == the agent is actively processing right now.
is_working() {
  tmux capture-pane -t "$pane" -p 2>/dev/null | tail -n 8 \
    | grep -qE '\([0-9]+m?[ ]?[0-9]*s[ )•·]'
}

# Some TUIs leave many blank terminal rows below the status bar. Trim only those trailing blank
# rows before taking the prompt-local tail; widening the raw tail would risk accepting an old
# transcript prompt during a transition.
trim_trailing_blank_rows() {
  awk '
    { row[NR] = $0; if ($0 !~ /^[[:space:]]*$/) last = NR }
    END {
      if (!last) exit
      for (i = 1; i <= last; i++) print row[i]
    }
  '
}

# Accept only a visibly empty current TUI prompt. Known placeholder text is not user input.
input_is_empty() {
  local prompt_line prompt_line_ansi after_prompt before_dim before_dim_plain
  prompt_line="$(
    tmux capture-pane -t "$pane" -p 2>/dev/null \
      | trim_trailing_blank_rows \
      | tail -n 12 \
      | grep -E '^[[:space:]]*([›❯>]|│[[:space:]]*[›❯>])' \
      | tail -n 1
  )"
  # Claude sometimes renders an empty input cell as "❯<NBSP>". Removing only
  # U+00A0 is safe: any real non-spacing input remains and still fails closed.
  prompt_line="${prompt_line//$'\302\240'/}"
  [[ "$prompt_line" =~ ^[[:space:]]*❯[[:space:]]*$ ]] \
    || [[ "$prompt_line" =~ ^[[:space:]]*›[[:space:]]*$ ]] \
    || [[ "$prompt_line" =~ ^[[:space:]]*\>[[:space:]]*$ ]] \
    || [[ "$prompt_line" =~ ^[[:space:]]*│[[:space:]]*\>[[:space:]]*│[[:space:]]*$ ]] \
    || [[ "$prompt_line" =~ ^[[:space:]]*›[[:space:]]*Explain[[:space:]]+this[[:space:]]+codebase[[:space:]]*$ ]] \
    || {
      # Claude/Codex can display arbitrary suggested next prompts. Plain capture-pane makes those
      # look like typed text; ANSI-preserving capture distinguishes them mechanically because the
      # suggestion after the prompt glyph is wrapped in SGR dim (ESC[2m). Actual user input is
      # rendered normally and therefore still fails closed.
      prompt_line_ansi="$(
        tmux capture-pane -t "$pane" -p -e 2>/dev/null \
          | trim_trailing_blank_rows \
          | tail -n 12 \
          | grep -E '[›❯>]' \
          | tail -n 1
      )"
      after_prompt="$prompt_line_ansi"
      case "$prompt_line_ansi" in
        *❯*) after_prompt="${prompt_line_ansi#*❯}" ;;
        *›*) after_prompt="${prompt_line_ansi#*›}" ;;
        *">"*) after_prompt="${prompt_line_ansi#*>}" ;;
      esac
      [[ "$after_prompt" != "$prompt_line_ansi" && "$after_prompt" == *$'\033[2m'* ]] || return 1
      before_dim="${after_prompt%%$'\033[2m'*}"
      before_dim_plain="$(
        printf '%s' "$before_dim" \
          | sed $'s/\033\\[[0-9;?]*[ -\\/]*[@-~]//g'
      )"
      before_dim_plain="${before_dim_plain//$'\302\240'/}"
      [[ "$before_dim_plain" =~ ^[[:space:]]*$ ]]
    }
}

is_ready() {
  ! is_working && input_is_empty
}

if ! is_ready; then
  if [ "$skip_busy" = yes ]; then
    echo "tmux_submit: SKIPPED $pane (busy, transitional, or non-empty input; pane untouched)"
    exit 0
  fi

  if [ "$wait_idle" -eq 0 ]; then
    echo "tmux_submit: REFUSED $pane (busy, transitional, or non-empty input; nothing pasted)" >&2
    exit 1
  fi

  deadline=$((SECONDS + wait_idle))
  until is_ready; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "tmux_submit: TIMEOUT waiting ${wait_idle}s for idle + empty input on $pane" >&2
      exit 1
    fi
    sleep 2
  done
fi

# 1) paste the text ONLY (no Enter), let the TUI ingest the paste.
tmux send-keys -t "$pane" -- "$msg"
sleep 3

# 2) send Enter separately and slowly until processing starts or the live input clears.
for i in $(seq 1 12); do
  tmux send-keys -t "$pane" Enter
  sleep 2
  if is_working; then
    echo "tmux_submit: SUBMITTED to $pane (agent processing) after $i Enter(s)"
    exit 0
  fi
  if input_is_empty; then
    echo "tmux_submit: SUBMITTED to $pane (input cleared) after $i Enter(s)"
    exit 0
  fi
done

echo "tmux_submit: WARN — input did not clear and processing did not start after 12 delayed Enters" >&2
tmux capture-pane -t "$pane" -p 2>/dev/null | grep -vE '^\s*$' | tail -3 >&2
exit 1
