# Dispatching Gemini 3.5 (agy CLI) — Reference

Third AI available to this project alongside Claude (Opus 4.7 / Sonnet 4.6) and Codex (gpt-5.5). Per user authorization 2026-05-20.

## Capability + constraints

- **What it is**: Google Antigravity CLI wrapper around Gemini 3.5 high-flash. Authored as a coding model, benchmark-comparable to Claude Opus 4.7 and Codex gpt-5.5 on coding tasks.
- **Speed**: ~4x faster than Opus 4.7. This is the load-bearing property — cheap to ask for parallel opinions.
- **Allowed roles** (per project constitution + user directive 2026-05-20):
  - **Second / third opinion** (side runner) alongside codex critical-review or Claude tester.
  - **Tie-breaker** when codex + Claude disagree.
  - **Test-tool author** (quick sanity scripts, one-off probes).
  - **Report drafter / summarizer** (read state, point out anomalies).
  - **Diagnosis side-runner** in parallel with codex diagnosis.
- **Forbidden roles**:
  - NEVER **primary worker** for sprint implementation. Worker = codex or Claude, per existing sprint contract.
  - NEVER **sole tester** for a sprint. Tester gate requires codex- or Claude-class AI. Gemini may run alongside, not instead of.
  - NEVER **sole reviewer / sole judge** for an ADR, milestone closeout, or sprint acceptance.
  - NEVER **sole critical-reviewer** for memory or skill patches.
- **Reasoning**: model is new to this project. Benchmarks ≠ track record. Until Gemini has accumulated ≥3 successful side-runner deliveries in this repo, treat its output as a third datapoint not a deciding vote.

## CLI invocation

Installed: `/home/enric/.local/bin/agy`. Auth state: per-user OAuth at `~/.gemini/antigravity-cli/oauth_creds.json` (separate from `~/.gemini/oauth_creds.json` used by the standard `gemini` CLI — they share neither token nor session).

| Flag | Effect |
|---|---|
| `-p "prompt"` / `--print "prompt"` | One-shot non-interactive. Single stdout response. Use this for side-opinion calls. |
| `-i "prompt"` / `--prompt-interactive` | Interactive REPL with initial prompt. Use this only when you actually need follow-up turns. |
| `--dangerously-skip-permissions` | Auto-approve tool permissions. Equivalent to codex `--dangerously-bypass-approvals-and-sandbox`. |
| `--continue` / `-c` | Resume last conversation. |
| `--conversation <ID>` | Resume specific conversation. |
| `--add-dir <path>` | Add workspace directory (repeatable). |
| `--print-timeout 5m0s` | Default timeout for `-p` mode. |

Standard side-opinion call:

```bash
agy --dangerously-skip-permissions -p "Read these files: <paths>. Question: <one sentence>. Output: <expected structure>. Hard rules: <read-only, no tool side effects, no file writes>." 2>&1 | tee /path/to/gemini-opinion.md
```

## First-time setup

OAuth flow opens a browser URL and waits 30 s for either callback or pasted code. Cannot be completed from within a Claude Code session — user must run `agy -p "ping"` themselves (or `! agy -p "ping"` via the `!` shell escape in their next message) once to complete login. Token persists across sessions.

## Dispatch patterns

**Hard requirements that apply to every Gemini dispatch — set 2026-05-20 by user directive**:

1. **Always run in a named tmux window** inside the user's session — never inline. User must be able to watch and inject mid-flight, same as codex/Claude dispatches.
2. **Always interactive REPL mode** (`agy --dangerously-skip-permissions -i "<onboarding+task>"`) — not `-p` print mode. This keeps the session open so user can ask follow-ups and so we can resume via `--continue` if needed.
3. **Always prefix the task with the onboarding prompt** at `.agent/references/gemini-onboarding-prompt.md`. Reason: Gemini is new to this project and will behave inconsistently with Claude / codex unless explicitly briefed on PROJECT_CONSTITUTION.md + AGENTS.md + dispatching-gemini.md + the relevant skill.
4. **Always pipe-pane the tmux window to a log file** so the full transcript is captured for audit, same as codex/Claude dispatches.
5. **Always tear down the window** after delivery (the completion handler reports back to manager then kills the window).

### Pattern A — interactive tmux side-opinion (canonical pattern, use this for every Gemini dispatch unless you have a reason not to)

```bash
WT=/home/enric/src/wrf_gpu2           # or worktree path for sprint-isolated work
WIN=gemini-<role>-<short-task-tag>    # e.g. gemini-side-m5-s1-thompson
TASK_PROMPT=/tmp/gemini-task-${WIN}.md
FULL_PROMPT=/tmp/gemini-full-${WIN}.md
OUT=/path/to/sprint/folder/${WIN}.md
LOG=/path/to/sprint/folder/${WIN}.log
DONE_MARK=/path/to/sprint/folder/.${WIN}-done
EXIT_FILE=/path/to/sprint/folder/.${WIN}-exit

# 1. Write the task-specific prompt
cat > "$TASK_PROMPT" <<'TASK_EOF'
# Task

<the actual question + files to read + expected output structure>
TASK_EOF

# 2. Prepend the mandatory onboarding prefix
cat /home/enric/src/wrf_gpu2/.agent/references/gemini-onboarding-prompt.md \
    "$TASK_PROMPT" > "$FULL_PROMPT"

MGR_SESS=$(tmux display-message -p '#S')
MGR_WIN=$(tmux display-message -p '#I')
MGR_TARGET="${MGR_SESS}:${MGR_WIN}"

# 3. Open tmux window with agy in interactive mode, with workspace dir added
tmux new-window -d -t "${MGR_SESS}:" -n "$WIN" \
  "bash -lc 'echo \"[dispatch-${WIN}] agy started at \$(date -u)\"; agy --dangerously-skip-permissions --add-dir $WT -i \"\$(cat $FULL_PROMPT)\" 2>&1 | tee $OUT; ec=\$?; echo \"\$ec\" > \"$EXIT_FILE\"; touch \"$DONE_MARK\"; MSG=\"AGENT REPORT [gemini side-opinion / $WIN] exit=\$ec report=$OUT\"; tmux send-keys -t \"$MGR_TARGET\" \"\$MSG\"; sleep 5; tmux send-keys -t \"$MGR_TARGET\" Enter; sleep 1; tmux kill-window -t \"${MGR_SESS}:${WIN}\" 2>/dev/null || true'"

# 4. Pipe pane to log for full audit trail
tmux pipe-pane -t "${MGR_SESS}:${WIN}" -O "cat >> \"$LOG\"" 2>/dev/null || true
```

The user sees the window in their tmux session, can switch to it (`Ctrl-b w`), watch Gemini's reasoning live, and inject corrections via `tmux send-keys` or simply by typing in the window.

### Pattern B — parallel side-runner during sprint

Same as Pattern A. The fact that Gemini is running concurrent with codex/Claude is just a matter of dispatching multiple windows in parallel; no different invocation. Use unique `$WIN` per dispatch.

### Pattern C — quick test-tool author

Even for "write me a Python diagnostic" tasks, use Pattern A (tmux + interactive + onboarding). The script gets written to `$OUT`, which we then extract via `grep -A 9999 "^\`\`\`python" $OUT | head -...`. Reason: even small write-tasks must go through onboarding so Gemini knows the project's no-silent-write-to-governance constraint.

### Pattern D — synchronous one-shot

```bash
agy --dangerously-skip-permissions -p "$(cat .agent/references/gemini-onboarding-prompt.md && echo && cat /tmp/gemini-task.md)" 2>&1 | tee /tmp/gemini-out.md
```

**Operational status (2026-05-20)**: due to the agy-tmux-OAuth quirk documented below, Pattern D is currently the working default. Use Pattern A (tmux + interactive) once the OAuth-per-pty issue is resolved.

## Known agy quirk: re-OAuth on fresh tmux pty

Observed 2026-05-20: agy stores credentials at `~/.gemini/antigravity-cli/implicit/*.pb` but the cached credentials only authenticate the pty/process tree that completed the OAuth flow. When a fresh tmux `new-window` spawns a new pty and invokes `agy ... -i`, agy presents the OAuth URL again rather than reusing cached creds. From the manager's pty (where the OAuth was completed) `agy -p` works without re-auth.

**Operational consequences**:
- **For one-shot side opinions (Pattern D)**: dispatch directly from manager pty (no tmux new-window). Cached creds work. This is the current default for Gemini dispatches.
- **For tmux interactive (Pattern A)**: user must manually complete OAuth in the new tmux pane after launch, OR pre-warm a long-lived agy interactive session that subsequent dispatches reuse via `agy --continue`.
- **Workarounds to investigate**: (1) pre-warm an interactive agy session; (2) check if there's an `agy --auth-file` flag or env var; (3) inspect the `.pb` files to see if creds can be made portable across ptys; (4) use `opencode run -m google/gemini-3.5-flash` instead (currently blocked by root-owned `~/.local/share/opencode/snapshot/`).
- Manager owns resolving this quirk as a hygiene task; until then, Pattern D is the operational default and the dispatch is run inline from the manager pty (still visible to user via this session's terminal).

The user can still inject corrections mid-flight by typing in the manager pane — same as for any inline Bash command. Pattern A's "user injects via tmux" benefit is not lost, only temporarily routed through the manager pane.

## Hygiene

- **Always tee output**: capture to a path-named file alongside other agent reports. Treat the report exactly like a codex or Claude agent report.
- **Cite the prompt**: save the prompt alongside the output so future reviewers can audit the question framing.
- **Tag in sprint contract**: when a Gemini opinion contributes to a decision, name it explicitly in the decision memo (e.g. "Codex Accept, Claude Reject, Gemini Accept-with-reservations → manager rules Accept-with-required-fixes").
- **Tmux window naming**: prefix with `gemini-` so the janitor can distinguish AI families.
- **Token cost**: nominally cheap due to speed, but parallelism multiplies wall-clock cost. Do not dispatch more than two Gemini side-runners simultaneously (1× Claude opus + 1× codex + 1-2× Gemini is the upper bound for a single decision point).

## Track record (update after each delivery)

| Date | Task | Outcome | Verdict |
|---|---|---|---|
| 2026-05-20 | M5-S1 attempt-4 third-opinion (Path A vs B) | Delivered Path-B recommendation with cited per-field rel-err evidence (`qc`=0.999998, `qr`=4.5e7, `qg`=9.8e8), constructed a non-trivial counterargument (PBL discovery > microphysics table fix), and raised one novel reviewer check that neither Claude nor codex had surfaced (HLO unroll / compile-OOM risk from baked lookup tables). Saved to `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/gemini-third-opinion.md`. **Note: this dispatch used pre-update Pattern D (inline `-p`, no onboarding prefix, no tmux). Pattern A is the target dispatch pattern; Pattern D is the current operational default pending resolution of the agy-tmux-OAuth quirk.** | **Useful side-runner**. Compile-OOM check was a real value-add. Confirmed Gemini can be adversarial about its own recommendation. 1/3 toward role promotion. |
| 2026-05-20 | Onboarding smoke test (Pattern D after Pattern A blocked by agy-tmux-OAuth quirk) | Three onboarding questions on PROJECT_CONSTITUTION + AGENTS + dispatching-gemini. All three answered correctly with file:line citations (`AGENTS.md:22`, `dispatching-gemini.md:15-19`, `PROJECT_CONSTITUTION.md:5`). High self-confidence. Generated its own track-record line as requested. Saved to `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/gemini-smoke-onboarding.md`. | **Onboarding prefix works.** Gemini's behavior in-project is now in-line with Claude/codex conventions (file:line evidence, terse, scoped). Counts as 0.5 toward promotion (sanity check, not analytic delivery). 1.5/3. |
| 2026-05-20 | M5-S1 attempt-4 parity-numbers sanity check (side-runner #2) | Verified diagnosis error budget holds (process-order 87% reduction matches, Ni 91% matches, lookup-table residual matches 20-30%). **Identified a real coefficient bug at `thompson_column.py:277-278`** — JAX uses `6.0 / clip` where WRF source `module_mp_thompson.F.pre:1920` uses `cie(2) / clip = 4.0 / clip`. Off by factor 1.5 in clipped `lami`, propagates into `Ni` and ice mass partition. Manager verified independently (WRF `cie(2) = bm_i + mu_i + 1 = 4.0`, JAX substituted `cig(2) = gamma(4) = 6`). Saved to `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/gemini-second-opinion-parity-sanity.md`. Manager injected the finding into tester A4's running tmux session for independent verification. | **High-value side-runner**. Found a specific 1-line bug that worker, diagnosis codex, and manager all missed. Concrete value-add: some fraction of the "20-30% lookup-table residual" is actually this typo, not table proxies. 2.5/3 toward role promotion. |

Manager updates this table after each Gemini delivery so future read-throughs can calibrate confidence in Gemini's role.

## Cross-links

- `.agent/rules/cross-model-review-policy.md` — three-AI review structure
- `.agent/skills/resolving-cross-model-disagreements/SKILL.md` — debate workflow
- `.agent/skills/managing-sprints/SKILL.md` — dispatch hygiene
