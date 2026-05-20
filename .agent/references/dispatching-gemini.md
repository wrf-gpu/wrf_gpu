# Dispatching Gemini 3.5 (agy CLI) — Reference

Third AI available to this project alongside Claude (Opus 4.7 / Sonnet 4.6) and Codex (gpt-5.5). Per user authorization 2026-05-20.

## Capability + constraints

- **What it is**: Google Antigravity CLI wrapper around Gemini 3.5 high-flash. Authored as a coding model, benchmark-comparable to Claude Opus 4.7 and Codex gpt-5.5 on coding tasks.
- **Speed**: ~4x faster than Opus 4.7. This is the load-bearing property — cheap to ask for parallel opinions.
- **Allowed roles** (per user directive 2026-05-20 evening, updated after Gemini's first two deliveries proved high-value):
  - **Second / third opinion** (side runner) alongside codex critical-review or Claude tester. **Always-on for any non-trivial decision.**
  - **Bug-fix parallel-pair (mandatory)**: every confirmed issue dispatches ≥2 AIs to identify and propose a fix. One of the two MUST be Gemini. The other is codex or Claude. Manager combines candidates. Rationale: hallucination risk on Gemini drops to ~zero when paired with a slower, deeper AI; speed advantage stays useful. Without the pair, single-Gemini fixes could ship a hallucinated coefficient. With the pair, the risk is bounded.
  - **Large / complex reviews — Gemini in parallel** (alongside the primary reviewer, not as the binding reviewer). Primary reviewer remains Claude Opus 4.7 (and codex for critical-review on memory/skill/governance patches). Gemini's parallel report is supplementary and feeds into the manager's decision memo.
  - **Tools / sidecars / scripts / report drafts / quick diagnostic probes**: unconstrained. Use Gemini whenever it brings the project forward — speed is the value.
  - **Sprint frontrunner — codex gpt-5.5 xhigh remains the default primary worker**. Gemini does not replace codex for new sprint implementation. Gemini may run as a second worker in a parallel-pair on bug-fix sprints per the rule above.
- **Forbidden roles** (still apply):
  - NEVER **sole primary worker** for sprint implementation. Workers are codex or Claude, with optional Gemini parallel-pair when the manager dispatches one.
  - NEVER **sole tester** for a sprint. Tester gate requires codex- or Claude-class AI. Gemini may run alongside, not instead of.
  - NEVER **sole reviewer / sole judge** for an ADR, milestone closeout, or sprint acceptance.
  - NEVER **sole critical-reviewer** for memory or skill patches.
- **Reasoning**: Gemini has demonstrated high-value side-runner output (1 novel reviewer check + 1 specific coefficient bug found in first two deliveries) but the project still has limited track record on its hallucination profile. The parallel-pair rule eliminates hallucination risk on consequential decisions; the unconstrained-tooling rule captures Gemini's speed value where the risk is low.

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

## Known agy quirk: daily quota exhaustion (silent failure pattern)

Observed 2026-05-20 evening after ~6 successful Gemini deliveries: agy starts returning empty output with exit code 0. The actual error is logged only to `~/.gemini/antigravity-cli/cli.log`:

```
RESOURCE_EXHAUSTED (code 429): Individual quota reached. Contact your administrator to enable overages. Resets in 3h26m46s.
```

**Detection**: if `agy -p "..."` returns empty stdout and exit 0, tail `~/.gemini/antigravity-cli/cli.log` for `RESOURCE_EXHAUSTED` to confirm quota vs. other failure.

**Workaround**: wait for the named reset window (typically ~24h sliding window on Google's free tier). Quota event is per-user-per-day.

**Operational rule**: budget ~5-8 Gemini dispatches per day during heavy bug-fix-parallel-pair work. Cheap-ping calls (smoke tests, ping) also count. When approaching limit, reserve remaining capacity for high-leverage side-audits (bug-class identification) over low-leverage ones (re-verification of already-confirmed fixes).

**Fallback**: `opencode run -m google/gemini-3.5-flash` accesses the same Gemini family via a separate quota pool. Currently blocked by a root-owned `~/.local/share/opencode/snapshot/` (one-time `sudo chown` needed). Once unblocked, opencode is the natural fallback when agy quota trips.

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
| 2026-05-20 | M5-S1 attempt-4 parity-numbers sanity check (side-runner #2) | Verified diagnosis error budget holds (process-order 87% reduction matches, Ni 91% matches, lookup-table residual matches 20-30%). **Identified a real coefficient bug at `thompson_column.py:277-278`** — JAX uses `6.0 / clip` where WRF source `module_mp_thompson.F.pre:1920` uses `cie(2) / clip = 4.0 / clip`. Off by factor 1.5 in clipped `lami`, propagates into `Ni` and ice mass partition. Manager verified independently. Tester A4 (Claude Opus 4.7 xhigh) independently confirmed in adversarial-probe section. Saved to `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/gemini-second-opinion-parity-sanity.md`. | **High-value side-runner**. Found a specific 1-line bug that worker, diagnosis codex, and manager all missed. 2.5/3 toward role promotion. |
| 2026-05-20 | M5-S1 attempt-5 parallel side-audit (bug-fix parallel-pair, side-audit role) | While codex worker A5 was applying the lami fix, Gemini scanned `thompson_column.py` + `thompson_constants.py` + `thompson_saturation.py` for OTHER coefficient confusions. **Identified a SECOND confirmed bug**: graupel sublimation/melting (`thompson_constants.py:90,92` and `thompson_column.py:463,492`) hardcodes `* 2.0` and `ilamg**CRE11` (rain values) where WRF live code (`module_mp_thompson.F.pre:2761,2872-2875`) uses `* cgg(11) = 1.7042533` and `ilamg**cge(11)` where `cge(11) = 2.8204808` for graupel mp_physics=8 (`:104,156,763`). Manager verified independently. Folded into attempt-5 scope as Fix 6 via mid-flight injection to worker A5. Also dismissed 2 other suspects with rationale. Saved to `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/gemini-side-audit-attempt5.md`. | **High-value side-runner #2**. Found a second specific 1-line-class bug that even tester A4 (Claude Opus) missed. The parallel-pair rule paid off immediately on first sprint where it was applied. Gemini's role is now firmly proven. Promotion gate ≥3 deliveries reached and exceeded; role-by-role expansion (per user directive evening 2026-05-20) already in effect. |
| 2026-05-20 | Stage-M4 architectural review (user-commissioned, ad-hoc) | Independent review of project plan w.r.t. 4x+ performance target on RTX 5090. Flagged 3 issues: FP64 throttling (existential), nesting omission (v0-scope), launch-bound latency on outer nested domains. User accepted and approved ADR-007 precision-policy sprint after M5-S1 close. Saved to `.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md`. | **High-value architectural review** — caught a project-existential precision-policy oversight in ADR-003. Manager + user concur on remediation path. |
| 2026-05-20 | M5-S1 attempt-5 fix-verification side-runner (planned) | Two consecutive dispatch attempts failed silently (empty stdout, exit 0). Root cause discovered via `~/.gemini/antigravity-cli/cli.log`: `RESOURCE_EXHAUSTED (code 429): Individual quota reached. Resets in 3h26m46s.` Quota exhaustion documented as known agy quirk (see above). Fix-verification was supplementary; Claude Opus reviewer remains binding. | **Operational signal** — daily Gemini quota hit after ~6 successful dispatches in a heavy day. Documentation updated. |

Manager updates this table after each Gemini delivery so future read-throughs can calibrate confidence in Gemini's role.

## Cross-links

- `.agent/rules/cross-model-review-policy.md` — three-AI review structure
- `.agent/skills/resolving-cross-model-disagreements/SKILL.md` — debate workflow
- `.agent/skills/managing-sprints/SKILL.md` — dispatch hygiene
