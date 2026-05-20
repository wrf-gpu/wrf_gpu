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

### Pattern A — quick side-opinion (preferred default)

Use when codex or Claude has produced a primary judgment and you want a cheap second datapoint:

```bash
PROMPT_FILE=/tmp/gemini-opinion-prompt.md
cat > "$PROMPT_FILE" <<'EOF'
You are a side-opinion agent. The primary judgment was rendered by <Claude Opus 4.7 / codex gpt-5.5>.
Your job: independently read <files> and give a one-paragraph verdict.
Do NOT modify anything. Read-only. No tool use that writes.
<the question>
EOF
agy --dangerously-skip-permissions -p "$(cat "$PROMPT_FILE")" 2>&1 > /tmp/gemini-opinion.md
```

Output then folds into manager's decision memo as one of N opinions.

### Pattern B — parallel side-runner during sprint

Use when codex/Claude is running a heavy sprint and you want Gemini to chase a different angle in parallel. Same prompt file pattern, dispatch in tmux window so it runs concurrent:

```bash
tmux new-window -d -t "$MGR_SESS:" -n gemini-side -c /home/enric/src/wrf_gpu2 \
  "bash -lc 'agy --dangerously-skip-permissions -p \"\$(cat /tmp/gemini-prompt.md)\" 2>&1 | tee /path/to/report.md; echo done > /path/to/.done'"
```

Manager combines findings from codex + Claude + Gemini.

### Pattern C — quick test-tool author

Use when you need a small diagnostic script (e.g. "write me a Python script that compares two NPZ files field-by-field with relative-error histograms"). Gemini's speed makes this cheaper than dispatching codex.

```bash
agy --dangerously-skip-permissions -p "Write a Python script that <does X>. Print only the script body, no commentary. The script must be self-contained, use only numpy + scipy, fit in <50 lines." > /tmp/diagnostic.py
```

## Hygiene

- **Always tee output**: capture to a path-named file alongside other agent reports. Treat the report exactly like a codex or Claude agent report.
- **Cite the prompt**: save the prompt alongside the output so future reviewers can audit the question framing.
- **Tag in sprint contract**: when a Gemini opinion contributes to a decision, name it explicitly in the decision memo (e.g. "Codex Accept, Claude Reject, Gemini Accept-with-reservations → manager rules Accept-with-required-fixes").
- **Tmux window naming**: prefix with `gemini-` so the janitor can distinguish AI families.
- **Token cost**: nominally cheap due to speed, but parallelism multiplies wall-clock cost. Do not dispatch more than two Gemini side-runners simultaneously (1× Claude opus + 1× codex + 1-2× Gemini is the upper bound for a single decision point).

## Track record (update after each delivery)

| Date | Task | Outcome | Verdict |
|---|---|---|---|
| 2026-05-20 | M5-S1 attempt-4 third-opinion (Path A vs B) | Delivered Path-B recommendation with cited per-field rel-err evidence (`qc`=0.999998, `qr`=4.5e7, `qg`=9.8e8), constructed a non-trivial counterargument (PBL discovery > microphysics table fix), and raised one novel reviewer check that neither Claude nor codex had surfaced (HLO unroll / compile-OOM risk from baked lookup tables). Saved to `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/gemini-third-opinion.md`. | **Useful side-runner**. Compile-OOM check was a real value-add. Confirmed Gemini can be adversarial about its own recommendation. 1/3 toward role promotion. |

Manager updates this table after each Gemini delivery so future read-throughs can calibrate confidence in Gemini's role.

## Cross-links

- `.agent/rules/cross-model-review-policy.md` — three-AI review structure
- `.agent/skills/resolving-cross-model-disagreements/SKILL.md` — debate workflow
- `.agent/skills/managing-sprints/SKILL.md` — dispatch hygiene
