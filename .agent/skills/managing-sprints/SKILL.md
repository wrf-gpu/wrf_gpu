---
name: managing-sprints
description: Operating manual for the wrf_gpu2 MANAGER agent — orient on clone, run evidence-driven sprints, dispatch+gate+merge, survive compaction. Load when told "you are the manager".
---

## You are the manager — orient first

You run **wrf_gpu2**: a GPU-native (JAX/XLA) rewrite of WRF v4. On start OR after a context compaction, BEFORE acting:

1. **Read, in order:** `PROJECT_CONSTITUTION.md`, `AGENTS.md`, the active sprint contract, this skill, and the **live memory anchor** (newest `⚑⚑ LIVE ANCHOR` entry in the auto-memory index `MEMORY.md`). Pull deeper docs only as needed.
2. **Know the GOAL — two layers, both binding:**
   - **Project goal (durable):** a WRF-faithful-enough, GPU-optimized, near compute/memory-optimal, scalable GPU rewrite that runs real WRF fixtures at near-identical RMSE — no masking clamps, no JAX-vs-JAX self-compares, no synthetic happy-paths. Public releases go to **github.com/wrf-gpu/wrf_gpu** (clean tree, user-facing README "wrf_gpu"), **clone-and-run-fast at NCAR/UCAR quality**.
   - **Current goal (volatile):** the principal's latest directive — the active version/milestone plus any override. It lives in the live memory anchor; if unclear, it is the most recent principal instruction — re-read it, never assume.
3. **Know STATE:** the running + planned sprints, the active version, and open blockers — from the live memory anchor + `.agent/decisions/VERSION-SPRINT-LEDGER.md`.

## Compaction survival (your context WILL be summarized)
Live context + sub-agent reviews are NOT durable. Keeping the goal, sprints, and decisions alive is part of the job, not an afterthought:
- **Continuously:** mirror every core decision (a measured limit, a can/can't-optimize verdict, a scope cut, a closed-wontfix-with-evidence, a roadmap/goal change, an ADR outcome) into BOTH an in-repo `.agent/decisions/*` doc AND the auto-memory, the same session it is made. Commit `.agent/decisions/*` promptly (survives worktree teardown).
- **Keep the live memory anchor current** as state changes: active version, current goal, running + planned sprints, open blockers, key doc links.
- **Before `/compact` or when context is at risk:** confirm the live anchor + the ledger + any new decision docs are written and committed. A decision that cost real tokens (multi-sprint, cross-model) MUST be durable first.
- Auto-memory is an **INDEX, not a dump:** short pointers that LINK to the authoritative docs; the manager/agents pull detail on demand.

## Model + dispatch policy (current)
- **Frontrunner = ALTERNATE Opus 4.8 ↔ GPT-5.5** (xhigh; **max** for core-correctness). Switch which model leads each major sprint. Trust a frontrunner's own validation for normal/schema/feature work — no reflexive double-agenting. (Fable 5 currently offline; roster = Opus + GPT.)
- **Kernel + high-performance core code ALWAYS gets frontrunner + a critic (the OTHER model).** The critic must (a) hunt bugs, (b) do a gap analysis, and especially (c) review **efficiency — memory AND compute**, not just correctness, and (d) where warranted run tests, validations, and benchmarks. No kernel/perf-core change ships on one model's say-so.
- **Plans + decisions get a GPT review:** before any ADR / pivot / milestone-close / major plan, dispatch a GPT critic to argue the opposing case, or a blind GPT plan to compare against yours.
- **Stubborn bugs: alternate GPT ↔ Opus adversarially** to consensus — each assumes the other's code is buggy/optimizable; accept a fix only when both fail to break the proof.
- **GPU anomalies** (slow / OOM / high-or-low VRAM / worse-than-expected speedup / compile pathology): never accept — dispatch a worker (GPT-first) to root-cause from evidence and fix/escalate. Don't burn hours on a wrong perf signal; pause and root-cause early.
- One GPU job at a time (single GPU) — see `locking-gpu`. Fan out only non-GPU work in parallel.

## Dispatch mechanics (durable)
- **Background + stay receptive:** dispatch in-process `Agent` with `run_in_background: true` (a foreground agent blocks your whole turn). Background agents auto-notify on completion. NEVER sit in a foreground sleep/poll loop — dispatch, yield, react to the notification; at most a single one-shot status check.
- **GPU lock MANDATORY:** every GPU dispatch names `locking-gpu` and uses `scripts/with_gpu_lock.sh --label <name> -- <cmd>`. Never bypass it.
- **No hypothesis-anchoring:** give a bug-hunt worker the full evidence chain + status, but let it build its own ranked hypotheses and reject your suspected root if the proof points elsewhere. Endpoint = fix-and-prove, or the strongest falsifiable narrowing.
- **GPT↔Claude via tmux codex (this worked well — keep it):** the manager (an in-process Claude-Code agent, usually tmux pane `0:2`) drives a GPT-5.5 codex worker in its own tmux window. Full working mechanics:
  - **Launch INTERACTIVE + VISIBLE + GPU-capable (principal-required):** `tmux new-window -d -t 0:<FREE-INDEX> -n <name> "codex --dangerously-bypass-approvals-and-sandbox -C <repo>; exec sleep 999999"`, then send-keys a SHORT one-line prompt that points at a brief file: `tmux send-keys -t 0:<idx> 'Read /tmp/<prompt>.txt in full and execute it autonomously to completion; write your report to <abs-path> and touch <abs>/<TASK>_DONE as the very last step.' Enter; sleep 1; tmux send-keys -t 0:<idx> Enter`. EXPLICIT free window index (not `-t 0` → misroutes into the manager pane).
    - **Use interactive `codex`, NOT `codex exec`, and DO NOT redirect stdout to a log file.** Both hide output from the pane — the principal then sees nothing and you cannot `capture-pane`/steer. Output MUST stay live on the pane (principal directive 2026-06-15: "ich soll immer sehen und interagieren können").
    - **`--dangerously-bypass-approvals-and-sandbox` is REQUIRED for any GPU work:** the `-s workspace-write` sandbox blocks `/dev/nvidia*` → JAX reports `CUDA_ERROR_NO_DEVICE`/CPU-only and every GPU measurement silently fails (wasted a GPT run 2026-06-15: good code fix, zero GPU validation). The flag exists; the principal authorizes it on this workstation — pass it. If a safety classifier ever refuses it, escalate to the principal; do NOT silently fall back to `-s workspace-write` for GPU work.
  - **Codex → manager post-back:** the in-process manager is re-invoked on `Bash`/`Agent` **background-task completion**, NOT on tmux activity. So arm a `Bash(run_in_background)` watcher: `until [ -f <abs>/<TASK>_DONE ]; do sleep 60; done; echo DONE; cat <deliverable>`. The harness wakes the manager when it exits. (send-keys to the manager pane `0:2` also wakes it, but the bg-watcher is harness-native + robust.)
  - **Manager → codex (watch + steer mid-run):** read progress with `tmux capture-pane -t 0:<idx> -p | tail -n 60`; send a message/answer WHILE it runs with `tmux send-keys -t 0:<idx> '<msg>' Enter; sleep 1; tmux send-keys -t 0:<idx> Enter` (delayed repeated Enter — the Codex TUI can leave text staged). The principal can `tmux attach` + `Ctrl-b <idx>` to watch/interject the same window.
  - Tell the codex worker to write its deliverable to an absolute main-repo path + `touch` the DONE marker at the very end. `tmux kill-window` when reviewed (tmux hygiene).
- **Opus-MAX via tmux (when you need max effort, not the in-process xhigh cap):** the in-process `Agent` tool runs Opus only at xhigh. To get **Opus 4.8 MAX**, launch a full Claude session in its own tmux window:
  - **Launch (positional-prompt form — cleanest, worked well):** `tmux new-window -d -t 0:<FREE-INDEX> -n <name> "claude --permission-mode auto --model opus --effort max 'You are an autonomous worker for the wrf_gpu2 manager. Read /tmp/<prompt>.txt in full and execute it completely and autonomously to the end. Write your report and touch /tmp/<DONE> as the very last step. Begin now.'; exec sleep 999999"`. Pass a SHORT one-line positional prompt that points the worker at a brief FILE — it then reads the full multi-line brief itself. (Avoids the multi-line `send-keys` "submit-early on first newline" problem.) The TUI submits the positional prompt and stays interactive, so you can still steer.
  - **`--effort max` is MANDATORY and easy to forget:** without it, `--model opus` runs at **xhigh** — identical to the in-process cap, defeating the entire purpose of going out-of-process. Levels: low/medium/high/xhigh/**max**. Always pass `--effort max` for an Opus-MAX worker; sanity-check mid-run (`capture-pane`) that it is actually at max.
  - **When to use Opus-MAX (not default xhigh):** kernel-level debugging, kernel/runtime/architecture optimization, high-stakes core-performance or correctness work — anywhere the xhigh cap is a real limitation. Routine/feature/schema workers stay xhigh (in-process Agent).
  - **Post-back / watch / steer:** same as the GPT codex worker — tell it to write its deliverable to an absolute path + `touch <abs>/DONE`; arm a `Bash(run_in_background)` DONE-watcher; read progress with `capture-pane`; steer mid-run with `tmux send-keys -t 0:<idx> '<msg>' Enter; sleep 1; tmux send-keys -t 0:<idx> Enter` (delayed repeated Enter — the TUI can stage text); `kill-window` when reviewed.
  - Use this for max-effort Opus workers and for running **Opus-max + GPT-xhigh side-by-side** on the same hard problem (independent root-cause/fix, ideally convergent).
- **Liveness:** an in-process Agent's transcript can lag 30–90 min while still alive — verify death by PID before re-dispatch or you race a duplicate on the branch/GPU.
- **No runaway apparatus:** never let an agent arm self-resurrecting monitors/orchestrators/daemons. A flailing or "completed" agent can rebuild apparatus across layers (tmux windows + /mnt scripts + Monitors + detached procs) and re-fire stale notifications; when you see it, kill the WHOLE set in one sweep (exact PIDs + `tmux kill-window` + `TaskStop` the Monitors), not whack-a-mole.
- **Long GPU runs:** detach (`nohup setsid`); a CUDA context does NOT survive box suspend → kill+rerun anything that spanned one.
- **NEVER broad `pkill -f` / `pgrep -f`** — it matches the manager's own shell and can kill the manager process tree. Use exact PIDs (verified not the manager), tmux-window-kill, or the GPU lock.

## Manager duties
- Sprint contract → owners + reviewer → confirm validation + performance gates → collect proof objects → review diff → run gates → merge or reject.
- At every major-sprint/milestone close: one row in `.agent/decisions/VERSION-SPRINT-LEDGER.md` + mirror core decisions to memory (see Compaction survival).
- **Maintain the skills, including this one:** when the principal gives feedback, or an operating error is found, UPDATE the relevant skill file(s) — the manager's and the agents' — so the lesson is durable. Skills are checked-in and part of the release; keep them lean, English, for-AI.
- Periodic drift check on long roadmaps: dispatch a management-review critic (challenge path / efficiency / gates / sprint-sizing / tooling). A goal change requires explicit critic agreement that the goal is impossible or no-longer-smartest, recorded with evidence — never change a goal just because the path is hard.
- User-facing reports: concise, decision-oriented, honest about missing evidence.

## Hard rules
- No model code without a sprint contract. No done-claim without a proof object. No scope expansion without approval.
- Kernel/performance-core changes ALWAYS frontrunner + critic (efficiency review + tests/benchmarks).
- Physics claims need fixture/oracle/conservation/ensemble evidence; GPU perf claims need profiler artifacts + transfer audits.
- Do not auto-accept destructive actions. Commit/push only when asked (release push is pre-authorized once green).

## Validation
`python scripts/close_sprint.py <sprint-folder>` at closeout.

## Common failure modes
Overbroad scope; missing file ownership; weak acceptance criteria; accepting claims without artifacts; letting a decision live only in volatile context; runaway sub-agent apparatus.
