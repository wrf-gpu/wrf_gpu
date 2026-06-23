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
- **v0.18 feature-work policy (principal 2026-06-16):** **GPT is the DEFAULT frontrunner / primary agent** for v0.18.
  - **GROUP BY FAMILY (principal 2026-06-16):** the endpoint scheme ports are "mostly a function per schema" → put a WHOLE physics family in ONE large agent run (all MP in one, all CU in one, all PBL in one, all RA in one, all LSM/surface in one) so the code context is built ONCE — running them one-by-one re-builds context every time = token + wallclock waste. **Run families IN PARALLEL** where sensible (separate `worker/gpt/v018-<family>` branches off the trunk; the per-family physics modules are disjoint; the shared registry/catalog/State get a set-UNION merge at integration). **Maximize sensible parallelism — wallclock-to-new-version matters (correctness first).**
  - **EACH scheme must be tested** (real-oracle / bit-identity unit-validation + GPU smoke; coupled-gate where the harness supports it). Already-GREEN (harvest) → implement+validate, no critic. Long-tail family ports = a BIG important step → **no per-scheme critic; instead ONE critic (the OTHER model) checks the whole family-batch** after the frontrunner reports: a GPT-frontrun family → **Opus critic**; an Opus-frontrun family (i.e. run on Opus as ≥3-GPT overflow) → **GPT critic**.
  - **lean MAXIMAL-PERFORMANCE design is MANDATORY** (GPU-efficient, no hot-path alloc, vectorized/fused where bit-safe).
  - **GPT capacity cap + continue-sweep (principal 2026-06-16):** the GPT/codex account is rate-limited. **NEVER run more than 3 GPT workers in parallel.** Need a 4th+ parallel worker? Launch **Opus 4.8** instead — **xhigh** (in-process `Agent`) for normal family ports, **`--effort max`** (tmux) for kernel / extremely-complex work. **Still prioritize GPT** for up to 3 slots (it's the default frontrunner); Opus fills the overflow. **Every ~30 min in the monitor loop, `capture-pane` each GPT window** and, for any showing *"Current model at capacity / try another model"* (or a rate-limit/429), send **`continue`** via the universal delayed-repeated-Enter submit. A background continue-sweep (`/tmp/v018_continue_sweep.sh`-style: grep panes for the capacity/rate-limit string, auto-`continue` only those) automates this safely — it must NOT `continue` a window idle for any other reason (e.g. awaiting a manager decision).
  - **GPT = PRIMARY frontrunner, Opus = overflow only (principal 2026-06-16, reaffirmed):** GPT leads by default for ALL v0.18 work; switch a NEW job to Opus (xhigh for normal/feature/family, `--effort max` for kernel-or-extremely-complex) ONLY when ≥3 GPT are already running in parallel. Do NOT re-host an in-flight Opus-overflow job back onto GPT just because a slot frees — finish it where it runs.
  - **Critic scope = BIG important steps ONLY (principal 2026-06-16):** the vice-versa (OTHER-model) critic/verify stays MANDATORY for big, important steps — family-batch ports, kernel/perf-core changes, ADR/pivot/milestone-close, major plans (GPT-frontrun → Opus critic; Opus-frontrun → GPT critic). For SMALL fix tasks, validation/verification agents, and similar routine work → use a SINGLE agent that PROTOCOLS the PROOF of work done; NO second-model critic (over-criticking small work wastes the rate-limited GPT budget + wallclock).
  - **EXCEPTION — do NOT group, keep solo + critic:** genuinely hard kernel-level changes where validation is the bottleneck anyway, or kernel-impact (dycore/State/ABI, e.g. K2 multi-GPU, WSM7/WDM7 qh-activation, an oracle-build-first scheme like RRTMG). Those get frontrunner + critic.
- **Kernel + high-performance core code ALWAYS gets frontrunner + a critic (the OTHER model).** The critic must (a) hunt bugs, (b) do a gap analysis, and especially (c) review **efficiency — memory AND compute**, not just correctness, and (d) where warranted run tests, validations, and benchmarks. No kernel/perf-core change ships on one model's say-so.
- **Plans + decisions get a GPT review:** before any ADR / pivot / milestone-close / major plan, dispatch a GPT critic to argue the opposing case, or a blind GPT plan to compare against yours.
- **Stubborn bugs: alternate GPT ↔ Opus adversarially** to consensus — each assumes the other's code is buggy/optimizable; accept a fix only when both fail to break the proof.
- **GPU anomalies** (slow / OOM / high-or-low VRAM / worse-than-expected speedup / compile pathology): never accept — dispatch a worker (GPT-first) to root-cause from evidence and fix/escalate. Don't burn hours on a wrong perf signal; pause and root-cause early.
- One GPU job at a time (single GPU) — see `locking-gpu`. Fan out only non-GPU work in parallel.

## Dispatch mechanics (durable)
- **Background + stay receptive:** dispatch in-process `Agent` with `run_in_background: true` (a foreground agent blocks your whole turn). Background agents auto-notify on completion. NEVER sit in a foreground sleep/poll loop — dispatch, yield, react to the notification; at most a single one-shot status check.
- **Bidirectional tmux back-channel (workers ↔ manager — principal 2026-06-16, "find consensus like the Opus-max session"):** every dispatched tmux worker is TOLD the manager's pane (the manager's own current window — find it with `tmux display-message -p '#S:#I'`, usually `0:2`) so it can **ask the manager mid-run**. Tell each worker: *"if you hit a decision you can't resolve, or you disagree with a manager instruction, send your question to the manager pane `<S:I>` with the **DELAYED-REPEATED-ENTER** pattern AND append it to `<your>_QUESTION.md`, then keep working the unblocked parts."* **UNIVERSAL TMUX-SUBMIT PATTERN (MANDATORY for EVERY send to a TUI agent — manager→worker AND worker→manager, codex AND claude):** the TUI **stages** pasted text; an Enter attached to the same `send-keys`, or a single Enter after only `sleep 1`, is **unreliable** (especially for long messages) and leaves the message staged so a human has to press Enter. Always **separate the text from the Enter, wait for the paste to ingest, then send Enter 3+ times with gaps AND verify-and-repeat with `capture-pane` until the input line is empty** (2 Enters is NOT enough for long messages — principal had to press Enter manually repeatedly):
  **CANONICAL — ALWAYS use the tested helper `scripts/tmux_submit.sh <pane> '<message>'`** (do
  NOT hand-roll a fixed `for k in 1 2 3; do Enter; done` — that is exactly what left messages
  staged and made the principal press Enter manually, 2026-06-19 and 2026-06-21). The helper
  pastes the text, then sends Enter ONE AT A TIME (delayed ~1.3 s) and after each checks whether a
  distinctive SIGNATURE of the message is still sitting in the bottom input region — it stops only
  when the message has LEFT the input (submitted, or queued behind a running task — both clear the
  input line). It prints `SUBMITTED (input cleared)` (rc 0) or `WARN` (rc 1, verify manually).
  **DO NOT trust an "agent is RUNNING" signal as proof of submit** — that was the 2026-06-21 bug:
  claude shows `esc to interrupt` as a PERMANENT status-bar legend (instant false positive after 1
  Enter while text is staged), and if you send to an already-busy agent the `Working`/`· 1m 45s ·`
  timer you detect is the PRIOR task, not yours, so your message sits QUEUED needing a real Enter.
  **OPERATING RULE: prefer sending only to an IDLE agent** (at its prompt, empty input) — capture-
  pane first; if it is mid-task, wait for it to go idle (or send and then re-confirm the queue took
  + that it is processed when the agent next idles). The equivalent manual fallback (only if the
  helper is unavailable) — paste text, sleep 2, then Enter+`capture-pane` REPEAT until the pasted
  text is gone from the input region (NOT a fixed count; NOT on a "running" legend):
  ```
  scripts/tmux_submit.sh <pane> '<message>'   # <-- preferred, verified-until-running
  # manual fallback ONLY:
  tmux send-keys -t <pane> -- '<message>'      # text only, no Enter
  sleep 2                                       # let the TUI ingest the paste
  # then: send Enter + `capture-pane` check, REPEAT until the input clears / "esc to interrupt"
  #   shows (NOT a fixed count — long messages routinely need >3); up to ~10x.
  ```
  Tell each worker to use exactly this to reach the manager pane `<S:I>` (and append the question to `<your>_QUESTION.md`). The manager uses the same pattern for every send to a worker and **MUST verify-and-repeat** with `capture-pane`: long messages routinely need MORE than 2 Enters, and a staged-but-unsubmitted message means a human has to press Enter manually (principal hit this repeatedly 2026-06-19) — so keep sending Enter (with 1s gaps) until `capture-pane` shows the input line is empty. Back-and-forth → **consensus** (the Opus-max ↔ GPT pattern: each assumes the other may be wrong; converge on the strongest proof). Keep the loop tight; record the consensus outcome in the sprint proof.
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
- **Release SHIP-GATE (principal 2026-06-16):** tag+push a release ONLY when EVERY known issue is (a) solved + green-tested, (b) proven computationally not-solvable, or (c) proven of no real-world-application relevance — with (b)/(c) DOCUMENTED and carried. No premature ship. For a feature-complete release, every scheme must be operational+oracle-green OR reference-only-WITH-a-real-oracle (fail-closed) OR proven-unsolvable/irrelevant — "reference-only without an oracle" is a silent gap and not allowed.

## Validation
`python scripts/close_sprint.py <sprint-folder>` at closeout.

**Identity plots + run-data retention (principal 2026-06-16):** every LARGER validation run (72h field-parity gates, coupled-family gates, big benchmarks) must **retain its paired GPU+CPU wrfout + proof data** so identity-proof dashboards are buildable later — and **generate the identity dashboard at the time** where feasible (`scripts/build_identity_proof_plots.py`, CPU-only). The v0.17 README could not build a needed plot because that run's data was not retained — do not repeat. **Every release's README refresh (a dedicated cleanup+refresh worker) MUST update + include ALL current identity plots for the larger runs** — never ship stale or missing plots.

## Version & remote sync (MANDATORY — Enric 2026-06-21)
The **local checkout, the local-project git (`origin`, private), and the organization git
(`wrfgpu`, public) must ALWAYS carry the SAME version** and be kept in sync, documented in
CHANGELOG / RELEASE_NOTES + the ledger. After EVERY release/tag:
- Verify `src/gpuwrf/__init__.py` + `pyproject.toml` version match across **local working
  tree == origin/main == wrfgpu/main**, and the tag exists on both remotes.
- **Gotcha that bit us:** doing releases via throwaway worktrees off `origin/main` advances
  the *remotes* but leaves the **local checkout on a stale worker branch** (the local repo
  showed 0.18.3 while both remotes were 0.19.1). FIX + make it the LAST step of every release:
  switch the local checkout to `main` and fast-sync to origin (`git switch main && git reset
  --hard origin/main`) so `git rev-parse HEAD == origin/main`.
- Quick check: `for r in origin wrfgpu; do git show $r/main:src/gpuwrf/__init__.py | grep __version__; done`
  + `git rev-parse --short HEAD origin/main` — all three must agree.
- The public `wrfgpu` tag points to the sanitized PUBLIC commit and `origin` to the private
  commit, but the VERSION STRING is identical on both. Never let them drift.

## Common failure modes
Overbroad scope; missing file ownership; weak acceptance criteria; accepting claims without artifacts; letting a decision live only in volatile context; runaway sub-agent apparatus; **local/remote version drift (see Version & remote sync).**
