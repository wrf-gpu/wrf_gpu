---
name: managing-sprints
description: Guides a manager agent when creating, assigning, gating, and closing evidence-driven sprints.
---

## When to use

Use when planning or running a sprint, assigning agents, enforcing gates, or closing work.

## Inputs required

Project constitution, current milestone, milestone plan, sprint objective, file ownership, validation commands, and required proof object.

## Workflow

1. Open a milestone with a manager-written milestone plan.
2. Get the milestone plan reviewed before implementation sprints start.
3. Create sprint folder from template.
4. Write a narrow sprint contract.
5. Assign owners and reviewers.
6. Confirm validation and performance gates.
7. Collect worker, tester, and reviewer reports.
8. Close with decision and memory-patch proposal.
9. **Record the sprint in the durable ledger (MANDATORY at every major-sprint
   close).** Append a one-row summary — sprint, branch/model, outcome, core
   decision — to `.agent/decisions/VERSION-SPRINT-LEDGER.md`, and capture any
   core decision (a theoretical limit, a what-can/can't-be-optimized verdict, a
   scope cut, a "closed wontfix with evidence", a roadmap change) into both the
   ledger and the manager's auto-memory. Expensive cross-model findings (e.g.
   Fable kernel-optimization sprints) MUST land in an authoritative in-repo doc
   (like `KERNEL-OPTIMIZATION-FINDINGS-FINAL.md`), not only in a review file or
   the volatile context — see "Core-decision capture" below.

## Core-decision capture (compaction survival) — manager responsibility

The manager's context WILL be compacted; review files and live context are NOT
durable. Keeping the project's hard-won decisions alive across compaction is part
of the manager job, not an afterthought. Rules:

- **Every major sprint** (and every milestone) gets a row in
  `.agent/decisions/VERSION-SPRINT-LEDGER.md` at close — branch/model, outcome,
  and the core decision in one line. The ledger is the in-repo, compaction-proof
  index of what was tried and decided per version.
- **Every core decision** — a measured theoretical limit, a what-can/can't-be-
  optimized verdict, a closed-wontfix-with-evidence, a scope cut, a roadmap
  change, an ADR outcome — is written to an authoritative in-repo doc
  (`.agent/decisions/*`) AND mirrored in the manager's auto-memory the same
  session it is made. Do not leave it only in a sub-agent review or in context.
- **Before any `/compact`** (or when context is at risk), run the maintaining-
  memory pre-compaction checklist: confirm the live anchor + the ledger + any
  new core-decision docs are written and committed. If a decision cost real
  tokens to reach (multi-sprint, cross-model), it MUST be durable before compact.
- Commit `.agent/decisions/*` changes promptly so they survive worktree teardown.

## Release publishing — dev folder vs public org repo (principal directive 2026-06-13)

THIS repo (`<USER_HOME>/src/wrf_gpu2`) is the **dev folder** — full of internal
agent/process cruft (`.agent`, `.codex`, `codex`, `cache`, `artifacts`, internal
`PROJECT_*`/`MORNING-REPORT*`/contract docs, the bulk of `proofs/`, oversized
fixtures). It is NOT what users see. Two remotes:
- **`origin`** = the private dev/backup remote (whole repo, cruft OK).
- **`wrfgpu`** = `git@github.com:wrf-gpu/wrf_gpu.git` — the **PUBLIC organizational
  project**. Each release is uploaded here as a **CLEAN, user-facing version**
  (src/tests/docs/README/RELEASE_NOTES/LICENSE + the docs-linked proof/evidence;
  NO dev cruft, NO >50 MB fixtures), and **the newest version becomes the default
  branch** (`main` HEAD) + the tagged release.

Release step (per version, after the honesty critic + tag): push the dev tag+main
to `origin` (backup), then dispatch a release-upload worker to produce the CLEAN
tree + push to `wrfgpu` main + tag + set newest=default. MANDATORY before any
public push: a **secrets/sensitive scan** (keys, credentials, private paths,
internal-tooling refs) — STOP if anything is found — and a package-import/collect
smoke so the strip didn't break it. Conservative: when unsure if user-facing, keep.

### Release-notes content standard (MANDATORY every version — principal 2026-06-13)

Every release's notes MUST show, with **tables AND plots**, the falsifiable picture:
- **What was tested** (the gate/test matrix — which modules/schemes/regions, which gates ran).
- **What works** (green, within its gate) — with the proving plots/tables.
- **What does NOT work** (every miss/carry/limitation), each tied to its evidence.
- **Disposition of everything that doesn't work** — each item is EITHER (a) a
  **diligent roadmap item to the next version** (named, with the lane), OR (b)
  **explicitly excluded from project scope** after careful analysis. A
  scope-exclusion is the ONE decision that REQUIRES principal feedback — reserve
  it for items that are so complex/impossible/huge-perf-sacrifice that scrapping
  them is the only realistic choice; bring the analysis, discuss, and on
  agreement **log it specifically in BOTH the release notes AND the project
  plan/roadmap**. Never silently drop a gap.

Nothing that doesn't work may be left undocumented or un-dispositioned. Plots
must PROVE the green claims (e.g. identity dashboards, stays-green/non-escalating
panels), not just assert them.

### Feedback model (sharpened, principal 2026-06-13)
Drive autonomously; do NOT stop to ask for routine decisions. The ONE case that
needs principal feedback is a **scope-EXCLUSION** (above). Everything else —
sequencing, dispatch, ship-timing, design — is the manager's call.

### Ship bar (from v0.16 on)
Do NOT ship a version until **everything in its scope is green AND proven (plots)
to stay green**, AND perf is **proven about as fast as realistically possible**
for what landed. Keep driving until that holds; then start the next version. (For
v0.16: test ideally ALL modules/schemes, all green, stays-green plots, fast-proven.)

## Agent dispatch mechanics

Match model+effort to task type (principal effort-tiers): **core/correctness-critical** (dycore, physics, coupling, proof generation) → **Opus 4.8 max** in-process `Agent`; **debugging / writing / review / harness** → **Opus 4.8 xhigh** or **GPT-5.5 xhigh**; go **parallel** for independent, file-disjoint work. ONE GPU job at a time (single GPU); fan out only non-GPU work in parallel.

- **GPU serialization is MANDATORY via the `locking-gpu` skill.** Every dispatch that may touch the GPU must name the `locking-gpu` skill and hand the worker the exact `scripts/with_gpu_lock.sh --label <name> -- <cmd>` invocation; the worker does CPU work first and the flock queues its GPU run behind any sibling. Two GPU workers may run in parallel ONLY because the lock serializes their actual GPU commands — never dispatch GPU work that bypasses the wrapper. Parallel agents share the lock by common knowledge of this one skill.

- **MODEL POLICY OVERRIDE (principal directive 2026-06-11, sits ABOVE every older model assignment in this file while in effect; when they conflict, this block wins until the principal lifts the GPT-5.5 freeze):**
  - **Opus 4.8 is the default model** for manager, frontrunner/worker, reviewer, critic, verifier, and debugger roles. Default effort xhigh; use **max** for core/correctness-critical implementation and proof generation.
  - **MODEL ROSTER (principal 2026-06-13): Fable is OFFLINE for the next days (US-gov decision) — do NOT attempt Fable.** The two available models are **Opus 4.8 max** and **GPT-5.5 xhigh**. Use them by TASK CLASS:
    - **Kernel-speed-relevant steps + complex debug/kernel-code → Opus max ↔ GPT-5.5 ALTERNATING + ADVERSARIAL, iterating to consensus.** Each MUST assume the other's code is *both* still optimizable AND contains errors, and may only accept the other's work as "perfect" when EVERY attempt to improve or break it has failed. Alternate Opus↔GPT (one implements/revises, the other adversarially audits errors + architecture + speed + memory) **until they reach consensus on the best-possible solution.** Brief BOTH to be brutally critical on speed AND memory (every per-step line × ~1M steps). **The manager is the referee** when they disagree — adjudicate with evidence (oracles, gates, op/alloc counts), don't just average.
    - **Feature implementations, benchmarks, validation, coverage, docs, grooming → ONE Opus max worker is ALWAYS enough, no cross-control needed.** Do NOT spend GPT on these.
  - GPT-5.5 is token-limited (near a budget reset) — its budget goes to the kernel adversarial loop, not routine work. Launch GPT-5.5 xhigh as an **interactive codex TUI in a fresh tmux window** (session `0`; windows 0–1 are the principal's, use an explicit free index e.g. `0:4`): `codex --dangerously-bypass-approvals-and-sandbox -m gpt-5.5 -c model_reasoning_effort=xhigh "$(cat /tmp/<prompt>.txt)"` (the `--dangerously...` substring can trip the Bash classifier — send via `tmux send-keys`; have the agent write its deliverable to an absolute path + print a unique DONE marker; detect via file + `capture-pane`, then `kill-window`).
  - **BRUTAL speed+memory critique on ALL kernel-level work (principal 2026-06-13).** Every agent that touches a per-step/per-substep kernel MUST be briefed to be ruthlessly critical of speed AND memory: every wasted op/array/allocation there is called ~a million times per simulation and can have drastic impact. Kernel solutions get an explicit speed+memory audit (op count, allocations, dtype, fusion) before acceptance — Opus implements, GPT (or a second Opus) adversarially audits.
  - **Fable 5 is the heavy-hitter for extremely complex debugging.** Default escalation ladder for a hard bug: (1) **Opus 4.8 xhigh first** — narrow the bug and make the first fix attempt; (2) hand to **Fable 5 high** if Opus does not close it; (3) **Fable 5 xhigh ONLY** for kernel-level persistent bugs that have already evaded several Opus and Fable-5-high rounds. Do not open a hard bug with Fable; do not reflexively reach for Fable xhigh. **Fable 5 medium** is the right tier for a bounded, already-narrowed bug the principal hands directly (e.g. a named residual lane with a built oracle).
  - **Core complex kernel tasks JUSTIFY Fable directly (principal 2026-06-13).** The "Opus-first" rule above is for *bugs* and bounded tasks. A **core, highly-complex, multi-step KERNEL task with a runtime-validation requirement** (e.g. an algorithmic kernel rewrite like BouLac O(nz²)→O(nz), a column-tiling of a transient, a faithful kernel restructuring that must pass a tiered/runtime gate) easily justifies **Fable (xhigh) as the FIRST dispatch** — it is not "opening a bug with Fable." The manager's call between two valid patterns: (a) dispatch **Fable xhigh end-to-end** straight away, or (b) send an **Opus "probe" first** to assess complexity + take a first shot, and delegate to Fable if it stalls. Either is fine; pick by how well-understood the task is. (BouLac was sent to Opus; Fable would have been equally/more appropriate — logged as a calibration of this threshold.)
  - **Fable 5 max** is reserved for principal-designated large analysis/optimization sprints (e.g. the v0.15 kernel optimization-explorer), authorized directly by the principal.
  - This override updates, but does not delete, the older GPT-centric dispatch table and Fable scarcity-ladder wording below — those describe the prior regime and resume only if the principal re-enables GPT-5.5 and rescinds this block.

- **Cross-model debug cadence:** for a complex correctness bug, after two focused GPT/debug sprints on the same problem fail to prove a fix or leave the conclusion methodologically uncertain, dispatch one Opus xhigh critic/debugger to challenge the method, hypotheses, evidence chain, performance implications, and candidate bug itself before the manager commits to the next conclusion. This cadence is principal-confirmed after the 2026-06-09 live-nest/base-source critique: Opus use should be more frequent at these proof gaps, but still targeted escalation rather than routine double-agenting.
- **Debug-tooling and wall-clock check:** at every planning step for a hard
  runtime/kernel-level bug, explicitly ask whether the team is using the right
  method and whether the current plan is the fastest rigorous wall-clock path.
  Runtime bugs can become expensive when each hypothesis needs a slow
  reproduction. It is often faster and cheaper to send one worker in parallel or
  serially to prove/refute a hypothesis, build a focused harness, savepoint
  emitter, comparator, schema freezer, or visualization, than to keep narrowing
  the bug by slow full-runtime steps. Treat one agent sprint spent building a
  valuable debug tool as cheap if it reduces the next 5-10 proof loops, lowers
  false-assumption probability, or makes the result more falsifiable. Prefer
  expert-style debugging methods that minimize number of steps to the target:
  isolate state boundaries, freeze schemas, create minimal reproducer/savepoint
  loops, compare exact oracles, and parallelize independent hypothesis tests
  without colliding on GPU or source ownership.
- **No manager-hypothesis anchoring for debug workers:** when dispatching a
  bug-hunt worker, provide the complete compact evidence chain, current status,
  prior fixes, failed hypotheses, proof artifacts, and any leading candidate
  roots, but do not frame the manager's current hypothesis as the required
  conclusion. The worker must independently build a ranked hypothesis ledger,
  decide what evidence supports or falsifies each candidate, and is explicitly
  allowed to reject the manager's suspected root if the proof points elsewhere.
  A suspected lane such as "advance_w/phi" is a candidate to test, not an
  assumption to preserve. The endpoint remains whole-task: find and fix the
  root if local and provable, or return the strongest falsifiable narrowing and
  next proof loop.
- **Opus worker/frontrunner:** in-process `Agent` tool (`subagent_type: general-purpose`, `model: opus`). **DISPATCH WITH `run_in_background: true` — STAY RECEPTIVE (principal directive 2026-06-01).** A foreground `Agent` call BLOCKS the entire manager turn until it returns; a 35-min foreground diagnosis agent locked out both the principal's messages and a finished GPT critic. Background agents auto-notify on completion via task-notification, so the manager stays free for the principal + other agents in between. **NEVER sit in a blocking `sleep`/poll loop** to watch an agent — dispatch, yield, react to the notification; do at most a single one-shot status check, never a waiting loop. Manager reviews diff, runs gates, commits/merges.
- **tmux hygiene before dispatch:** before launching new tmux agents, close completed/no-longer-needed worker windows from prior sprints so the shared tmux session remains clean. Do not close active workers, the manager pane, or principal-owned panes.
- **Manager polling cadence / no micromanagement (principal directive 2026-06-11):** after dispatching a worker, set an expected wall-clock duration and poll at a cadence that preserves manager context. Default to roughly **15-minute checks** for normal debug/analysis workers; use longer intervals for large end-to-end tasks and shorter checks only for jobs expected to finish in minutes, GPU-lock handoffs, user-requested status, suspected crashes, or hard external deadlines. Do not keep rereading tmux transcripts every few minutes. Workers should receive whole endpoint-defined assignments and return a proof object, verified outcome, source diff if any, excluded hypotheses, and their own reasoning if they fail. The manager integrates results, verifies artifacts, and decides the next sprint; the manager should not drip-feed hypotheses or steer the worker midstream unless a material new fact, safety issue, or resource conflict changes the task.
- **GPT-5.5 critic/debugger (codex):** launch as an **INTERACTIVE codex TUI session in a tmux window** so the principal can attach (`tmux attach`, Ctrl-b <n>) and watch/interject — principal directive 2026-06-01. **`tmux new-window -t <session>:<EXPLICIT-FREE-INDEX>`** (e.g. `-t 0:5`) then `tmux send-keys -t 0:5 'codex -s workspace-write -a never -m gpt-5.5 -c model_reasoning_effort=xhigh "$(cat /tmp/<prompt>.txt)"' Enter`. **GOTCHAS (both bit us 2026-06-01):** (1) `tmux new-window -t 0` means "create AT index 0" → fails "index 0 in use" and the follow-up `send-keys -t 0:` misroutes into the MANAGER's own pane — ALWAYS give an explicit free window index. (2) `--full-auto` is REMOVED in the current codex CLI (`error: unexpected argument '--full-auto'`) — use `-s workspace-write -a never` (sandboxed, auto-progress, no prompts, won't trip the manager's Bash classifier) or `--dangerously-bypass-approvals-and-sandbox` (full access; the principal's own pattern, but the dangerous substring may trip the classifier when sent via Bash). NOT headless `codex exec >log`. The TUI isn't file-logged, so instruct the agent to write its deliverable to an absolute main-repo path + print a unique DONE marker (e.g. `GPT <TOPIC> DONE`); detect via that file + `tmux capture-pane`, then `kill-window`. Completion messages to the manager pane must use delayed repeated Enter presses, for example `tmux send-keys -t 0:2 '<DONE MARKER>' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter`, because a single Enter can leave text staged in the Codex TUI. See memory [[Launch all agents in the same tmux session, close their windows when done]].
- **Liveness before re-dispatch:** an in-process Agent's transcript can LAG (look stale 30–90 min) while alive — verify death by PID (`ps -p <pid>`) + no child GPU procs, or you spawn a duplicate that races the branch + GPU.
- **Long GPU runs:** detach (`systemd-run --user --scope` / `nohup setsid`) + commit each proof immediately. The box hibernates; a CUDA context does NOT survive suspend → kill+rerun anything that spanned one. Kill orphan model GPU procs before each launch.
- **Worktree isolation caveat (bit us 2026-06-01):** `isolation: "worktree"` may branch from a STALE commit, not the current HEAD — a nesting agent's worktree came up at an M5-era commit missing recent fixes. Tell worktree agents to verify their base (`git log -1`) and `checkout -b <fresh> <current-tip>` if stale (never hard-reset — auto-denied). Worktree agents commit on their own branch; the manager merges after review + after any GPU sibling frees the branch/index (don't merge into a branch another agent is actively committing to).
- **GPU hand-off between agents:** an agent that arms a "GPU-free monitor" to auto-run its gates will GRAB the GPU the instant it drops free — which can be a GAP BETWEEN a sibling's multi-run sequence, not the sibling's true end. Risk: collision/box-crash. The manager must not dispatch a competing GPU job into that window, and should verify GPU sanity when the holding agent's completion notification arrives.
- **Fable/Mythos heavy-problem lane (principal directive 2026-06-09):** Fable
  (Mythos, tmux `0:1`) is a scarce high-end debug resource. Conserve its tokens:
  do not use it for routine polling, proof grooming, simple instrumentation,
  standard validation triage, or issues likely solvable by one focused GPT 5.5
  sprint. For validation failures, first send GPT 5.5 workers to collect,
  localize, and attempt direct fixes when feasible. Escalate only the unresolved
  hard core to Fable/Mythos, and frame it as one whole endpoint-defined
  assignment, not narrow micro-prompts. The manager remains manager: write the
  contract, freeze file/GPU locks, require proof objects, review the diff, run
  gates, merge or reject, and continue the milestone. Before sending each new
  Fable/Mythos sprint after a completion or context risk, first send `/compact`
  to `tmux 0:1`, wait about two minutes for the TUI to finish compaction and
  return to a prompt, then send the full assignment and press Enter. Use delayed
  repeated Enter presses when needed because the TUI can leave text staged.
  When a GPT worker narrows a hard task but does not solve it, the default Fable
  escalation is the **entire remaining task** with all current evidence and the
  release endpoint, not the next tiny diagnostic. The requested Fable endpoint is
  "roadmap checkbox done": fix the blocker and prove it with the release gate, or
  produce an exact WRF-anchored proof that the remaining blocker is impossible or
  strictly outside the assigned scope. Do not spend Fable tokens on manager
  status, incremental hypothesis asks, or partial micro-runs unless the principal
  explicitly redirects.
- **Fable medium/high debug lane (principal directive 2026-06-10):** Fable is
  also allowed, and encouraged, for *medium-hard* debug tasks when the failure is
  already bounded enough that one or two focused proof/debug runs should settle
  it, but it has become awkward, cross-module, or token-expensive for routine
  GPT work. Use a fresh tmux window for these jobs, normally
  `claude --model fable --effort medium|high --permission-mode auto`, and give a
  sprint contract with a whole, manager-actionable endpoint: prove the bug class,
  fix it if local and safe, or return the exact blocker/proceed signal. This
  lane is appropriate for targeted validation residuals, provenance/root-cause
  checks, or compact analysis workers. It is not for routine polling, simple
  report formatting, or long-running GPU gates. Keep **Fable/Mythos xhigh/max**
  reserved for truly hard kernel-level failures, many burned debug attempts,
  high-risk architecture/code reviews, or tasks the medium/high lane cannot
  close. When reusing the scarce `0:1` Mythos session, compact first; when using
  a fresh medium/high tmux window, a new session is preferred and no legacy
  context should be assumed.
- **Fable xhigh scarcity ladder (principal directive 2026-06-11):** after a
  Fable/Mythos xhigh sprint consumes a large context budget, do **not** launch a
  second xhigh run reflexively. If the bug remains open after that sprint,
  continue first with GPT-5.5 xhigh workers: have them verify/reject the Fable
  diff, narrow the residual, and attempt non-destructive or local fixes. If GPT
  stalls, try Fable **medium/high** as the next scarce-model tier with one
  whole endpoint-defined assignment. Reserve another Fable/Mythos **xhigh** run
  for roughly ten inconclusive GPT-5.5 xhigh / Fable-high follow-up sprints, or
  for a clearly documented exceptional case where the manager can justify that
  cheaper tiers are very unlikely to close the kernel-level blocker. This
  preserves Fable xhigh for genuinely hard fresh-context resets, not ordinary
  continuation.
- Manager stays manager: re-dispatch dead agents; don't hand-debug.

## Long-roadmap drift prevention

For long, correctness-critical roadmaps such as v0.14, prevent manager drift with
a periodic Opus 4.8 xhigh **management review**:

- Dispatch one Opus 4.8 xhigh management reviewer after every 15 closed sprints
  on the active milestone, and sooner if the roadmap direction changes, the
  proof chain becomes hard to summarize, or the manager considers changing the
  milestone goal.
- The reviewer reads only the compact current handoff/roadmap, the last 15
  sprint closeouts/reviews/proof summaries, and the manager's current
  conclusions. Do not ask it to reread broad source trees unless it identifies a
  specific gap.
- The review goal is drift control, not pair programming: challenge whether the
  manager is still taking the most efficient, highest-leverage, validated path
  to the milestone goal; identify waste, stale assumptions, missing gates,
  under-parallelization, unsafe parallelization, over-narrow or over-broad
  sprints, and whether the current debugging tools are the right ones for the
  problem.
- The reviewer must ask top-level whether more runtime chasing is still the
  cheapest path. If a focused debug tool, savepoint/comparison harness, schema,
  or visualization would make the next proof loop faster and more reliable, the
  reviewer should recommend that tooling sprint explicitly. The reviewer should
  also challenge whether a parallel or serial worker could cheaply prove/refute
  a key hypothesis while the main lane continues, and whether the plan matches
  expert kernel/runtime debugging practice rather than incremental log-chasing.
- Output must be context-sparing: maximum one short verdict paragraph, one
  ranked table of at most eight findings, one "next 3 sprints" recommendation
  list, and one explicit yes/no on whether the current goal should change.
- v0.14 goal changes are not allowed merely because the path is hard or long.
  A v0.14 goal change is allowed only if Opus 4.8 xhigh explicitly agrees in a
  management review that the current goal is technically impossible or no longer
  the smartest useful target under the latest evidence, and the manager records
  the evidence-backed replacement goal in the roadmap.
- Re-anchor major decisions to the project goal: build a WRF-faithful-enough,
  GPU-optimized, near compute- and memory-optimal, scalable GPU rewrite, not a
  station-score workaround or CPU-WRF wrapper.

Reusable Opus management-review prompt:

```text
You are Opus 4.8 xhigh, independent management reviewer for wrf_gpu2 v0.14.
Goal: prevent roadmap drift. The project goal is a WRF-faithful-enough,
GPU-optimized, near compute- and memory-optimal, scalable GPU rewrite.

Read only:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md
- .agent/decisions/V0140-VALIDATION-PLAN.md
- the last 15 sprint folders' sprint-contract.md, manager-closeout.md,
  memory-patch.md, and linked proof/review summaries

Critique the manager's current 0.14 roadmap, conclusions, proof chain,
parallelization, sprint sizing, next-sprint plan, and debug tooling. Decide
whether the manager is still on the fastest rigorous wall-clock path to the
goal. At top level, answer whether we are using the right tools and methods:
should the next sprint build a focused harness/savepoint/comparator/schema/
visualization, or dispatch a parallel/serial worker to prove/refute a key
hypothesis, instead of chasing another slow runtime reproduction? Evaluate the
method like an expert kernel/runtime debugger: minimize steps to the target,
minimize false-assumption probability, prefer minimal reproducible proof loops,
freeze schemas/boundaries, and avoid expensive full-run iteration unless it is
actually the fastest rigorous path. Do not propose a goal change unless the
current goal is technically impossible or clearly no longer the smartest useful
target under the latest evidence.

Output exactly:
1. Verdict paragraph, max 120 words.
2. Ranked findings table, max 8 rows: severity, issue, evidence, fix.
3. Next 3 sprints, max 3 bullets, each with objective and proof gate.
4. Goal-change gate: "NO_GOAL_CHANGE" or "GOAL_CHANGE_RECOMMENDED: <why>".
5. Method/tooling verdict: "RIGHT_TOOLS_FASTEST_WALL_CLOCK" or
   "CHANGE_METHOD: <tool/worker/hypothesis path and why>".
6. Context-sparing handoff: max 10 bullets the manager should remember.
```

## Hard rules

- No implementation without a sprint contract.
- No first implementation sprint in a milestone without reviewed milestone plan.
- No done claim without proof object.
- No scope expansion without approval.

## Deliverables

Milestone plan, sprint contract, assignments, closeout, merge recommendation, memory patch.

## Validation

Run `python scripts/close_sprint.py <sprint-folder>` at closeout.

## Common failure modes

Overbroad scope, missing file ownership, weak acceptance criteria, and accepting claims without artifacts.
