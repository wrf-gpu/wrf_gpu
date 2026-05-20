# Overnight Report — 2026-05-19 / 2026-05-20

Manager: Claude Opus 4.7 (1M context)
Period: ~2026-05-19 ~23:30 → 2026-05-20 ~07:30 (8 hours of autonomous work)
Working tree: `/home/enric/src/wrf_gpu2`; remote `origin/main`

## TL;DR

Closed **M4** (reduced dycore) and **M5-S0** (Thompson selected). Three ADRs finalized with cross-AI critical-reviews applied. M5-S1 (Thompson microphysics implementation) **still cycling** — attempt 2 in flight as of report write time; attempt 1 was rightly Rejected for a path-B tautology (worker compared JAX to its own NumPy re-implementation instead of WRF), attempt 2 must use Berry-Reinhardt/Srivastava-Coen WRF-faithful formulas with an independent fixture oracle.

**On `origin/main` you will find new since you went to sleep**:

| Commit | What |
|---|---|
| `9c045bc` | Merge M4-S1 — reduced split-explicit dycore (RK3 + 5H/3V advection + acoustic), debug-gated, tier-1/2/3 validated. M5 gate trips on 24 launches (reporting-only per ADR-001). |
| `880d353` | M4 closeout — reviewer Accept-with-required-fixes; 3 documented residual limits. |
| `09a3738` | Merge M5-S0 scout — ADR-005 Thompson selected as first physics, with 5 codex critical-review fixes applied. |
| `a674e5c` | M5-S1 sprint contract — Thompson microphysics column kernel. |
| `a1e5032` | ADR-003 dycore precision — fp64 lock through M5-S1 + Authorization Matrix + no perf-downcast without profiler (codex critical-review applied). |

## Milestones & ADRs

### M4 — Reduced Dycore (CLOSED)

- Single sprint M4-S1 across 2 worker attempts + 3 tester attempts + 2 reviewer attempts.
- Attempt-1 reviewer Rejected for 4 blockers (textbook RK3 scaling bug, tautological tier-1, no-op tier-2, dycore-bypassing tier-3) + 2 majors (1/3-complete velocity advection, HLO sibling not literally hand-stripped).
- Attempt-2 fixed all 7 fix-cycle ACs; tester (Claude Opus xhigh) Accept with 25 adversarial regression tests; reviewer (codex xhigh) Accept-with-required-fixes (3 documented limits, all using OR-amend-escape clauses).
- **Numbers worth remembering** (architectural baseline):
  - Zero post-init host/device transfers
  - 0-byte HLO debug-vs-stripped diff (constitutional debuggability gate held)
  - 0 temporary bytes per step (no hot-path allocations)
  - 24 kernel launches per dycore step — **trips M5 gate (≤10), per ADR-001 reporting-only**
  - Tier-3 observed order 4.65 through public `run()` API
  - **384 tests passing**
- **Three documented residual limits** carried forward to M5+ per `.agent/decisions/MILESTONE-M4-CLOSEOUT.md`:
  1. Debug snapshot is host-callback per-stage, NOT contracted JAX-side last-N ring. Production HLO is still zero-leak.
  2. Acoustic substep is reduced-proxy with placeholder constants. No manufactured sound-wave validation.
  3. Tier-2 mass evidence is `theta_total` surrogate, NOT WRF-canonical `mu`/density mass-continuity.

### M5-S0 — First Physics Suite Selection (CLOSED, ADR-005 ACCEPTED with critical-review applied)

- Decision: **Thompson 2008 microphysics, WRF `mp_physics=8` semantics**.
- Codex `gpt-5.5` xhigh critical-review on ADR-005 returned `Accept with required fixes` — 5 findings, ALL APPLIED:
  - Frozen Thompson target: WRF call boundary, 6 hydrometeor species + 2 number concentrations, sedimentation OUT of M5-S1 scope, Tier-1 fixture variables + tolerances.
  - Explicit "sequencing-not-operational-sufficiency" framing + MYNN-EDMF M5-S2 follow-on hook.
  - M2 column profile numbers labeled as hypothesis-prior, NOT readiness evidence.
  - Non-discretionary gray-zone gate rule (mandatory restructuring + 2nd profile + cross-model signoff + human-arbiter visibility).
  - Direct citation of `PROJECT_PLAN.md:176` Gen2 operational stack.
- **`origin/scout/codex/m5-s0-physics-scheme-selection`** + merged to main as `09a3738`.

### ADR-003 — Dycore Precision (ACCEPTED with codex critical-review applied)

- Codex critical-review: `Accept with required fixes` — 4 majors + 2 minors, ALL APPLIED.
- **Net effect**: `fp64 is the ONLY authorized production dycore precision through M5-S1`. All downcast statements demoted to experimental candidates with NOT-AUTHORIZED default. Added Authorization Matrix (per-field artifact paths + tolerances + profile gate + authorization outcome).
- M5 physics rows explicitly gated on Thompson fp64 frozen target passing first.
- Performance-downcast rule: no production downcast without launch + register + local-memory + occupancy + transfer audit + fp64-vs-candidate timing evidence.

### M5-S1 — Thompson Microphysics Column (IN PROGRESS — attempt 2)

**Attempt 1 outcome: REJECTED by codex reviewer (Decision: Reject)** for two blockers:
1. **Path-B Tier-1 fixture tautology** — worker's fixture generator and JAX kernel use the same compact analytic formulas, so Tier-1 became a JAX-vs-NumPy self-consistency check, not WRF Thompson parity.
2. **Compact analytic approximations** — worker used simplified time-relaxation rates for autoconv, freezing/melting, and vapor deposition instead of WRF's Berry-Reinhardt + Srivastava-Coen + particle-diameter formulas required by AC 2.2.

The cross-AI gate worked: tester (Claude Opus xhigh) independently surfaced the same tautology concern + verified the worker's claims; reviewer (codex xhigh) confirmed Reject with specific WRF source-line citations (`module_mp_thompson.F.pre:2242-2268`, `:3561-3636`, `:2709-2770`, `:4033-4142`).

**Attempt 2 dispatched** with amended contract (`fd50df0`):
- Replace fixture oracle with Path A (Fortran wrapper around compiled WRF Thompson driver) OR Path B-strict (line-by-line WRF transcription with citations) OR hybrid.
- Replace compact approximations with Berry-Reinhardt autoconv, Khairoutdinov-Kogan accretion, Srivastava-Coen rain evap, particle-diameter deposition, WRF mass/number balance constraints.
- **Anti-tautology guard**: tester + reviewer must verify ≥50% of fixture output is produced by formulas distinct from the JAX kernel.
- **Backstop**: if Path A AND Path B-strict both infeasible in attempt-2 time budget, worker files `BLOCKER-m5-s1-thompson-fixture.md` and manager opens dedicated M5-S0.5 Fortran-wrapper sub-sprint OR formally amends ADR-005 to authorize narrower scope (with re-dispatched ADR-005 critical-review).

Attempt 2 status at report write time: codex xhigh worker active on worker branch `worker/gpt/m5-s1-thompson-microphysics-column`, ~55 min in, in oracle-cascade final-validation phase. Worker has modified ADR-006, all M5 artifacts (tier1, tier2, profile, gate, HLO dumps, maintainability), and the fixture manifest + .npz — suggesting a substantial re-implementation against WRF formulas is in progress.

## Process Wins & Skills Updates (in queue)

Cross-AI gate caught real bugs at every milestone this overnight cycle. **`5 staged skill updates`** in `.agent/skills/*/SKILL.proposed.md` + `.agent/patches/2026-05-19-skill-updates-m2-m3-lessons.md` (validated `{"ok": true}` by `scripts/validate_memory_patch.py`) encode the M2/M3/M4 lessons:

1. **`writing-gpu-kernels`** — Python-scalar `static_argnames` rule (the `dt`-static lesson); pytree `__hash__ + __eq__` for JIT cache; debug-vs-stripped HLO byte-identity gate; eliminate cause not symptom.
2. **`validating-physics`** — analytic-fixture preference; operator-mismatch handling (match-or-sibling-not-coerce-tolerance); `pass: bool` machine-checkable contract.
3. **`conducting-blind-review`** — cross-AI verification axis; constitutional-gate independent reproduction; Allocation Audit recount; per-line attestation.
4. **`managing-sprints`** — `dispatch_role.sh` single-source-of-truth; per-milestone goal+oracle+runbook trio; hard rule "no manager commits during active worker"; wake-cadence discipline.
5. **`designing-gpu-state`** — pytree `__hash__ + __eq__`; `jax_enable_x64` at import; halo signature freeze ≠ multi-GPU drop-in guarantee.

Patch needs codex reviewer pass before merging. Dispatch is queued (task #24); held tonight to avoid concurrent-codex limit conflict with active M5-S1 worker. Will dispatch at M5-S1 close.

## Infrastructure improvements (committed)

- **`dispatch_role.sh` interactive mode** (commit `f4f1107`): drops `claude -p` / `codex exec` non-interactive; agents now run in real tmux REPL panes user can watch + inject keystrokes into. Logs captured via `tmux pipe-pane`. Prompt seeded via paste-buffer + Enter. **Validated 6× this overnight on real dispatches.**
- **`dispatch_role.sh` auto-detects active milestone goal file** (commit `6cc36d3`).
- **`scripts/check_m4_done.py`** (commit `6cc36d3`): single-command M4 oracle following M3 pattern.

## Lessons captured for skill-patch (tasks 25-27)

- **Oracle recursion is a major time-sink**: each `check_m<N>_done.py` recursively invokes `check_m<N-1>_done.py` etc, and each spawns a full `pytest -q`. A single M4 worker validation took ~40-50 min on the loaded 4-core system. **Fix proposal**: flatten oracles so `check_m4_done.py` does not invoke `check_m3_done.py`; require user/manager to run prior-milestone checks once at milestone entry.
- **Hibernate breaks long-running API-bound agents silently** (`claude -p` got stuck on dead socket for 90 min). Janitor heuristic added: "process alive AND log unchanged >15 min AND no new child processes → kill + redispatch".
- **Claude Code background-task watch-loops have a PID-recycle bug**: `until ! kill -0 $(pgrep -f X | head -1); sleep N; done` polls a captured PID; if PID is recycled to another live process, loop runs forever. Janitor sweeps and kills these.
- **Manager-spawned concurrent pytest contaminates worker validation**: when manager runs `check_m3_done.py` during active worker, both compete for GPU. Skill update §6.

## Tmux state when report was written

| Window | Agent | Sprint |
|---|---|---|
| 0 | manager (Claude Opus 4.7 1M) | watchman + reports |
| 1 | M5-S1 worker A2 (codex xhigh) | Thompson microphysics WRF-faithful attempt 2 |

## What to expect when you read this

If M5-S1 worker A2 completed before you woke, you'll see additional commits on `worker/gpt/m5-s1-thompson-microphysics-column` (`fd50df0` was the attempt-2 contract amendment; worker A2's actual implementation commit will follow). Look for `worker-report.md` timestamp after 07:00.

If tester+reviewer also completed, M5-S1 may already be merged to main with a `MILESTONE-M5-S1-CLOSEOUT.md`.

If only worker A2 finished and tester is in flight, tmux:1 will show Claude Opus xhigh tester.

If worker A2 hit the backstop and filed `BLOCKER-m5-s1-thompson-fixture.md`, manager will be running the documented backstop procedure (Fortran-wrapper sub-sprint OR ADR-005 narrower-scope amendment).

## Decisions made under your overnight autonomy directive

Per your 2026-05-19 23:xx delegation ("you are free to proceed to M5 WITHOUT my confirmation … if you are in doubt, ask gpt … no sprint limit even if persistent problem"):

- **ADR-003 ACCEPTED** by manager (codex critical-review applied; full fp64 lock through M5-S1).
- **ADR-005 ACCEPTED** by manager (codex critical-review applied; frozen Thompson target).
- **M4 closed** with 3 documented residual limits (reviewer Accept-with-required-fixes via OR-amend escape clauses; no actual physics regressions).
- **M5-S1 attempt 1 Rejected** (cross-AI gate); attempt 2 dispatched with explicit WRF-faithful guidance.

No constitutional gate was bypassed: ADR-003 and ADR-005 have user-post-hoc visibility flagged in their status lines for your reading. Both adopt fp64 retention as the binding posture (no production fp32 anywhere). If you disagree with any decision: respond explicitly and I will flip status back to "draft pending revision" and we re-do the cycle.

## Open decision queue for your morning attention

1. **M5-S1 outcome**: depending on where worker A2 is at your wake, may need your call on whether to accept narrower-scope ADR-005 amendment if backstop triggered.
2. **Skill-patch reviewer dispatch**: held until M5-S1 closes. Approve dispatch + then merge?
3. **Oracle recursion fix**: ready to dispatch but blocks on M5-S1 worker idle window. Approve at next idle?
4. **M5-S2 next sprint**: MYNN PBL is the ADR-005 follow-on hook per critical-review Major #2. Ready to dispatch when M5-S1 closes. Approve?

## Top-level project trajectory

You started this overnight session with M3 closed (state/grid skeleton). You end it with M4 closed (real dycore code) + M5-S0 decided (Thompson) + ADR-003 + ADR-005 finalized + M5-S1 cycling. That's **one full milestone closed + one decision-gate sprint closed + one implementation sprint mid-cycle** in 8 hours of autonomy, with cross-AI gate catching real bugs at every step.
