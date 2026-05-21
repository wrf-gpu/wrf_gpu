# Sprint Tracker — Live Dashboard

Manager-maintained. Updated every watchman tick. Source of truth for parallel-management state.

**Per user directive 2026-05-21 ~09:35**: M6 prologue full-steam-ahead; 3 parallel codex workers dispatched (gpt rate-limit cap untested but try).

## Currently in flight (3 parallel)

| Window | Sprint | Worker | Worktree | Branch | Wall-h budget | Launched |
|---|---|---|---|---|---|---|
| `worker-s1y` | M5-S1.y Thompson HLO + residuals | codex gpt-5.5 xhigh | `/tmp/wrf_gpu2_s1y` | `worker/codex/m5-s1y-thompson-hlo-and-residuals` | 4-10 | 2026-05-21 09:35 |
| `worker-s2x` | M5-S2.x MYNN follow-ups | codex gpt-5.5 xhigh | `/tmp/wrf_gpu2_s2x` | `worker/codex/m5-s2x-mynn-followups` | 2-6 | 2026-05-21 09:36 |
| `worker-s3x` | M5-S3.x RRTMG transfer-solver rewrite | codex gpt-5.5 xhigh | `/tmp/wrf_gpu2_s3x` | `worker/codex/m5-s3x-rrtmg-transfer-solver` | **8-16** | 2026-05-21 09:36 |

**Auto-notify**: all 3 dispatched with the canonical completion handler from `.agent/references/dispatching-agents-pattern.md` — each will tap-type `AGENT REPORT [worker / <sprint> / codex] exit=N report=PATH` to manager pane `1:0` on `/exit`. Window self-destructs after report.

**File ownership disjointness verified**:
- s1y: `src/gpuwrf/physics/thompson_*`, `scripts/wrf_thompson_*`
- s2x: `src/gpuwrf/physics/mynn_*`, `scripts/wrf_mynn_*`, `validation/tier2_mynn.py`
- s3x: `src/gpuwrf/physics/rrtmg_*`, `scripts/wrf_rrtmg_*`
- All three independently amend their own ADR (006, 008, 009)

**Sandbox note** (2026-05-21 09:33): manager's first dispatch attempt blocked by sandbox safety gate ("create unsafe agents"). User approved manually 09:35. Permanent rule still pending; future autonomous dispatches may re-trigger gate.

## On deck (queued)

| Sprint | Trigger | Notes |
|---|---|---|
| Opus reviewer for each of s1y / s2x / s3x | On worker `AGENT REPORT` | Mandatory per sprint-lifecycle hard rule |
| P4 M6 plan consensus (codex critical-review) | After ≥1 prologue worker frees gpt-5.5 quota | User flagged 3x gpt rate-limit uncertainty; hold for now |
| M6-S1 coupled interface freeze | After all 3 prologue sprints close + P4 ratified | Serial — blocks M6-S2/S3 |
| M6-S2 forecast driver | After M6-S1 closes | Serial — blocks S3-S8 |
| M6-S3 surface + Noah-MP minimum | After M6-S2 smoke passes | Parallelizable with S4/S5/S6/S7 |
| M6-S4 Tier-2 coupled invariants | After M6-S2 smoke | Parallelizable |
| M6-S5 ADR-007 4× verdict | After M6-S2 smoke | Parallelizable |
| M6-S6 Tier-3 TSC1.0 | After M6-S2 + M6-S4 | Parallelizable |
| M6-S7 Tier-4 probtest prototype | After M6-S1 | Parallelizable |
| M6-S8 operational Gen2 comparison + closeout | After all M6 sprints | Serial — final |

## Watchman policy

- Wakeup every 20-30 min during prologue
- On each tick: check `.worker-done` markers, capture tmux panes, dispatch Opus reviewer for any newly-finished worker, update this tracker
- On all 3 prologue workers done + Opus reviewers Accept: queue M6-S1 dispatch (single worker, no parallelism needed yet)

## Recent ticks

- 2026-05-21 09:30 — user approved M6 prologue plan, gave Go!; manager opened P1/P2/P3 contracts + worktrees + role-prompts; sandbox blocked first dispatch; user authorized 09:35
- 2026-05-21 09:35 — P1 worker-s1y dispatched, codex confirmed Working
- 2026-05-21 09:36 — P2 worker-s2x + P3 worker-s3x dispatched in parallel, both codex confirmed Working
- 2026-05-21 09:37 — tracker updated; watchman 25-min cadence scheduled
