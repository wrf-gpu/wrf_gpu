# Sprint Tracker — Live Dashboard

Manager-maintained. Updated every watchman tick. **30-min cadence per user 2026-05-21 11:10**.

## Currently in flight (2 Opus reviewers, both using upgraded WATCHDOG launcher)

| Window | Sprint | AI | Phase | Started | Notes |
|---|---|---|---|---|---|
| `reviewer-s3y` | M5-S3.y RRTMG setcoef+taumol+Planck | claude-opus-4-7 xhigh | reviewer | 11:38 | Worker self-flagged "do not accept" + 1.31MB HLO budget burst; reviewer to confirm REJECT-bounded-as-M5-S3.z |
| `reviewer-m6s1` | M6-S1 coupled interface freeze | claude-opus-4-7 xhigh | reviewer | 11:39 | Worker self-PASS all 7 AC + 0-byte transfer audit + ADR-010; reviewer to verify + check boundary-forcing-handle gap per critic amendment #3 |

**Watchdog now installed** (`.agent/references/dispatching-agents-pattern.md` updated `fd5c214`): every launcher polls report file + force-fires AGENT REPORT after 60s stability. No more stuck-at-/exit incidents.

## Closed this tick (P4 manager-integrated; no Opus needed per "manager+1 gpt" rule)

| Sprint | Verdict | Action |
|---|---|---|
| **P4 M6 plan consensus** | codex critic **RATIFY-WITH-AMENDMENTS** (8m) | Manager integrated 10 amendments → `manager-amendments.md`. Sequencing + proof schemas now load-bearing. |

## Sprint contracts ready to dispatch (queued for after M6-S1 Opus closes)

| Sprint | Wall | Owns | Dependency |
|---|---|---|---|
| **M5-S3.z RRTMG intermediate-oracle extraction** (NEW per M5-S3.y worker §3) | 16-24h | `src/gpuwrf/physics/rrtmg_*` + harness extensions + `validation/rrtmg_intermediate_oracles.py` | M5-S3.y Opus verdict (chooses between REJECT-bounded vs PARTIAL-ACCEPT vs REJECT-revert) |
| **M6-S2a Gen2 backfill accessor + d02 boundary replay** (NEW per critic amendment #2) | 12-18h | `src/gpuwrf/io/**` (NEW module) + shared validation I/O + CPU denominator + proof-object schemas + ADR-011 | M6-S1 Opus ACCEPT |

These two run in parallel after M6-S1 closes (disjoint file ownership: M5-S3.z owns physics/rrtmg, M6-S2a owns io/).

## Next swarm planned (after M5-S3.z + M6-S2a close)

| Sprint | Wall | Parallel-with |
|---|---|---|
| M6-S2 coupled forecast driver (1h → 6h → 24h d02 with boundary replay) | 24-36h | M6-S3 |
| M6-S3 surface layer + bounded Noah-MP minimum | 30-48h | M6-S2 |

## After that swarm (validation phase, 4-way parallel)

| Sprint | Wall |
|---|---|
| M6-S4 Tier-2 coupled invariants (external/cross-implementation oracle, no self-consistency) | 16-24h |
| M6-S5 ADR-007 4× verdict (fair Gen2 CPU denominator + FAIL fallback ladder) | 12-20h |
| M6-S6 Tier-3 TSC1.0 (controlled dt-refinement reduced case) | 18-30h |
| M6-S7 Tier-4 probtest prototype (stratified by land/sea/elevation) | 18-30h |

## Final M6 close

| Sprint | Wall |
|---|---|
| M6-S8 operational Gen2 + closeout (CPU-vs-obs binding gate, GREEN/PARTIAL/BLOCKED/FAIL statuses) | 24-36h |

## Big-picture path to PROJECT_CONSTITUTION end goal (Canary 3km/1km daily forecast)

```
NOW: 2 Opus reviewers in flight
  ↓
~30-60 min: M6-S1 Opus ACCEPT → dispatch M5-S3.z + M6-S2a parallel
  ↓
~16-24h: both close → dispatch M6-S2 + M6-S3 parallel
  ↓
~24-48h: both close → dispatch M6-S4 + M6-S5 + M6-S6 + M6-S7 parallel (4-way)
  ↓
~18-30h: validation closes → dispatch M6-S8 closeout
  ↓
M6 GREEN → M7 Canary operational v0 (3km then 1km pipeline, I/O, restart, daily-run, ops verification)
  ↓
M7 GREEN → M8 forkable release (docs, packaging, public review)
```

**Calendar estimate**: M6 close 7-10 days from now; M7 + M8 add 14-21 days = end-goal landing in **~4-5 weeks**.

## File-ownership snapshot (post-M6-S1)

- `src/gpuwrf/contracts/**` (state.py, precision.py): FROZEN by M6-S1
- `src/gpuwrf/coupling/**`: M6-S1 owns interfaces; M6-S2 owns driver
- `src/gpuwrf/io/**`: NEW, M6-S2a owns (single owner for shared validation I/O)
- `src/gpuwrf/physics/thompson_*, mynn_*`: M5-S1.y/S2.x closed, frozen
- `src/gpuwrf/physics/rrtmg_*`: M5-S3.z reopens for intermediate-oracle work
- `src/gpuwrf/physics/surface_layer.py, noah_mp.py`: NEW, M6-S3 owns
- `src/gpuwrf/dynamics/**`: M4 frozen
- `src/gpuwrf/validation/tier{2,3,4}_coupled.py`: M6-S4/S6/S7 own respectively
- `scripts/m6_*.py`: per-sprint ownership in ADR-010

## Watchman policy

- 30-min cadence per user directive
- On each tick: check tmux panes, dispatch reviewers/workers per the planned sequence
- Next tick: ~12:13

## Recent ticks

- 2026-05-21 11:11-11:12 — dispatched next swarm: M5-S3.y + M6-S1 + M6-plan-critic (3 codex parallel)
- 2026-05-21 11:42 — watchman tick #4 (user-triggered):
  - All 3 codex agents finished but stuck at /exit (recurring auto-notify bug)
  - **Fixed**: upgraded launcher pattern with WATCHDOG; skill file updated
  - M5-S3.y worker self-rejected (budget burst) → Opus reviewer dispatched (with watchdog)
  - M6-S1 worker self-PASS → Opus reviewer dispatched (with watchdog)
  - M6 plan critic RATIFY-WITH-AMENDMENTS → manager integrated 10 amendments
  - **New stubs**: M5-S3.z (intermediate-oracle extraction) + M6-S2a (Gen2 accessor + boundary replay + shared validation I/O)
- Next: 30-min tick at ~12:13

## Rate-limit watch

After this tick: 2 Opus + 0 codex active. Opus quota separate from gpt quota. Plenty of headroom for next dispatch round.
