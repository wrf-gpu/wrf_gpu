# Sprint Tracker — Live Dashboard

Manager-maintained. 30-min cadence. **Per user 2026-05-21 12:35: update this every time new agents dispatch so user can watch progress live.**

## Currently in flight (3 agents — 2 Opus + 1 codex)

| Window | Sprint | AI | Phase | Started | Wall budget | File ownership |
|---|---|---|---|---|---|---|
| `reviewer-m6s2a` | M6-S2a Gen2 accessor + d02 boundary replay + shared I/O | Opus 4.7 xhigh | reviewer (multi-Enter watchdog) | 12:34 | ~15-25min | reviewer-report.md |
| `reviewer-s3z` | M5-S3.z RRTMG intermediate-oracles | Opus 4.7 xhigh | reviewer (multi-Enter watchdog) | 12:34 | ~15-25min | reviewer-report.md |
| `scout-m7plan` | M7 Canary operational v0 plan scout | codex gpt-5.5 xhigh | scout (multi-Enter watchdog) | 12:36 | 30-60min | NEW `m7-milestone-plan.md` (read-only otherwise) |

**All 3 disjoint**: opus reviewers read existing files; m7plan scout writes a single new plan doc. No risk of merge conflicts.

## Just dispatched (this swarm)

- Three Opus reviewer prompts written + multi-Enter launchers + dispatched (m6s2a + s3z)
- M7 scout prompt written + dispatched (parallel productive use of idle gpt capacity per user directive)
- Two M6 sprint contracts pre-staged (M6-S2 + M6-S3) ready for dispatch the moment M6-S2a Opus accepts

## M5 sprint table

| Sprint | Worker attempts | Reviewer | Verdict | Merge | Status |
|---|---|---|---|---|---|
| M5-S0 scout | codex 1 | codex crit | ADR-005 ratified | `09a3738` | ✓ CLOSED |
| M5-S1 Thompson | codex 6 | Opus + Gemini | ACCEPT (CGG11 caught by Gemini) | `d768194` `00e7ee8` | ✓ CLOSED |
| M5-S1.x Thompson lookup tables | codex 1 | manager | partial; debt → M5-S1.y | `fe959d2` | ✓ CLOSED (partial) |
| ADR-007 precision policy | codex 1 | Gemini (pre-quota) | ACCEPT | `445c49f` `6c9df22` | ✓ CLOSED |
| M5-S2 MYNN attempt-1 | codex 1 | retroactive Opus | REJECTED | rescinded | ✗ REJECTED |
| M5-S2 MYNN attempt-2 | codex 1 | Opus | ACCEPT (real WRF-EDMF nm-verified) | `fe64e8f` | ✓ CLOSED |
| M5-S2.x MYNN follow-ups | codex 1 | Opus | ACCEPT (independent budget probe) | `9625d73` | ✓ CLOSED |
| M5-S3 RRTMG attempt-1 | codex 1 | Opus | REJECT (synthetic tables) | rescinded | ✗ |
| M5-S3 RRTMG attempt-2 | codex 1 | Opus | REJECT-bounded (clip-pinning) | rescinded | ✗ |
| M5-S3 RRTMG attempt-3 | codex 1 | Opus | ACCEPT-AS-GROUNDWORK | `b1a3102` | ✓ CLOSED |
| M5-S1.y Thompson HLO + residuals | codex 1 | Opus | ACCEPT-AS-GRAY-ZONE-CHECKPOINT | `0bd1fd2` | ✓ CLOSED |
| M5-S3.x RRTMG Eddington transfer-solver | codex 1 | Opus | ACCEPT-AS-GROUNDWORK-PHASE-2 | merged | ✓ CLOSED |
| M5-S3.y RRTMG setcoef+taumol+Planck | codex 1 | Opus | PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3 | merged | ✓ CLOSED (4 permanent artifacts preserved) |
| **M5-S3.z RRTMG intermediate-oracles** | codex 1 (done) | **Opus IN FLIGHT** | TBD | pending | 🟡 reviewer-s3z |
| M5-S3.zz follow-up (SW or LW or harness) | queued — scope set by S3.z Opus | — | — | — | ⚪ queued |
| M5-S1.z Thompson collision tables | queued — only if M6 RMSE flags microphysics | — | — | — | ⚪ optional |

## M6 sprint table

| Sprint | Status | Wall | Notes |
|---|---|---|---|
| M6 plan scout | ✓ CLOSED | 9m | `3392d04` codex scout |
| M6 plan critic | ✓ CLOSED (RATIFY-WITH-AMENDMENTS) | 8m | codex critic; manager integrated 10 amendments |
| M6-S1 coupled interface freeze | ✓ CLOSED (ACCEPT-WITH-MINOR-FOLLOWUPS, 12P/5F/0R) | 22m+9m | `2c6748a`; 5 prereqs bundled into M6-S2 contract |
| **M6-S2a Gen2 accessor + boundary replay + shared I/O** | codex done; **Opus IN FLIGHT** | 12-18h+~20m | 🟡 reviewer-m6s2a — critical-path infrastructure |
| M6-S2 coupled forecast driver (1h→6h→24h on d02) | contract READY; blocked on M6-S2a Opus ACCEPT | 24-36h | Bundles all 5 M6-S1 prereqs |
| M6-S3 surface layer + bounded Noah-MP | contract READY; blocked on M6-S2 smoke | 30-48h | Manager recommends Option A (prescribed land state) |
| M6-S4 Tier-2 coupled invariants | queued; parallel after M6-S2 smoke | 16-24h | External oracle (no self-consistency) |
| M6-S5 ADR-007 4× verdict | queued; parallel after M6-S2 smoke | 12-20h | Uses M6-S2a CPU denominator (`-r4` caveat) |
| M6-S6 Tier-3 TSC1.0 | queued | 18-30h | Controlled dt-refinement |
| M6-S7 Tier-4 probtest prototype | queued | 18-30h | Stratified by land/sea/elevation |
| M6-S8 operational Gen2 + closeout | queued — serial final | 24-36h | CPU-vs-obs binding gate |

## M7 sprint table (pre-flight)

| Sprint | Status | Notes |
|---|---|---|
| **M7 plan scout** | **codex IN FLIGHT** (12:36) | 🟡 scout-m7plan — pre-stages M7 contracts |
| M7 plan critic | queued — after scout closes | Manager-pattern: scout → critic → manager amendments |
| M7-S0..Sn Canary operational v0 | queued — defined by M7 plan integration | 3km then 1km pipeline, daily-run, ops verification |

## M8 (queued, post-M7)

| Sprint | Status | Notes |
|---|---|---|
| M8 forkable release | queued | docs, packaging, public review |

## Path to PROJECT_CONSTITUTION end goal

```
NOW (3 agents) → M6-S2a + M5-S3.z Opus verdicts (+~20min) + M7 plan scout (+~45min)
  → +24-48h: M5-S3.zz + M6-S2 + M6-S3 parallel dispatch (3 codex)
  → +18-30h: M6-S4 + M6-S5 + M6-S6 + M6-S7 4-way parallel
  → +24-36h: M6-S8 closeout
  → M6 GREEN → M7 implementation (plan already prepared by scout)
  → M7 GREEN → M8 forkable release
```

**Calendar**: M6 close 5-8 days from now (faster than earlier estimate thanks to parallel M7 prep); end-goal landing **~3-4 weeks**.

## File-ownership snapshot

- `src/gpuwrf/contracts/state.py, precision.py`: FROZEN by M6-S1 (M6-S2 will add boundary leaves)
- `src/gpuwrf/coupling/physics_couplers.py`: FROZEN by M6-S1 (M6-S2 threads GridSpec)
- `src/gpuwrf/coupling/{driver.py, boundary_apply.py}`: NEW, M6-S2 owns
- `src/gpuwrf/io/**`: M6-S2a OWNS (in flight Opus review)
- `src/gpuwrf/physics/thompson_*, mynn_*`: M5-S1.y/S2.x CLOSED, frozen (M5-S1.z optional reopen)
- `src/gpuwrf/physics/rrtmg_*`: M5-S3.z in flight Opus; M5-S3.zz to reopen
- `src/gpuwrf/physics/{surface_layer,noah_mp}.py`: NEW, M6-S3 owns (queued)
- `src/gpuwrf/dynamics/**`: M4 frozen
- `src/gpuwrf/validation/{tier2,tier3,tier4}_coupled.py`: M6-S4/S6/S7 own (queued)

## Watchman policy

- 30-min cadence per user
- **Routine: update this table on EVERY new agent dispatch** so user has live visibility
- Next watchman ~13:05

## Recent ticks

- 2026-05-21 12:08-12:10 — watchman #5 (user-triggered): M5-S3.y + M6-S1 closed; M5-S3.z + M6-S2a dispatched; watchdog fix encoded
- 2026-05-21 12:30-12:35 — watchman #6 (user-triggered via 2 AGENT REPORTs): both workers reported (single Enter still unreliable — manager Enter'd manually); 2 Opus reviewers dispatched with multi-Enter watchdog; M6-S2 + M6-S3 contracts pre-staged
- 2026-05-21 12:36 — **M7 scout dispatched in parallel** (codex idle capacity used for forward motion); table-update routine encoded per user directive
- Next: 30-min tick at ~13:05

## Manager utility opportunities for "what else can happen in parallel?"

| Idle capacity opportunity | Status |
|---|---|
| M7 plan scout (next milestone planning) | ✓ DISPATCHED NOW |
| M5-S1.z Thompson collision tables (speculative) | Held — only if M6 RMSE flags |
| Skill-patch consolidation (formalize watchdog/multi-Enter/GROUNDWORK-PHASE-N patterns) | Held — useful but lower priority than M7 planning |
| Gemini bug-chase | Held — quota-conserved for M7 operational |
| pyproject.toml zarr add | Manager can do this directly; will batch with next commit |
