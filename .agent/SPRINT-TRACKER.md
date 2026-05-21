# Sprint Tracker — Live Dashboard

Manager-maintained. Updated every watchman tick.

## Currently in flight (1)

| Window | Sprint | AI | Phase | Started | Status |
|---|---|---|---|---|---|
| `reviewer-s1y` | M5-S1.y Thompson | claude-opus-4-7 xhigh | reviewer | 10:43 | thinking/leavening; needs to decide ACCEPT-AS-GRAY-ZONE vs REJECT-bounded |

## Closed this tick (2)

| Sprint | Verdict | Merge | Closeout |
|---|---|---|---|
| **M5-S2.x MYNN follow-ups** | Opus **ACCEPT** | `dec3e8c` (worker `7f9f4f1` + reviewer `9625d73`) | `.agent/sprints/.../m5-s2x-mynn-followups/manager-closeout.md` |
| **M5-S3.x RRTMG transfer-solver** | Opus **ACCEPT-AS-GROUNDWORK-PHASE-2** | `0dad...` (worker `cbce2e5` + reviewer `e52857d`) | `.agent/sprints/.../m5-s3x-rrtmg-transfer-solver/manager-closeout.md` + new `m5-s3y-rrtmg-setcoef-taumol-planck/` stub |

## M6 prologue debt (running tally)

| Sprint | Status | Wall budget |
|---|---|---|
| M5-S1.y Thompson HLO + residuals | Opus reviewer in flight | 4-10h (delivered 43m) |
| M5-S2.x MYNN follow-ups | ✓ CLOSED ACCEPT | 2-6h (delivered 27m + 23m review) |
| M5-S3.x RRTMG transfer-solver | ✓ CLOSED GROUNDWORK-PHASE-2 | 8-16h (delivered 29m + 7m review) |
| **M5-S3.y RRTMG setcoef+taumol+Planck (NEW)** | STUB; awaits manager Eddington-vs-PIFM decision then dispatch | **16-32h** (largest M5 item) |

## On deck

| Sprint | Trigger | Notes |
|---|---|---|
| M5-S1.y manager closeout | After Opus reviewer verdict | Worker self-flagged GRAY-ZONE; reviewer to decide |
| M5-S3.y Eddington-vs-PIFM decision (manager+1 codex) | Now (manager) — option (a) patch local WRF kmodts=1 / option (b) retarget JAX to PIFM | Recommended (a); preserves M5-S3.x progress |
| M5-S3.y worker dispatch | After Eddington decision + P1 closure | Codex worker; 16-32h; file-disjoint can run parallel with M6-S1 prep |
| P4 M6 plan consensus (codex critical-review) | Now (codex quota freed; only P1 reviewer in flight, no codex active) | Can dispatch |
| M6-S1 coupled interface freeze | After M5-S1.y closes + ratification of M6 plan | Serial — blocks M6-S2/S3 |

## Watchman policy

- Next tick: 20 min (~11:05) — P1 reviewer expected to finish ~10:55-11:10
- Then: dispatch M5-S3.y worker + P4 codex critical-review in parallel
- After P1 closes: dispatch M6-S1 interface freeze (single worker)

## Recent ticks

- 2026-05-21 09:30-09:37 — 3 codex workers dispatched
- 2026-05-21 10:10-10:18 — watchman tick #1: P2+P3 workers done, Opus reviewers dispatched
- 2026-05-21 10:38-10:43 — watchman tick #2:
  - P2 Opus reviewer ACCEPT → merge + closeout committed
  - P3 Opus reviewer ACCEPT-AS-GROUNDWORK-PHASE-2 → merge + closeout + M5-S3.y stub committed
  - P1 worker delivered GRAY-ZONE (10 launches honest, HLO 421 KB > 350 KB target, Ni 126975→772, qr met, qg/qv/T/Ni/Nr still miss strict)
  - P1 Opus reviewer dispatched 10:43
- Next: 11:05 tick to catch P1 verdict + dispatch parallel M5-S3.y + P4

## File-ownership snapshot (M6 prologue)

- thompson_*: M5-S1.y owns (in-flight review)
- mynn_*: M5-S2.x CLOSED → free for M6-S3
- rrtmg_*: M5-S3.x CLOSED → M5-S3.y will reopen for setcoef+taumol+Planck

## Anti-pattern observations (this cycle)

- **NO** spec-gaming recurrences from M5-S2 / M5-S3 prior cycles. Both workers honest about scope limits.
- Codex worker pattern improving: self-flag honest partial > pretend parity. Reviewer's verifiability-triple checks (`nm` + non-clipped + non-vacuous) all passed.
- Process learning: dispatching pattern's `/exit` auto-fire reliability issue noted again — manual tap-Enter on stuck panes worked. To encode in pattern doc later.
