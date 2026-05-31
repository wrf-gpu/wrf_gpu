# AI Process Ledger — sprints / agent-runs per stage

**Status:** evidence table for the v0.1.0 paper. Compiled READ-ONLY from git history,
`.agent/sprints/`, `.agent/decisions/`, and `.agent/milestones/` on branch
`worker/opus/final-verdict`, 2026-05-31. **Every number is approximate** and derived from
the git/sprint-dir spine described in the Method note at the bottom. No per-agent token logs
exist (see `effort_accounting.md`), so "agent-runs" is a structural proxy, not a metered count.

## Spine totals (exact, from git)

- **Total commits, nothing → HEAD:** 884
- **First commit:** `896149f` 2026-05-18 23:20 ("Bootstrap AgentOS factory")
- **v0.0.1 freeze (paper APPROVED_FOR_PDF):** `f668937` 2026-05-28 02:52 — 685 commits to here
- **Project reset to M8–M23 scope:** `fce655d` 2026-05-28 12:33 (same day)
- **HEAD (v0.1.0 drive):** `234265a` 2026-05-31 14:26 — 199 commits after the v0.0.1 freeze
- **Sprint directories** under `.agent/sprints/` (excl. templates): **249** dated dirs
- **Distinct agent worktrees** ever created (`.claude/worktrees/agent-*`): 32 (late-stage only;
  early sprints did not use worktrees, so this undercounts agent invocations)
- **Sprint dirs carrying a separate reviewer/tester artifact** (i.e. ≥2 agent-runs in that
  sprint): ~83

There is **no git tag** in the repo; v0.0.1 was a *publish-repo/paper* freeze, not an
annotated tag in this working repo (confirmed: `git tag` is empty). v0.0.1 and v0.1.0
boundaries below are anchored on commit messages and ADR-028.

## Stage definitions and boundaries

| Stage | Label | Commit / date anchor |
|---|---|---|
| (a) | Foundations & governance (M0–M7) | first commit 2026-05-18 → M7 close + paper PDF `f668937` 2026-05-28; this stage **is** the frozen v0.0.1 kernel |
| (b) | F7 dycore rewrite | F1–F7 chain, `f7-*` sprint dirs, 2026-05-28 → 2026-05-29 (F7N touchdown) |
| (c) | Phase-B physics (M8–M17) | post-reset `fce655d` 2026-05-28, M8/M9/M10–M14/M17 sprints |
| (d) | M19 viability / skill | coupled real-case skill, Coriolis/wind/V10, 2026-05-29 → 2026-05-31 |
| (e) | Perf | segmented-scan, fp32 gates, roofline, D2H, bottleneck synthesis (M6-perf, M7-perf; spans b/c/d in time) |
| (f) | v0.1.0 finish | `v010` D02/D03 validation, publish/paper, POST-0.1.0 roadmap, 2026-05-30 → HEAD |

Note: stages (b)–(f) **overlap in calendar time** (parallel agent lanes after the reset), so
keyword commit-counts below double-count some commits across stages. They are reconciled
against the exact spine totals (685 pre-freeze + 199 post-freeze = 884) and should be read as
*approximate effort weight per theme*, not a partition.

## Stage ledger

### (a) Foundations & governance (M0–M7) — the v0.0.1 kernel
- **Approx commits:** ~685 (all commits up to the v0.0.1 paper-PDF freeze `f668937`). This is
  ~77% of all project commits — the bulk of raw activity, because it includes the entire first
  build (M1 fixtures → M2 backend bakeoff → M3 state/grid → M4 dycore → M5 physics suite →
  M6 coupled forecast + the long M6.x dycore bug-hunt → M7 operational v0 + the v0.0.1 paper).
- **Approx sprints / agent-runs:** ~195 sprint dirs in the M1–M7 / m6x / c1–c2 / publication /
  testing-plan / benchmark / bottleneck families; ~71 of them dated 05-18→05-22 (M0–M6) and
  ~123 dated 05-23→05-27 (the M6.x dycore bug-hunt, M7, perf, and v0.0.1 publication burst).
  Many sprints ran manager + frontrunner + verifier, so **agent-runs materially exceed sprint
  count** (≈1.5–3× per multi-role sprint).
- **Dominant models / roles:** Manager **Opus 4.7** (handover to 4.8 noted 2026-05-23,
  `[manager handover 2026-05-23]`). Frontrunner **GPT-5.5 (codex)** for most implementation.
  Verifier **GPT-5.5 per-sprint** (cross-AI tester became Opus 4.7 from `94ffe18` 2026-05-19).
  Tiebreak **Gemini 3.5 (agy)** appears at C2 architecture review (`2026-05-22-gemini-c2-*`).
- **Key proof objects / milestones:** ADR-001 backend (JAX) → ADR-002 state layout →
  ADR-005 physics suite → ADR-006/008/009 (Thompson/MYNN/RRTMG) → ADR-007 precision →
  M1–M7 CLOSEOUTs in `.agent/decisions/`; M7-PERF-MEASUREMENT-CLOSEOUT; the v0.0.1 paper
  (Sprint #4 PUBLISHABLE → #5 rewrite → #6 APPROVED_FOR_PDF). **Caveat captured in README:**
  the v0.0.1 "bitwise dycore parity" headline was later found to be a JAX-vs-JAX self-compare —
  this stage's headline was honest-effort but partly false, which is *why* stages (b)–(f) exist.

### (b) F7 dycore rewrite
- **Approx commits:** ~79 commits matching the F-stage / dycore-rewrite theme (`[F7*]`,
  acoustic-core, straka, skamarock, buoyancy), 2026-05-28 → 2026-05-29.
- **Approx sprints / agent-runs:** ~26 sprint dirs (`f1`–`f7n`, `f7-mega-dry-dycore-rewrite`,
  `f7-acoustic-core`, `f7d`…`f7n`, `f7-sprint-b/c`, `regroup-plan`, `sprintU-operationalize`).
  This was the densest *single-theme* sprint chain in the project (F7A→F7N = 14 lettered
  sub-sprints in ~2 days), each typically frontrunner + critic.
- **Dominant models / roles:** Manager **Opus 4.8**. Frontrunner shifts to **Opus 4.8 (max)**
  for the rewrite itself; **GPT-5.5** runs as the decisive per-sub-sprint critic (e.g.
  `[F7G council: GPT-5.5 decisive]` `5697739`, `[F7D-verify] GPT-5.5 PARTIAL` `fd28622`).
  **Gemini 3.5 (agy)** dispatched for the dycore deep review (`2026-05-28-agy-dycore-deep-review`).
- **Key proof objects:** `proofs/f7/DYCORE_STATUS.md`; F5 WRF cadence spec; F6 12-step
  transaction audit; both idealized cases (Skamarock warm bubble, Straka density current)
  PASS vs pristine WRF v4.7.1 ground-truth savepoints at F7N (`88ed694`). This stage replaced
  the broken v0.0.1 operational dycore (the ~7 missing WRF operators).

### (c) Phase-B physics (M8–M17)
- **Approx commits:** ~64 commits matching M8–M17 / savepoint / microphysics / radiation /
  surface-flux / lateral-BC themes, mostly 2026-05-28.
- **Approx sprints / agent-runs:** ~19 sprint dirs (`m8a/m8b`, `m9a/b/c`, `m10`, `m11`/`m11p1-3`,
  `m12`, `m13`, `m14`, `m17`, `diagnostic-harness`, `oracle-baseline-regression-suite`,
  `project-reset-blinded/critic`).
- **Dominant models / roles:** Manager **Opus 4.8**; frontrunner mixed **Opus 4.8 / GPT-5.5**;
  verifier consolidating to **per-milestone** (Opus cross-AI + GPT critic). Gemini reactive.
- **Key proof objects:** M8 evidence-freeze + proof registry + savepoint harness
  (`M8_VERIFIED`); M9 divergence_map.json + viability VIABLE; M10 LU_INDEX bitwise match
  (Phase A 3/3); ADR-029 TOST statistics design; ADR-030 M16 conditional. Several closed
  **PARTIAL** (M11/M12/M13/M14/M17) — honestly logged in the closeout commits.

### (d) M19 viability / skill
- **Approx commits:** ~32 commits matching m19 / case3 / V10 / Coriolis / wind / persistence /
  terrain-w, 2026-05-29 → 2026-05-31.
- **Approx sprints / agent-runs:** spans the late `sprintU-operationalize-dycore`,
  `f7-dycore-close-critique`, and the `.agent/reviews/2026-05-29/30` agy + GPT review files
  (~12 review artifacts: agy-phaseB, agy-v10, gpt-coupler-and-plan). Worktree-based agent
  runs cluster here (the 32 `.claude/worktrees/agent-*` are predominantly late-stage).
- **Dominant models / roles:** Manager + frontrunner **Opus 4.8 (max)**; **GPT-5.5** and
  **Gemini 3.5 (agy)** both dispatched as parallel reactive reviewers/cross-checkers
  (`agy-verdict-crosscheck`, `gpt-coupler-and-plan-review`).
- **Key proof objects:** `proofs/m19/` (3-case verdict + persistence baseline + terrain-w
  resolution); MEMORY M19 PREVIEW (T2 RMSE 1.33 K / U10 2.23 / V10 3.70 vs CPU-WRF at +1h);
  the missing-Coriolis root-cause fix (`5319b8d`) that repaired the prognostic winds.

### (e) Perf
- **Approx commits:** ~69 commits matching perf / segmented-scan / roofline / bottleneck /
  fp32 / D2H. Time-distributed (M6-perf-design 05-25, M7 D2H/profile 05-26→05-27, fp32 gates
  later) — overlaps (a)/(c)/(d) rather than being a contiguous block.
- **Approx sprints / agent-runs:** ~15+ sprint dirs (`m6-perf-design`, `m6-perf-pcr-vs-thomas`,
  `m6b-d2h-*`, `m7-d2h-probe-opus/codex`, `m7-1km-memory-audit`, `bottleneck-analysis-codex/agy`,
  `benchmark-tables-consolidation`, `m7-honest-speedup-skill-diff`). Notably dispatched in
  **parallel Opus + GPT + agy** probes (the "parallel localization" pattern).
- **Dominant models / roles:** parallel **Opus 4.8** + **GPT-5.5 (codex)** + **Gemini 3.5 (agy)**
  bottleneck probes; manager synthesizes (`BOTTLENECK-CROSS-MODEL-SYNTHESIS.md`).
- **Key proof objects:** `proofs/perf/` (segmented scan, speedup denominator, fp32 gates,
  roofline cost JSONs); M7-PERF-MEASUREMENT-CLOSEOUT; the corrected **22.26×** apples-to-apples
  vs 28-rank CPU WRF (and the honest post-rewrite 10–15× projection).

### (f) v0.1.0 finish
- **Approx commits:** ~38 commits matching `v010` / D02–D03 validation / publish / paper /
  roadmap / README, 2026-05-30 → HEAD.
- **Approx sprints / agent-runs:** the late `.agent/reviews/` final-verdict + crosscheck set,
  plus the publish/paper drafting lane (this `final-verdict` branch + the paper-table workers).
- **Dominant models / roles:** Manager + frontrunner **Opus 4.8 (max)**; **GPT-5.5** + **agy**
  as final cross-check verdict reviewers; this ledger itself is an Opus 4.8 worker product.
- **Key proof objects:** `7c864fa [v010] D02_VALIDATED` (GPU 3 km d02 ≈ nightly CPU-WRF over 3
  real days, Coriolis-corrected); `234265a` d03 24 h validation; POST-0.1.0-ROADMAP.md;
  `publish/paper/` + `publish/tables/` (paper-grade runtime/compute-cycle analysis).

## Reconciliation

- Exact partition by freeze point: **685** commits to v0.0.1 kernel (stage a) + **199** commits
  in the v0.1.0 drive (stages b–f combined) = **884** total.
- The per-stage keyword counts for (b)–(f) sum to more than 199 because the post-reset lanes
  ran concurrently and many commits touch multiple themes (e.g. a perf commit inside the dycore
  rewrite). Treat (b)–(f) counts as **theme weights**, the freeze partition as the hard split.
- **249** sprint directories is the best whole-project agent-activity proxy; with multi-role
  sprints (~83 carry a separate reviewer/tester artifact) the **agent-run count is higher than
  249** — see `effort_accounting.md` for the order-of-magnitude agent-run estimate.

## Method note

1. Commit counts: `git rev-list --count` for hard partitions; `git log --format=%s | grep -iE`
   for theme weights (approximate, overlapping).
2. Stage boundaries: anchored on `f668937` (v0.0.1 paper PDF) and `fce655d` (M8–M23 reset),
   both 2026-05-28, per ADR-028 and README.
3. Sprint/agent-run counts: `ls -d .agent/sprints/2026-*` and family `grep`; multi-role inflation
   estimated from sprint dirs containing reviewer/tester artifacts.
4. Model/role attribution: commit-message mentions (Opus 78, GPT/codex 20, Gemini/agy 26 across
   all 884 messages) cross-referenced with `.agent/sprints/` reviewer filenames and the
   operating-model memory (manager Opus 4.7→4.8; frontrunner GPT-5.5→Opus 4.8 max; verifier
   per-sprint→per-milestone; Gemini reactive tiebreak).
5. **No token logs exist** — agent-runs are a structural proxy, not metered usage.
