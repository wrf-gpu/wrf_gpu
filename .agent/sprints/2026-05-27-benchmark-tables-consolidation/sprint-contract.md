# Sprint Contract — Benchmark + Test Tables Consolidation

**Sprint ID**: `2026-05-27-benchmark-tables-consolidation`
**Created**: 2026-05-27 (user direction: benchmark/test tables for publication)
**Status**: READY — final-mile tables sprint
**Predecessors**:
- Iter 1: `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/`
- Iter 2: `.agent/sprints/2026-05-27-m7-skill-fix-iter2/`
- Publication revision: `publication/draft/paper.md` + `publication/draft/tables/`

## Objective

Produce the consolidated benchmark and test tables needed to make the paper publication-ready and to populate `publish/tables/`. The user explicitly asked for these. Specifically:

1. **Comparators table** — speedup of this work vs published GPU NWP results (Pace 3.5-4×, ICON-exclaim 5.5× socket-to-socket, SCREAM 1.26 SYPD at 3.25 km, NIM 34× dynamics-only, AceCAST 5-14×). Pull numbers + citation keys from `publication/research_brief/english_brief.txt`.

2. **Per-gate M7 acceptance table** — for each of the 8 M7 gates (per MILESTONES.md), the current status + proof object path.

3. **Sprint ledger table** — every sprint that ran across M6/M7 with wall-time, verdict, and key proof object.

4. **Performance evolution table** (extending the existing `performance_summary.md`) — pre-fix, iter-1 post-fix, iter-2 post-fix wall-clock + speedup numbers in a single comparison.

5. **Skill evolution table** (extending `skill_regression_summary.md`) — pre-fix / iter-1 / iter-2 GPU vs CPU BIAS/RMSE/MAE on T2/U10/V10.

6. **Test suite coverage table** — every pytest file in the repository that pins a project invariant or fix, with what it pins.

7. **Figure specs** (markdown only, no rendering) for:
   - Timeline diagram (M0→M8)
   - Role taxonomy (manager / worker / tester / reviewer / debugger)
   - Validation pyramid (Tier 1-4)

Place all in `publish/tables/` and `publish/figures/`. Update `publish/README.md` checklist when done.

## Acceptance

- **AC1 — comparators.md**: `publish/tables/comparators.md` with the speedup-comparison table, full citation keys from references.bib, columns (System, Approach, Hardware, Reported Speedup, Source Citation), at least 5 rows (Pace, ICON-exclaim, SCREAM, NIM, AceCAST).

- **AC2 — m7_gates.md**: `publish/tables/m7_gates.md` with all 8 M7 acceptance gates from `MILESTONES.md`, status, proof object path.

- **AC3 — sprint_ledger.md**: `publish/tables/sprint_ledger.md` with every sprint that ran across M6/M7 — sprint folder, role, ai, branch, verdict, key proof object. Read `.agent/sprints/` directory entries to populate.

- **AC4 — performance_evolution.md**: `publish/tables/performance_evolution.md` with: pre-fix wall-clock, iter-1, iter-2; pre-fix speedup, iter-1, iter-2; all with proof-object pointers.

- **AC5 — skill_evolution.md**: `publish/tables/skill_evolution.md` with: per-variable RMSE / MAE / BIAS for CPU baseline, pre-fix GPU, iter-1 GPU, iter-2 GPU; relative deltas vs CPU.

- **AC6 — test_coverage.md**: `publish/tables/test_coverage.md` listing every pytest in `tests/test_m6*.py` and `tests/test_m7*.py` with: filename, count of tests, what invariant or fix it pins, what would fail-rate look like.

- **AC7 — figures/timeline.md, figures/role_taxonomy.md, figures/validation_pyramid.md**: markdown specs of these three figures (ASCII art OK; will be re-rendered by LaTeX or a separate tool). Include axis labels and key annotations.

- **AC8 — publish/README.md updated** with checklist marking each item complete.

- **AC9 — paper.md updated** to reference the new tables: §2.2 (Related Work) should cite the comparators table; §7 (Results: Performance) should reference performance_evolution.md; §8 (Results: Skill) should reference skill_evolution.md.

- **AC10 — Worker report** with verdict `TABLES_READY` or `BLOCKED`.

## Files Worker May Modify

- `publish/tables/*.md` (NEW — six tables)
- `publish/figures/*.md` (NEW — three figure specs)
- `publish/README.md` (update checklist)
- `publication/draft/paper.md` (small references-to-tables additions in §2.2, §7, §8)
- `.agent/sprints/2026-05-27-benchmark-tables-consolidation/**`

## Files Worker Must Not Modify

- `publication/draft/references.bib` — citations are now frozen
- `publication/draft/honesty_audit.md` — frozen
- `publication/draft/tables/performance_summary.md` and `skill_regression_summary.md` — these are the original pre-iter-2 splits; new evolution tables live in `publish/tables/`
- `src/gpuwrf/**`
- governance files
- `.agent/decisions/**`

## Hard Rules

1. **No new citations.** Use only the 40 entries already in `references.bib`. Add to publish/tables/comparators.md inline using `\cite{...}` keys exactly as in references.bib.
2. **No fabricated numbers.** Every cell must trace to a proof object or a brief-cited prior-art number.
3. **CPU pinning**: `taskset -c 0-3`.
4. **No GPU runtime.** Pure documentation sprint. Parallel-safe with the Gemini agy review.
5. **No remote push.** Local commit on `worker/gpt/benchmark-tables-consolidation` only.

## Proof Objects

- `publish/tables/comparators.md` (AC1)
- `publish/tables/m7_gates.md` (AC2)
- `publish/tables/sprint_ledger.md` (AC3)
- `publish/tables/performance_evolution.md` (AC4)
- `publish/tables/skill_evolution.md` (AC5)
- `publish/tables/test_coverage.md` (AC6)
- `publish/figures/timeline.md`, `role_taxonomy.md`, `validation_pyramid.md` (AC7)
- `.agent/sprints/2026-05-27-benchmark-tables-consolidation/worker-report.md` (AC10)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-4 h
- Branch: `worker/gpt/benchmark-tables-consolidation`
- Worktree: `/tmp/wrf_gpu2_tables`
- GPU usage: NONE (parallel-safe with Gemini)
