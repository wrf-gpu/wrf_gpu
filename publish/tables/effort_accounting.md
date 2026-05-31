# Effort Accounting — wall-clock span, agent-runs, token & cost approximation

**Status:** evidence table for the v0.1.0 paper. Compiled READ-ONLY from git history and
`.agent/` on branch `worker/opus/final-verdict`, 2026-05-31. **Every number is marked
approximate (≈) unless it is a direct git count (exact).** Per the human author's framing
(`publish/paper/human_author_notes.md` §3): this was **not** 24/7 wall-clock — it was **mostly
nightly + solely free-token runs**, an unfunded hobby project. The honest unit of effort is
**sprints / agent-runs**, not human-equivalent hours. The **dead earlier attempt is excluded**
entirely; the spine below starts at this repo's first commit.

## 1. Wall-clock calendar span (exact dates, from git)

| Milestone | Commit | Timestamp | Calendar span from start |
|---|---|---|---|
| First commit (nothing) | `896149f` "Bootstrap AgentOS factory" | 2026-05-18 23:20 | day 0 |
| **v0.0.1 working kernel** (paper APPROVED_FOR_PDF) | `f668937` | 2026-05-28 02:52 | **≈ 9.2 days** |
| **v0.1.0 working replacement** (HEAD now) | `234265a` (d03 24 h validation) | 2026-05-31 14:26 | **≈ 12.6 days** |
| Point of publication | this drive (paper + tables, in progress) | 2026-05-31 | ≈ 12.6 days |

- **Nothing → v0.0.1 kernel: ≈ 9.2 calendar days** (≈ 9 days).
- **Nothing → v0.1.0 (now): ≈ 12.6 calendar days** (≈ 13 days).
- **v0.0.1 → v0.1.0 sub-span: ≈ 3.4 calendar days.**
- **Total project to publication: ≈ 12.6 calendar days (~1.8 calendar weeks).**

**These are calendar spans, NOT continuous work.** Per the author notes and the per-day commit
profile, the swarm ran **mostly overnight / free-token windows** with ~1–2 human top-level
check-ins per day. Active *agent* wall-time is a fraction of the 12.6 calendar days and is not
separately metered. **Sprints / agent-runs (below) are the honest effort unit.**

## 2. Honest effort unit — commits and agent-runs (exact + approx)

| Metric | Value | Source |
|---|---|---|
| Total commits, nothing → HEAD | **884** (exact) | `git rev-list --count HEAD` |
| Commits to v0.0.1 kernel | 685 (exact) | `git rev-list --count f668937` |
| Commits in v0.1.0 drive (v0.0.1 → now) | 199 (exact) | `git rev-list --count f668937..HEAD` |
| Dated sprint directories (`.agent/sprints/`) | 249 (exact) | `ls -d .agent/sprints/2026-*` |
| Sprints with a separate reviewer/tester artifact (≥2 agent-runs) | ≈ 83 | filename scan |
| Distinct late-stage agent worktrees | 32 (exact, late only) | `.claude/worktrees/agent-*` |

**Approximate total agent-runs:** a sprint is one *unit of dispatched work*, but most sprints
ran **2–3 agents** (manager dispatch + frontrunner + verifier/critic; bug-hunt sprints fanned
out 3–4 parallel angles). Taking the 249 sprint dirs with a conservative **≈ 2–3 agent-runs per
sprint** (≈ 83 confirmed multi-role + many bug-hunt fan-outs), the **total agent-runs are
≈ 500–700** across the project. This is an **order-of-magnitude estimate**, not a metered count.

**Per-stage agent-run weights** (≈, from `ai_process_ledger.md`):

| Stage | Approx sprints | Notes |
|---|---|---|
| (a) Foundations & governance M0–M7 (= v0.0.1 kernel) | ≈ 195 | ~71 in M0–M6 (05-18→05-22), ~123 in M6.x/M7/perf/publication (05-23→05-27) |
| (b) F7 dycore rewrite | ≈ 26 | F1→F7N, densest single-theme chain |
| (c) Phase-B physics M8–M17 | ≈ 19 | post-reset |
| (d) M19 viability / skill | ≈ 12 reviews + late sprints | worktree-based agent runs cluster here |
| (e) Perf | ≈ 15 | time-distributed, parallel Opus+GPT+agy probes |
| (f) v0.1.0 finish | ≈ small (this drive) | publish/paper/tables + final-verdict cross-check |

Per-day commit profile (exact, illustrates the nightly bursts):
`05-18: 2 · 05-19: 76 · 05-20: 56 · 05-21: 134 · 05-22: 58 · 05-23: 68 · 05-24: 59 ·
05-25: 85 · 05-26: 47 · 05-27: 85 · 05-28: 93 · 05-29: 76 · 05-30: 27 · 05-31: 18.`
Peak 134 commits/day (05-21) confirms heavy parallel-lane bursts, not steady single-thread work.

## 3. Token approximation

**There are NO per-agent token logs in the repository.** Searched for `*token*`, `*usage*`,
`*cost*`: the only hits are perf roofline cost JSONs (compute FLOPs, unrelated to LLM tokens)
and a sprint `usage.md` that is a script-invocation doc, not a token meter. The token figure is
therefore an **engineering estimate with a stated method and a wide uncertainty band**, not a
measurement.

**Method (agent-runs × typical tokens/run):**

- **Agent-runs:** ≈ 500–700 (Section 2).
- **Typical total tokens per agent-run** (input + output, including the large context each
  worker loads — constitution, sprint contract, skills, WRF Fortran / JAX source references):
  a frontrunner/critic run on this codebase plausibly consumes **≈ 200k–800k tokens** end-to-end
  (long-context Opus/GPT runs with repeated tool calls), with short reviewer runs lower and
  deep dycore/perf sessions higher.
- **Midpoint estimate:** 600 runs × ≈ 450k tokens/run ≈ **2.7 × 10⁸ total tokens**.
- **Uncertainty band:** **≈ 1 × 10⁸ to 6 × 10⁸ total tokens** (order 10⁸; could touch low 10⁹
  if the heaviest long-context runs dominated). **Output tokens** are a minority of this
  (~10–25%), so **total output tokens ≈ 3 × 10⁷ to 1.5 × 10⁸ (order ~10⁷–10⁸)**.

State this in the paper as: **"order 10⁸ total tokens (input+output), ~10⁷–10⁸ output tokens,
estimated as agent-runs × typical-tokens-per-run; no token meter was kept."**

## 4. Cost envelope (approximate, from the resourcing facts in the author notes)

- **Plans used:** a **€200/mo Claude Max** subscription + a **€100/mo GPT Pro** subscription,
  used as a side project. **All runs fit within those plan token limits** ("free-token" runs —
  no metered overage, no API spend beyond the flat subscriptions).
- **No funding of any kind.** The project consumed **no marginal/API cost** beyond the two flat
  monthly subscriptions it shared with the author's other use.
- **Attributable cost envelope:** bounded above by **≈ €300 for the active project month**
  (both subscriptions), and realistically **a fraction of that** since the subscriptions were
  not dedicated to this project. Author's own estimate: **with optimization it could likely have
  been done for ≈ €100 or less.**
- **Headline cost claim (defensible):** *one of the most complex 20-year-old geoscience
  codebases was ported to a validated GPU core within the token limits of a ≈ €300/mo (Claude
  Max + GPT Pro) hobby subscription, unfunded, plausibly reproducible for ≤ €100.*

## 5. Caveats (be honest)

- Calendar spans are **exact git dates** but represent **nightly/free-token, non-continuous**
  work; do not convert to FTE-hours.
- The v0.0.1→v0.1.0 partition is **anchored on a paper-PDF/reset commit, not a git tag** (no
  tags exist in this repo). The boundary is hours-precise but is a *publish freeze*, not a code
  tag.
- Per-stage keyword commit counts **overlap** (parallel lanes); only the 685/199 freeze split is
  a hard partition.
- Agent-run count (≈ 500–700) is a **structural proxy** from sprint dirs × multi-role factor,
  not a dispatch log.
- The token figure is an **estimate, order 10⁸**, with no underlying meter; treat the band
  (1–6 × 10⁸) as the honest uncertainty, and note it **fit inside the stated plan limits**.

## 6. Method summary

`git log --reverse` (first commit), commit-message anchors + ADR-028/README (v0.0.1 vs v0.1.0
boundaries), `git rev-list --count` (hard partitions), `ls .agent/sprints/` + reviewer-artifact
scan (agent-run proxy), `git log --format=%ad --date=short | uniq -c` (nightly-burst profile),
filesystem search for token/usage/cost logs (**none found**), author notes §3 (resourcing &
cost). Token estimate = agent-runs × typical-tokens-per-run with a stated band.
