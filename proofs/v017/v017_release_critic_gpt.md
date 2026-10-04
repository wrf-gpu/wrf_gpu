# v0.17 lean release critic - GPT gap analysis

Date: 2026-06-16
Reviewer: GPT pre-release critic
Scope: docs/claims/evidence review only; no GPU run.

## Verdict

**NO-GO for tagging `v0.17.0` from the artifacts currently visible.**

The underlying performance evidence in `proofs/v017/hostgap_fix_opus.md` is broadly
honest and usable for a public release: the measured opt-in fast path is labeled
as 1.27x/1.30x, the default path is described as bit-identical, and the >=2x
negative result is not buried. Late in this review, Opus release docs appeared in
`.wt-opus-hostgap`; they are directionally aligned with the lean brief, but they
are still uncommitted/in-progress and have several public-release blockers.

## Must Fix Before Tag

1. **Release assembly is still incomplete and uncommitted.**

   The manager brief required a version bump, new `RELEASE_NOTES_v0.17.0.md`,
   a prepended `CHANGELOG.md` entry, README updates, 0.18 exclusion, CPU sanity
   tests, and `proofs/v017/v017_release_assembly.md` (`/tmp/opus_release_brief.md:5-21`).
   `.wt-opus-hostgap` now has modified `README.md`, `CHANGELOG.md`,
   `src/gpuwrf/__init__.py`, and an untracked `RELEASE_NOTES_v0.17.0.md`, but
   `proofs/v017/v017_release_assembly.md` is still absent, `/tmp/opus_release_DONE`
   is still absent, and those release-doc changes are not committed. The release
   notes themselves point to the missing assembly report
   (`.wt-opus-hostgap/RELEASE_NOTES_v0.17.0.md:111-118`). This alone blocks a tag:
   there is no files-changed list, no excluded-0.18 list, no CPU sanity-test result,
   and no ready-to-tag checklist.

2. **The README's manager quickstart points to a skill tree that is absent in the
   Opus release worktree.**

   The new README section says the shipped skill `.agent/skills/managing-sprints`
   is the operating manual (`.wt-opus-hostgap/README.md:165-180`). I found no
   `.wt-opus-hostgap/.agent` directory at all. If `.wt-opus-hostgap` is the release
   branch to tag, this is a hard clone-and-run-fast blocker: either ship the
   `.agent/skills/managing-sprints` tree as promised, or remove the manager
   quickstart. The manager brief requires the former (`/tmp/opus_release_brief.md:13-17`).

3. **Capability wording still risks an overclaim unless MEASURED/PROJECTED labels
   are made explicit at first mention.**

   The release notes headline says the GPU's real value is "1 km fits one card,
   bit-identical + large grids + cluster weak-scaling" (`.wt-opus-hostgap/RELEASE_NOTES_v0.17.0.md:10-16`),
   and repeats "large single grids + cluster weak-scaling are where the throughput
   lives" (`.wt-opus-hostgap/RELEASE_NOTES_v0.17.0.md:99-107`). The README has the
   same pattern (`.wt-opus-hostgap/README.md:57-60`, `.wt-opus-hostgap/README.md:200-210`).
   The whole-Earth memory note is correctly labeled PROJECTED, and multi-GPU
   throughput is later called UNMEASURED, but the first headline mention can still
   read like measured cluster throughput. Fix by saying, at first mention, that
   "1 km single-domain fit" and "all-7 runs end-to-end" are MEASURED, while
   cluster weak-scaling and whole-Earth/rack-scale throughput remain PROJECTED /
   unmeasured.

4. **0.18 forward-work exclusion is not proven.**

   The manager brief explicitly says the 0.17 release must not contain the 0.18
   roadmap and names `.agent/decisions/V018-PRIORITY-QUEUE-20260614.md` for
   exclusion (`/tmp/opus_release_brief.md:17`). I did not find a `.agent` tree in
   `.wt-opus-hostgap`, so I did not find a committed 0.18 roadmap leak there.
   However, the required assembly report listing exclusions is absent, so scope
   hygiene is not yet proven. In the main checkout, `.agent/decisions/V018-PRIORITY-QUEUE-20260614.md`
   is visible and is a direct v0.18 implementation queue
   (`.agent/decisions/V018-PRIORITY-QUEUE-20260614.md:1-15`); ensure that file is
   not added to the public release commit.

5. **No release sanity proof exists.**

   The brief requested the fast CPU nesting/domain-tree/edge-only tests and a
   "ready-to-tag" checklist in `proofs/v017/v017_release_assembly.md`
   (`/tmp/opus_release_brief.md:18-21`). I found no such report. The host-gap proof
   cites edge-only and CPU orchestration tests indirectly, and the new release notes
   cite "23 CPU gates" (`.wt-opus-hostgap/RELEASE_NOTES_v0.17.0.md:41-45`), but
   the release assembly must record the exact test command and result for the
   release candidate.

## Evidence Assessment

- **Default bit-identity claim:** acceptable if shipped with proof links. The report
  states the churn and root-sync fixes are host-side device placement / host wait
  changes with no HLO/op changes (`proofs/v017/hostgap_fix_opus.md:37-38`), and
  summarizes CPU orchestration plus GPU/wrfout bit-compare evidence
  (`proofs/v017/hostgap_fix_opus.md:339-342`). Edge-only is reported as
  bit-identical and default-on (`proofs/v017/hostgap_fix_opus.md:454-456`,
  `proofs/v017/hostgap_fix_opus.md:489-490`). The new Opus release docs state this
  correctly in substance (`.wt-opus-hostgap/RELEASE_NOTES_v0.17.0.md:20-48`), but
  the missing assembly report should pin the exact test commands and outputs.

- **Opt-in fused fast-mode:** acceptable if kept opt-in. The report is clear that
  `GPUWRF_NESTED_FUSE` is tolerance-PASS but not bitwise, with P decorrelation
  from 1.34 to 20.0 over 2 h and a required 72 h gate before default-on
  (`proofs/v017/hostgap_fix_opus.md:40-47`, `proofs/v017/hostgap_fix_opus.md:321-342`,
  `proofs/v017/hostgap_fix_opus.md:513-516`). The new release notes and README do
  make this caveat prominent (`.wt-opus-hostgap/RELEASE_NOTES_v0.17.0.md:50-78`,
  `.wt-opus-hostgap/README.md:39-46`).

- **Performance/headline honesty:** no overclaim found in the source proof. The
  measured table gives 702 s = 1.27x and 689 s = 1.30x vs the 12-rank CPU baseline
  (`proofs/v017/hostgap_fix_opus.md:324-332`). The >=2x negative result is stated
  as a geometry-specific truth for the canary all-7 tiny-nest case, with the
  capability headline separated from single-card speed (`proofs/v017/hostgap_fix_opus.md:450-472`).
  The public docs mostly preserve this; only the cluster/large-grid capability
  wording needs the explicit measured/projected split noted above.

## Nice To Have

1. Add a compact README table with columns: mode, default, measured speed, CPU
   denominator, bitwise status, tolerance status, and proof link. This would make
   the opt-in/default distinction hard to misread.
2. Include the profiler limitation in release notes: nsys trace exists, but
   hardware occupancy counters were blocked by admin privileges, so occupancy is
   inferred from trace/kernel shape plus the ledger.
3. Keep the v0.18 roadmap in the dev repo, but exclude it from the public release
   tag or put it behind an explicitly non-release archive path.

## Decision

**NO-GO now.** This is not a rejection of the v0.17 performance evidence or the
late Opus release-doc draft; it is a release-readiness failure. After the release
assembly lands, the expected path is GO if the `.agent/skills/managing-sprints`
tree is actually shipped (or the README promise is changed), the cluster/large-grid
capability wording is explicitly labeled, the 0.18 roadmap is excluded or explicitly
not shipped, and the release sanity report records green CPU tests.

## Commands / Checks Run

- Read `/tmp/gpt_release_critic.txt`, `/tmp/opus_release_brief.md`,
  `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `.agent/skills/conducting-blind-review/SKILL.md`,
  and `.agent/skills/reporting-to-human/SKILL.md`.
- Read `proofs/v017/hostgap_fix_opus.md`, `README.md`, `CHANGELOG.md`,
  `src/gpuwrf/__init__.py`, `.agent/decisions/V017-PERFORMANCE-RELEASE-PLAN.md`,
  and `.agent/decisions/V018-PRIORITY-QUEUE-20260614.md`.
- Checked for `RELEASE_NOTES_v0.17.0.md`, `proofs/v017/v017_release_assembly.md`,
  `/tmp/opus_release_DONE`, `/tmp/gpt_release_critic_DONE`, and v0.17 release
  artifacts across the main checkout and visible v0.17 worktrees.
- Re-read the late `.wt-opus-hostgap` release draft: `RELEASE_NOTES_v0.17.0.md`,
  `README.md`, `CHANGELOG.md`, and `src/gpuwrf/__init__.py`; checked its git
  status and whether `.agent/skills/managing-sprints` exists there.
