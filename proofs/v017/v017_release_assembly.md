# v0.17.0 LEAN release assembly + the 2 must-fix items (opus)

**Release branch: `worker/opus/v017-release`** (cut from `worker/opus/v017-hostgap-fix`
@ `35a21580`). Assembled per `/tmp/opus_release_brief.md` + the 2 must-fixes
`/tmp/opus_release_fix2.md`. **NO tag/push** — the manager verifies the tree and
tags `v0.17.0` + pushes to `wrfgpu`. NO new schemes, NO fp32-physics (→ v0.18).

## Files changed (this release)
| file | change | commit |
|---|---|---|
| `src/gpuwrf/__init__.py` | `__version__` **0.16.0 → 0.17.0** | `35a21580` |
| `RELEASE_NOTES_v0.17.0.md` | NEW performance release; default-on bit-identical vs opt-in `GPUWRF_NESTED_FUSE` fast-mode (+caveats); honest GPU-compute-bound/fp32-STOP ceiling; capability headline | `35a21580`, `b7a967aa` |
| `CHANGELOG.md` | `[0.17.0]` entry | `35a21580` |
| `README.md` | v0.17 perf section + opt-in flags + "Use the manager" quickstart + status bump | `35a21580`, `b7a967aa` |
| `.agent/skills/` (53 files), `AGENTS.md`, `PROJECT_CONSTITUTION.md`, `CLAUDE.md`, `.claude/` | **shipped operating tree** (FIX1) | `4970820b` |
| `proofs/v017/v017_release_assembly.md` | this report | — |

## FIX 1 — `.agent/` operating tree shipped (skills IN, 0.18 roadmap OUT) ✓
`git checkout worker/gpt/v016-fp32-s2 -- .agent/skills AGENTS.md PROJECT_CONSTITUTION.md CLAUDE.md .claude`
(the cleaned lean tree), deliberately **NOT** `.agent/decisions` or `.agent/reviews`.
**Verified on the release branch:**
- `.agent/skills/managing-sprints/SKILL.md` **present** (the README "Use the manager"
  quickstart path resolves; SKILL header = "Operating manual for the wrf_gpu2 MANAGER
  agent").
- `git ls-files .agent | wc -l` = **53**; `git ls-files .agent | grep -c decisions`
  = **0**; `git ls-files .agent | grep -c reviews` = **0**.
- Root AgentOS shipped: `AGENTS.md`, `PROJECT_CONSTITUTION.md`, `CLAUDE.md`,
  `.claude/` (CLAUDE.md + rules/ + skills/README.md).

## FIX 2 — MEASURED/PROJECTED labels at first capability mention ✓
At the FIRST capability-headline mention in **both** docs:
- `RELEASE_NOTES_v0.17.0.md` intro blockquote **and** §4 HEADLINE.
- `README.md` v0.17 HEADLINE paragraph.
Labels: **MEASURED** = 1 km single domain fits one RTX 5090 bit-identical; all-7 1 km
nested runs end-to-end on one card; ~1.27–1.30× opt-in fast-mode vs 12-rank CPU.
**PROJECTED / UNMEASURED** = cluster / multi-GPU weak-scaling + whole-Earth/rack
throughput. The text now explicitly says *"do not read cluster throughput as
measured."*

## Excluded 0.18 forward-work
`.agent/decisions/` (incl. `V018-PRIORITY-QUEUE-20260614.md` and the internal
history/roadmap) and `.agent/reviews/` are **NOT in the release tree** (verified 0
tracked). The only remaining "v0.18" strings are honest forward-pointers
("fp32-physics deferred → v0.18", `RC_MANIFEST.md:24` "v0.18-scoped … not claimed")
— scope honesty, not a roadmap a fresh manager would continue.

## Honest numbers (MEASURED, canary all-7 9/3/1 km vs same-box 12-rank CPU 893 s/hr)
default (eager) 1005 s/hr 0.89× util 56% (**bit-identical to v0.16**) · `GPUWRF_NESTED_FUSE=1`
702 s/hr **1.27×** util 96% (opt-in, tolerance-PASS, NOT bitwise) · fused+edge-only
689 s/hr **1.30×** (edge-only bit-identical eager). Ceiling: GPU-compute-bound
~674 s/hr; fp32 STOP; ≥2×/3× NOT single-card reachable (nsys: many ~1.5 µs kernels,
no hot-spot). Full evidence: `proofs/v017/hostgap_fix_opus.md`.

## Sanity — fast CPU tests (GREEN, re-run on the release branch)
```
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' taskset -c 0-3 \
  python -m pytest tests/test_v0110_domain_tree.py tests/test_v014_noahmp_nested_pipeline.py \
                   tests/test_v017_edge_only_boundary.py tests/test_p0_1a_nesting.py -q
→ 65 passed
```
No multi-hour GPU gate run — the default config is bit-identical (the v0.16 WRF-identity
gates transfer unchanged; §4.3/§5.3 of `hostgap_fix_opus.md`).

## Ready-to-tag checklist (manager verifies the tree, then tags + pushes)
- [x] `__version__ = "0.17.0"`.
- [x] `RELEASE_NOTES_v0.17.0.md` — honest; default-on vs opt-in separated; FUSE caveat (not-bitwise + ~38 min compile) explicit; MEASURED/PROJECTED labelled at first mention.
- [x] `CHANGELOG.md` [0.17.0].
- [x] `README.md` perf section + opt-in flags + "Use the manager" quickstart + status bump + MEASURED/PROJECTED labels.
- [x] **FIX1:** `.agent/skills` + root AgentOS + `.claude` shipped; `.agent/decisions` + `.agent/reviews` excluded (verified skills-in / decisions-out).
- [x] **FIX2:** MEASURED/PROJECTED at first mention in both docs (cluster ≠ measured).
- [x] fast CPU tests green (65).
- [ ] **MANAGER:** verify the tree (`git ls-files .agent | grep -c decisions` = 0; managing-sprints present), GPT critic re-pass, then tag `v0.17.0` + push to `wrfgpu`.
