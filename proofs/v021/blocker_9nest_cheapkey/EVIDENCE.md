# RELEASE BLOCKER — 9-nest AOT cheap-key cold→warm non-determinism

**Status:** v0.21.0 release blocker. 3-domain AOT warm-load PASSES; 9-nest AOT warm-load FAILS
(re-lowers every domain instead of loading the cold's blobs). Fix handed to a fresh agent.

Measured on the shared RTX 5090, real-world case `<DATA_ROOT>/wrf_downscale/runs/20240901/cpu`,
branch `worker/opus/vnext-parallel-compile`, worktree `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile`.

## The failure (definitive)

The 9-nest WARM process (fresh process, same config `max_dom=9`, same cache) computes a
**different cheap_key than the COLD process for the SAME domain**, despite byte-identical HLO
(d01 logs `hlo=706d8b8964e3` in both). Result: `loaded=true = 0`; every domain went
`fallback:missing` → `fallback:jit-compiled+aot-captured` (full re-lower). The "warm-start in
seconds" headline is DEFEATED on the real 9-nest target — warm becomes a ~60-min re-lower.

### Per-domain cold-vs-warm cheap_key pairs (from the logs / on-disk blobs)

| domain | COLD wrote (k_…) | WARM requested (k_…) | match? |
|---|---|---|---|
| d01 | `95b6cc99d37c1ef1` | `8dd5141b0b9a99b6` | NO |
| d02 | `2ea33b6db631e1b4` | `48004e1e61cea57f` | NO |
| d03 | `c07b431cf767fb4b` | `681a15cfceb1741e` | NO |
| d04 | `01d856d773b4992f` | `32b16c233281f6f7` | NO |
| d05 | `2b45149ded704441` | `ffd6a226a4b488aa` | NO |
| d06 | `435e32edc873d721` | (warm killed before) | — |
| d07 | `4a30bceac6be2d53` | (warm killed before) | — |
| d08 | `78c8b72563c88d41` | (warm killed before) | — |
| d09 | `27655573c238ab0c` | (warm killed before) | — |

`loaded=true=0`, all `fallback:missing` → re-lower. (warm killed after d06 re-lower to avoid
the wasteful ~60-min full re-lower; failure already proven across d01–d06.)

### 3-domain PASS vs 9-domain FAIL contrast
The 3-domain de-risk (cache `…/canary_gate_cache_ck3seq`) cross-process warm PASSED:
`d01/d02/d03 loaded=true source=aot_blob`, key matched (d01 `k_95b6cc99…` in BOTH cold and warm),
0 fallback, 0 re-lower. So the cheap-key cross-process fix holds for 3-dom but NOT for the 9-nest
run as executed.

## ROOT CAUSE (CPU static analysis + reproduction — by investigation subagent)

`source_fingerprint_hash()` in `src/gpuwrf/runtime/aot_cheap_key.py` (~L317-361, folded into every
domain's program_key at ~L866) content-hashes the **ENTIRE** `src/gpuwrf` tree
(`pkg_root.rglob("*.py")`). So **any uncommitted/edited `.py` under `src/gpuwrf` between the cold
and warm processes shifts ALL domains' cheap_keys uniformly** — even an edit that does NOT change
the lowered HLO. `_advance_chunk_fori` (the traced body) lives in `operational_mode.py:4869`
(untouched), so the HLO is invariant while the key shifts.

**The "3-vs-9 max_dom" pattern is a RED HERRING / timing artifact, NOT a max_dom-specific bug:**
`src/gpuwrf/runtime/domain_tree.py` is **uncommitted-modified** (`git status` = `M`, a +55/-20 AOT
orchestration diff) and was being edited DURING this window (its mtime moved to 15:21; a parallel
agent is implementing "de-fuse+AOT default-on flip"). The 3-domain cold+warm both ran in one
unedited window (keys matched); the 9-nest cold ran before an edit and the 9-nest warm after →
uniform key shift across all 9. Reproduced on CPU: appending one comment line to `domain_tree.py`
flips `source_fingerprint_hash` (`441ab8c3…`→`80e74bfd…`) while `fn_identity_hash(_advance_chunk_fori)`
stays constant.

**Why it is still a real blocker (not just artifact):** `source_fingerprint_hash` conflates
"source identity" with "HLO identity". It will break warm-load whenever cold and warm see any
working-tree/commit difference under `src/gpuwrf` (concurrent agents, a commit between runs,
any HLO-irrelevant edit) — which defeats the feature's purpose (compile-once-reuse across runs/versions).

## PROPOSED MINIMAL FIX (identity-preserving; from subagent)
1. Scope the content digest to **trace-reachable subtrees only** (the existing
   `TRACE_REACHABLE_ENV_SCAN_ROOTS`: `dynamics`/`physics`/`coupling`/`nesting` + `operational_mode.py`
   + traced contracts), and EXCLUDE orchestration/IO/CLI/`aot_*.py`/`domain_tree.py`/`*_pipeline.py`
   that provably cannot change the lowered `_advance_chunk_fori` HLO.
2. Keep `GPUWRF_AOT_VERIFY=1` (lower-once + HLO-digest compare, fail-closed) as the correctness
   backstop so a genuine HLO divergence is always caught (never a silent wrong blob).

## MANDATORY RE-GATE CONDITION
Any cold→warm 9-nest re-gate MUST hold `src/gpuwrf` **frozen (committed, unmodified) across both
the cold and warm runs** — the current evidence pair straddles a live edit. With the source frozen,
even today's `source_fingerprint_hash` would already produce matching keys; the scope-narrowing fix
is what makes the cache robust to HLO-irrelevant edits/commits going forward.

## COLD-run deliverables (these are COMPLETE and valid — the cold is unaffected)
- **Cold de-fuse 9-nest compile wall ≈ 60 min** (all 9 AOT blobs serialized, fresh cache).
- **Clean peak host RAM = 27.6 GB** = `/usr/bin/time -v` Maximum resident set size (28,977,396 kB)
  == `/proc/PID/status` VmHWM. (System-wide `free`/meminfo read ~52 GB but is CONTAMINATED by the
  0:2 12-rank CPU corpus + page cache — do NOT use it. `PARALLEL_COMPILE=0` → no spawn children →
  time -v captures the whole footprint cleanly.)
- **Integration FINITE:** in-process `loaded=true source=aot_blob` ×168+ (≈19+ root steps), no NaN,
  no finite-detector raise (the in-process per-step AOT loads DO work — same-process key is stable).
- **9/9 AOT blobs serialized** (d01–d09 on disk).
- **Per-domain cold-compile breakdown** (for B200 large-grid compile-ETA): d01 ~15 min (incl JAX
  init + 9-domain input load), d02–d06 ~3–8 min each, d07/d08/d09 ~13–18 min each (the large 1km
  inner domains have multiple per-segment shape-variant `_advance_chunk` lowers, ~2.5 min each;
  8 "Very slow compile" XLA alarms total).
- **3-domain de-fuse bonus (manager ck3seq run): PASS** — finite, 8 frames, CPU tol-match worst
  T2 max_abs 2.96 K / max_rel 1.03%, all fields finite, clean RSS 18.3 GB. (Carries the de-fuse+AOT
  field CPU-compatibility — SAME code path, fewer domains.)

## Artifacts
- Cold/warm logs here: `9aotseq_cold.stderr`, `9aotseq_cold.stdout`, `9aotseq_warm.stderr`.
- 9-nest cache (cold blobs `k_<coldkey>` STILL ON DISK, reusable by a fix agent):
  `<DATA_ROOT>/wrf_downscale/canary_gate_cache_9aotseq/aot/0.21.0-jax0.10.0-jaxlib0.10.0-cuda_sm120/d0X/`
  (each d0X dir now has BOTH the cold key blob and the warm-re-lowered key blob for d01–d05).
- Headline table: `../canary_gate/HEADLINE_COMPILE_TABLE.md` (+ `.json`).
- Cheap-key source: `src/gpuwrf/runtime/aot_cheap_key.py` (L317-361, 866). Traced body:
  `src/gpuwrf/runtime/operational_mode.py:4869`. Edited orchestration: `src/gpuwrf/runtime/domain_tree.py`.
- Diagnostic: `proofs/v021/parallel_compile/cheapkey_xproc_components.py`.
