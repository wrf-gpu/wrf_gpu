# FIX REPORT — v0.21.0 9-nest AOT cheap-key cold→warm non-determinism

**Status:** CPU root-cause + fix + regression tests COMPLETE and PROVEN. GPU
9-nest cold→warm re-confirm RUNNING (cold compiling under a fresh cache; warm to
follow). Owner: cheapkey-9nest-fix agent. Worktree
`<USER_HOME>/src/wrf_gpu2_wt/parallel-compile` (branch
`worker/opus/vnext-parallel-compile`).

---

## 1. Root cause — the differing subcomponent

**Component: `source_fingerprint_hash()` in `src/gpuwrf/runtime/aot_cheap_key.py`.**

The old `_source_tree_content_digest(pkg_root)` content-hashed **every** `.py`
under `src/gpuwrf` (`pkg_root.rglob("*.py")`) AND folded in the repo-wide
`git rev-parse HEAD`. That conflates *source identity of the whole package* with
*HLO identity of `_advance_chunk_fori`*. Consequence: **any** edit/commit to
**any** `.py` in the package — including orchestration/IO modules that are NOT
traced into the lowered HLO — shifts `source_fingerprint_hash`, hence
`program_key`, hence `cheap_key`, for **all 9 domains uniformly**, even though the
lowered StableHLO is byte-identical.

The GPU 9-nest cold ran at ~14:13 and the warm at ~15:07. **Between them, a
concurrent agent's uncommitted edit to `src/gpuwrf/runtime/domain_tree.py`** (the
de-fuse default-on flip, +55/−20 AOT orchestration) landed. `domain_tree.py` is a
pure consumer of the traced body — it `imports operational_mode`, it is NOT
imported BY it — so it cannot change the lowered HLO. But the whole-tree digest
saw the edit and shifted every cheap_key. The warm process then looked under keys
the cold process never wrote → `loaded=true=0`, all `fallback:missing` → full
~60-min re-lower.

**The "3-dom passes / 9-dom fails" pattern is a timing artifact, NOT a max_dom
bug:** the 3-dom cold+warm ran inside one unedited window (keys matched); the
9-nest cold+warm straddled the `domain_tree.py` edit.

### Subcomponent dump proof (CPU, A-vs-B, max_dom=9)
Tool: `proofs/v021/parallel_compile/cheapkey_xproc_9nest.py` (builds the REAL
9-domain nest on a GPU-less host via `_load_domains`, GPU-device check shimmed to
CPU; the cheap_key is metadata-only). Per-subcomponent JSON dumps:
`9nest_A.json` (PYTHONHASHSEED=0), `9nest_B.json` (12345), `9nest_C_domedit.json`
(post-fix, with `domain_tree.py` edited, seed 777), `9nest_D_clean.json`
(post-fix, clean, seed 4242).

| evidence | result |
|---|---|
| **A vs B** (frozen src, seeds 0 vs 12345) | **ALL 9 cheap_keys IDENTICAL** → the bug is NOT PYTHONHASHSEED, and the `:214 repr()`-sort suspect ("A") is REFUTED (seed-stable). |
| Non-traced edit (`cli.py`) shifts the old whole-tree `source_fingerprint`? | YES (`441ab8c3…`→`a2b8ef5c…`) — reproduces the mechanism. |
| `fn_identity_hash(_advance_chunk_fori)` across all of the above | constant — the traced body is untouched; only `source_fingerprint` moves. |

**This is exactly the subcomponent the GPU run saw differ.** The differing GPU
keys (d01 cold `k_95b6cc99` vs warm `k_8dd5141b`, etc.) all moved uniformly,
consistent with a single shared component (`source_fingerprint`) shifting — not a
per-domain (carry/static) component. `static_config_hash` and `carry_aval_hash`
are per-domain and were verified IDENTICAL across A/B/C/D for each domain.

---

## 2. The fix (file:line) — scope the digest to the trace-import closure

`src/gpuwrf/runtime/aot_cheap_key.py` (diff: +200/−39):

- **`_TRACE_ROOT_MODULE = "gpuwrf.runtime.operational_mode"`** (L317) — the module
  defining the traced body `_advance_chunk_fori`.
- **`_module_to_source_path()`** (L320), **`_gpuwrf_imports_in_file()`** (L335),
  **`_trace_reachable_source_files()`** (L381) — compute, by AST only (no
  execution, process-stable), the **static import closure** of `_TRACE_ROOT_MODULE`
  over `gpuwrf*` imports (module-level AND function-level, relative AND absolute).
- **`_source_tree_content_digest(pkg_root, files=None)`** (L412) — now hashes the
  given closure file list; `files=None` falls back to the whole tree (fail-open).
- **`source_fingerprint_hash()`** (L444) — uses the closure; **drops the repo-wide
  `git HEAD`** (it shifted on any commit anywhere, re-introducing the churn). Folds
  a `scope` marker so a future scope change forces a clean miss. Removed the now-dead
  `_git_head_commit` + the `subprocess` import.

**Why this is the right scope:** a module can only contribute ops to the lowered
`_advance_chunk_fori` HLO if it is reachable from the traced module's transitive
imports. So the import closure is a **provable SUPERSET** of the trace-reachable
set — it **cannot miss** an HLO-affecting source edit (the silent-wrong-result
risk). Measured: closure = **117 files** vs whole tree = 288; it **excludes**
`domain_tree.py`, `nested_pipeline.py`, `cli.py`, `aot_precompile.py`,
`aot_cheap_key.py` (orchestration/IO, not traced) and **includes** every
`dynamics/`, `physics/`, `coupling/`, `operational_mode.py`, `contracts/state.py`,
`contracts/grid.py` (traced callees).

**Backstop retained:** `GPUWRF_AOT_VERIFY=1` (lower-once + HLO-digest compare,
fail-closed, quarantine on mismatch) remains the correctness net for any HLO
divergence the scope could ever miss. Over-scoping (the fail-open whole-tree
fallback when the closure can't be computed) is SAFE: only an extra miss, never a
wrong load.

---

## 3. Identity-preserving — negative-control evidence

The fix must still distinguish genuinely-different HLO and must still invalidate
on a real traced-callee edit. Proven:

- **Non-traced edit → key INVARIANT** (the bug fixed):
  `source_fingerprint` STAYS `f98aa261…` after appending to `cli.py`
  (before fix it shifted). Regression test
  `test_source_fingerprint_invariant_to_orchestration_edit` covers
  `domain_tree.py`, `nested_pipeline.py`, `cli.py`, `aot_precompile.py`.
- **Traced edit → key CHANGES** (safety preserved):
  `source_fingerprint` `f98aa261…`→`67f1b50b…` after appending to
  `dynamics/core/acoustic.py`. Regression test
  `test_source_fingerprint_responds_to_traced_callee_edit` (parametrized over
  `operational_mode.py`, `dynamics/core/acoustic.py`, `physics/__init__.py`,
  `coupling/physics_couplers.py`, `contracts/state.py`).
- **The existing HLO-injectivity matrix is unaffected**: the per-config
  `static_config_hash`/`carry_aval_hash` discriminators are untouched; `dt_s`,
  `epssm`, `acoustic_substeps` still move the key, `date`/`n_steps`/`cadence`
  still don't. `tests/test_aot_cheap_key.py` injectivity + collision-detection
  tests stay green.

**End-to-end CPU proof at max_dom=9 (the exact cold→warm scenario):** process C
(post-fix, `domain_tree.py` EDITED, seed 777) vs process D (post-fix, CLEAN,
seed 4242): **`source_fingerprint` IDENTICAL (`f98aa261…`) and ALL 9 per-domain
cheap_keys MATCH** — across the concurrent orchestration edit AND across different
hash seeds. This is the GPU failure reproduced and fixed on CPU.

---

## 4. Regression tests (CPU, `tests/test_aot_cheap_key.py`, +168 lines)

Would have caught this blocker:

1. `test_source_fingerprint_invariant_to_orchestration_edit` — an HLO-irrelevant
   orchestration edit must NOT shift the fingerprint.
2. `test_source_fingerprint_responds_to_traced_callee_edit` (×5 params) — a
   traced-callee edit MUST shift it (the under-scope/silent-wrong-result guard).
3. `test_trace_closure_includes_traced_excludes_orchestration` — structural
   assertion on the closure set (contains dynamics/physics/coupling/contracts;
   excludes domain_tree/nested_pipeline/cli/aot_*; strict subset of the tree).
4. `test_cheap_key_stable_across_process_and_concurrent_orchestration_edit` —
   END-TO-END: two fresh processes, different PYTHONHASHSEED, with a
   `domain_tree.py` edit applied between them; the cheap_key (and the lowered HLO)
   MUST match. This is the direct regression for the 9-nest blocker.

**Full file result: 38 passed, 0 failed** (`pytest_full.log`, 260 s) — the new
tests plus every pre-existing identity/injectivity/collision/MetaTy/env test.

---

## 5. GPU re-confirm — PASS (9/9)

Methodology: FRESH cache `canary_gate_cache_9aot_fixed` (the old 9aotseq blobs are
keyed under the buggy key, won't load post-fix), GPU-lock held, src FROZEN across
all runs. `cold_warm_driver.py` prints `SAFETY:SOURCE_FINGERPRINT` at startup so
cold==warm is verifiable. The cold/warm chains were capped (exact-PID SIGTERM)
right after the AOT phase to stay under the bg-task ~60 min duration limit and the
#123 ~90 min integration OOM — the cache feature is fully exercised by the load
phase, not the full forecast.

- **Safety check PASS:** every process logged
  `source_fingerprint=f98aa2615b3368ae6c51cebac…` — IDENTICAL cold↔warm and ==
  the CPU baseline. No trace-reachable edit during the window → runs valid.
- **COLD (compile, fresh cache):** all 9 `_advance_chunk_fori` blobs serialized
  under the FIXED scoped key (d01 `k_eb4d0d7b…` … d09 `k_eb8bb269…`), each with
  `.xlaexec` + `.meta`. HLOs: d01 `706d8b8964e3` … d09 `4414a5b916c0`. (The cold
  was split across two capped passes — the first wrote d01–d08 before its 60-min
  cap, the second loaded those 8 and compiled d09; total 9/9 on disk.)
- **8/9 cross-process correctness (verify-ON relaunch):** a FRESH process computed
  the SAME key the cold wrote for **d01–d08** and logged `loaded=true
  source=aot_blob` with the matching HLO, then re-lowered-once (verify-ON) to
  confirm the loaded HLO == the on-disk meta HLO (fail-closed verify, no
  quarantine). Cross-process **correctness** proof for 8 domains (d09 compiled
  fresh in that pass).
- **9/9 cross-process load — THE HEADLINE (clean full WARM, verify-OFF =
  v0.21.0 default):** a FRESH process, all 9 cached:
  - **`loaded=true source=aot_blob` for ALL 9 distinct domains** (283 load events
    over the run; the 9 initial loads are contiguous at pipeline entry, log lines
    10–18, with NO compile/lower between them).
  - **0 `fallback:missing`, 0 `jit-compiled` (re-lower)** — the whole 317-line
    warm log contains ZERO "Very slow compile" alarms → pure cheap-key
    deserialize, not a re-lower.
  - **All 9 warm keys == the cold keys on disk** (d01–d09 byte-identical key match
    cross-process) — the determinism the blocker lacked.
  - **Load wall = seconds** (the 9 deserializes back-to-back at forecast entry) vs
    the cold's ~60 min compile — the v0.21.0 warm-start headline, restored on the
    real 9-nest target.
- **WARM runtime gate:** **12 root-step groups advanced, 0 NaN / 0 non-finite**
  (FINITE integration on the warm-loaded blobs), peak RSS **16.4 GB** (far under
  the fused ~60 GB and the #123 OOM regime). `time -v` wall 7:52 (JAX init +
  9-domain input load + 12 forecast groups; SIGTERM-capped, exit 143). A
  frame-level CPU-match was NOT run here (the cap stopped before output flush) —
  the de-fuse+AOT field-level CPU tolerance match is already carried by the 3-dom
  ck3seq PASS (same code path) and is outside THIS blocker, which is purely
  cache-key determinism (numerically inert: the key only LOCATES the blob; the
  loaded executable is byte-identical to a cold compile).

**Verdict: the v0.21.0 9-nest AOT cold→warm cache-key non-determinism blocker is
FIXED.** 9/9 warm load=true, 0 re-lower, keys deterministic cross-process and
robust to HLO-irrelevant edits, finite integration, identity-preserving.

---

## Artifacts
- Fix: `src/gpuwrf/runtime/aot_cheap_key.py` (worktree parallel-compile).
- Tests: `tests/test_aot_cheap_key.py`; full log `pytest_full.log` (38 passed).
- CPU repro tool: `proofs/v021/parallel_compile/cheapkey_xproc_9nest.py`.
- CPU subcomponent dumps: `9nest_{A,B,C_domedit,D_clean}.json`.
- GPU driver + launchers: `cold_warm_driver.py`, `run_9nest_{cold,warm}.sh`.
- GPU logs: `9aotfixed_cold.stderr` (cold pass 1), `9aotfixed_cold2.stderr`
  (cold pass 2 + 8/9 verify-ON), `9aotfixed_warm.stderr` (9/9 verify-OFF headline).
- Fresh cold cache (9/9 blobs under the fixed key):
  `<DATA_ROOT>/wrf_downscale/canary_gate_cache_9aot_fixed/`.
