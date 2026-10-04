# v0.21.0 DEFUSE-DEFAULT FLIP — Validation Report

**Verdict: PASS.** The UNSET nest-compile default is now DE-FUSED + AOT cheap-key
warm-start (was fused). All explicit opt-outs preserved. Spawn-safe (parallel
prewarm stays opt-in; the global default is sequential, no spawn). Full CPU suite:
**ZERO new failures** vs the baseline (edited FAILED set ⊆ baseline FAILED set).
Numerics unchanged (the eager per-domain path is the same bit-identical path the
BITWISE / NESTED_FUSE=0 opt-outs already use; only XLA compile partitioning differs).

Worktree: `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile` @ branch
`worker/opus/vnext-parallel-compile` (base HEAD `63032d24`). CPU-only
(`JAX_PLATFORMS=cpu`). Did NOT touch the running GPU measurement's files/caches, did
not push/tag/merge.

---

## 1. The change (diff, file:line)

### `src/gpuwrf/runtime/domain_tree.py`

**`_nested_fuse_default_enabled()` (~L1303) — flip the UNSET default to de-fused.**
- `fuse is None` branch (~L1320): `defused=False, source="default-fused", return True`
  → `defused=True, source="default-defused-aot-v0210", return False`.
- final fallback for a malformed `GPUWRF_NESTED_FUSE` value (~L1332): same flip
  (`return True/"default-fused"` → `return False/"default-defused-aot-v0210"`).
- All explicit branches UNCHANGED and still honored: `GPUWRF_NESTED_DEFUSE_COMPILE`
  truthy → de-fuse; `GPUWRF_BITWISE` truthy → de-fuse; `GPUWRF_NESTED_FUSE=0`/falsey
  → de-fuse (`source="env:GPUWRF_NESTED_FUSE=0"`); **`GPUWRF_NESTED_FUSE=1` → fused**
  (`source="env:GPUWRF_NESTED_FUSE=1"`). Docstring updated to cite the
  the user-approved v0.21.0 policy.

**`maybe_prewarm_defused_nest()` step (b) (~L1612) — SPAWN-SAFETY GUARD (new).**
Added, after the existing `workers == 0` opt-out:
```python
if workers is None:
    # Unset => sequential AOT sub-mode (no spawn). Parallel prewarm is
    # opt-in via GPUWRF_NESTED_PARALLEL_COMPILE=N.
    NESTED_PRECOMPILE_STATUS["source"] = "skip:sequential-aot-default"
    return dict(NESTED_PRECOMPILE_STATUS)
```
Also: the source label at step (c) was `... if workers else "auto"`; `workers` is now
always an explicit `N>0` there, so it is unconditionally
`f"GPUWRF_NESTED_PARALLEL_COMPILE={workers}"` (the dead `"auto"` branch removed).

**`nested_defuse_env_help()` (~L1694) — text updated** to state the v0.21.0
de-fused+AOT default and that the parallel prewarm is OPT-IN / sequential-when-unset
(removed the now-false "Default = fused" and "unset=auto" phrasing). Substring
assertions in the suite still hold.

---

## 2. Spawn-safety resolution

**Hazard:** the de-fuse path's parallel prewarm spawns child processes
(`multiprocessing.get_context("spawn")` in `aot_precompile.prewarm_defused_nest`,
which re-imports the entry module). With de-fuse now the GLOBAL default, a plain
nest run with NO env flags reaches the gate, and an UNGUARDED entry point (pytest,
web UI, ad-hoc scripts with no `if __name__ == "__main__"`) would spawn-recurse / fail.

**`GPUWRF_NESTED_PARALLEL_COMPILE` unset default** (`_nested_parallel_compile_workers`
~L1536): returns `None` when unset. BEFORE this change `None` flowed into step (c)
and `prewarm_defused_nest(max_workers=None)` → `default_parallel_workers()` → **SPAWN**.

**Resolution / new default sub-mode:** when `GPUWRF_NESTED_PARALLEL_COMPILE` is UNSET
(`workers is None`), the gate now takes the **SEQUENTIAL no-spawn sub-mode**: it skips
the prewarm entirely (`source="skip:sequential-aot-default"`, `active=False`) and lets
the unchanged eager loop cold-compile + AOT-warm-load each domain. The spawning
parallel prewarm is now strictly **OPT-IN** — it fires only when
`GPUWRF_NESTED_PARALLEL_COMPILE` is explicitly set to `N>0`. (`=0` is the explicit
opt-out, unchanged.) So the global default = **de-fuse + SEQUENTIAL + AOT** =
low host-RAM (~20 GB) + AOT warm-start, **zero spawn**.

**Proof it doesn't spawn** (direct assertion harness, CPU, no env flags):
```
DEFAULT (no env):  fused=False  defused=True  source=default-defused-aot-v0210
AOT default-on:    True
FUSE=1 forces fused: OK, source = env:GPUWRF_NESTED_FUSE=1
PREWARM gate (default de-fuse, no PARALLEL): active=False source=skip:sequential-aot-default error=None
PREWARM gate (PARALLEL=2 explicit):          active=True  source=GPUWRF_NESTED_PARALLEL_COMPILE=2  has_error=True (fail-open on fake tree)
PREWARM gate (PARALLEL=0 opt-out):           active=False source=skip:GPUWRF_NESTED_PARALLEL_COMPILE=0
```
The default de-fuse run logs `skip:` (no spawn), exactly as required.

---

## 3. Full CPU suite vs baseline (exact counts)

**Method (rigorous A/B):** to attribute any failure delta to the source diff (not to
the documented single-process CPU-XLA-backend exhaustion warned in `tests/README.md`,
nor to data-path differences), I ran the FULL suite **per-file-isolated** on CPU in
TWO trees with the identical invocation:
- **Baseline** = a clean `git worktree` at HEAD `63032d24` (NO edits).
- **Edited** = the real worktree with the change.
Then diffed the FAILED node-id SETS. 316 test files, 2435 tests collected.

Invocation per file:
`JAX_PLATFORMS=cpu PYTHONPATH=<root>/src python -m pytest -q -p no:cacheprovider -o addopts="" <file>`

| Tree | PASS | FAIL (distinct node-ids) | SKIP | XFAIL | XPASS | TIMEOUT |
|------|------|--------------------------|------|-------|-------|---------|
| Baseline (HEAD, no edits) | 1943 | 54* | 376 | 36 | 2 | 1 (`test_v013_ra_sw_gsfc.py`) |
| Edited (after v0110 fix)  | 1956 | **48** | 376 | 36 | 2 | 1 (same file, identical rc=124) |

\* Baseline's 54 includes **6 data-symlink-control artifacts** unrelated to the diff
(see below); the real shared pre-existing-debt FAILED count is **48** in both trees.

**FAILED node-id set diff (the decisive check):**
- **IN EDITED but NOT in BASELINE = EMPTY → ZERO new failures.**
- IN BASELINE but NOT in EDITED = 6, ALL in `test_rrtm_lw_operational_wiring.py`
  (4) and `test_v013_operational_smoke.py` (2). These are **baseline-control
  artifacts**: a fresh `git worktree` does NOT recreate the untracked
  `data/wrf_pristine -> <USER_HOME>/src/wrf_pristine` symlink (memory item #115), so
  the baseline tree failed `ra_lw_rrtm.py:327 read_text(.../phys/module_ra_rrtm.F)`.
  After I added the same symlink to the baseline tree, `test_rrtm_lw_operational_wiring.py`
  passes 12/12 — confirming these are NOT a source-diff effect. The edited (real)
  tree always had the symlink, so it passes them.

The per-file FAIL-count diff confirms only three files differ, all explained:
`test_rrtm_lw_operational_wiring.py` (edited 0 / baseline 4 — data artifact),
`test_v013_operational_smoke.py` (edited 0 / baseline 2 — data artifact),
`test_v0110_domain_tree.py` (edited's recorded log was STALE — pre test-fix; re-run
fresh = **19 passed, 0 fail**). Every other file matches exactly.

The TIMEOUT file `test_v013_ra_sw_gsfc.py` (rc=124, heavy GSFC radiation savepoint)
times out **identically in BOTH trees** under concurrent CPU load — a runner-time
artifact, not a failure, not caused by the change. The "ERROR=17" in the raw tally
were the runner's `NOSUMMARY`/`TIMEOUT` markers (one per file that had a FAILED test);
the real pytest-ERROR node-id count is 0.

The 48 edited FAILED are all pre-existing data/oracle/env debt (the documented
~53-baseline): savepoint parity needing reference fixtures, GRIB/forcing decode,
oracle checksums, fp32/fp64 reference parity, agentos smoke. None are physics/numeric
regressions; none are attributable to the de-fuse flip.

---

## 4. Test-expectation updates (legitimate, NOT regressions)

Six tests hard-asserted the OLD fused default; updated to the new de-fused default
(plus companion tests added for the explicit fused opt-in path). No assertion was
weakened or skipped.

- `tests/test_b2_compile_efficiency.py`
  - `test_defuse_default_is_fused` → **`test_defuse_default_is_defused_aot`**: no env
    flags now expects `fused=False, defused=True, source="default-defused-aot-v0210"`.
  - NEW `test_fuse_force_keeps_fused`: `GPUWRF_NESTED_FUSE=1` → fused.
  - `test_defuse_falsey_keeps_fused` → **`test_defuse_falsey_falls_through_to_default_defused`**:
    a FALSEY `GPUWRF_NESTED_DEFUSE_COMPILE` no longer forces de-fuse, so the v0.21.0
    default (de-fused) applies — it does NOT fall back to fused.
  - NEW `test_fuse_falsey_defuses`: `GPUWRF_NESTED_FUSE` falsey still de-fuses
    (identity-proof opt-out, `source="env:GPUWRF_NESTED_FUSE=0"`).
- `tests/test_parallel_compile.py`
  - `test_gate_is_noop_when_fused_default` → **`test_gate_is_sequential_nospawn_on_default_defuse`**:
    no env flags now expects `active=False, source="skip:sequential-aot-default"`
    (spawn-safety). NEW `test_gate_is_noop_when_fused_forced` keeps the old
    `skip:fused-default` assertion under explicit `GPUWRF_NESTED_FUSE=1`.
  - `test_gate_active_when_defused_and_fails_open` → **`..._parallel_explicit_fails_open`**:
    now sets `GPUWRF_NESTED_PARALLEL_COMPILE=2` (parallel is opt-in), expects
    `source="GPUWRF_NESTED_PARALLEL_COMPILE=2"`. NEW
    `test_gate_sequential_nospawn_when_defused_parallel_unset` covers the new
    sequential default sub-mode (de-fused, PARALLEL unset → no spawn).
  - `test_verify_env_default_off_and_threaded`: set `GPUWRF_NESTED_PARALLEL_COMPILE=1`
    explicitly (was deleted) so the gate still fires the faked driver.
- `tests/test_v0110_domain_tree.py`
  - `test_fused_factory_default_on_and_gates_d02_only` → **`test_fused_factory_default_off_v0210`**:
    no env flags now expects the fused-cascade factory DISABLED (all parents → None).
    NEW `test_fused_factory_forced_on_gates_d02_only` preserves the d02-only gating
    check under explicit `GPUWRF_NESTED_FUSE=1`.

The three directly-affected test files run **72 passed** together. `test_v0110_domain_tree.py`
re-run fresh = **19 passed**.

---

## 5. Constraints honored

- No numerics/physics change (de-fuse is the existing bit-identical eager path).
- CPU-only throughout (`JAX_PLATFORMS=cpu`); no GPU launch, no nest integration.
- Did not push / tag / merge. Did not touch the running measurement's files
  (`proofs/v021/canary_gate/*`, `canary_gate_driver.py`) or the production JAX cache.
- Minimal diff: default flip + spawn-safety guard + env-help text + test-expectation
  updates. No new real failure was papered over.

**PASS** — ready for manager review.
