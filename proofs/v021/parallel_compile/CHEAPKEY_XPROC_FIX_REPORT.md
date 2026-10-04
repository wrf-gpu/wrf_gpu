# Cheap-key AOT: cross-process WARM-LOAD fix (v0.21.0 cache-speed blocker)

Branch: `worker/opus/vnext-parallel-compile`
Worktree: `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile` (base `3a8a0eed`)
GPU repro under `scripts/with_gpu_lock.sh` (single-domain canary, ~minutes).

## TL;DR

The warm process computed a **DIFFERENT cheap_key than the cold process**, so it
looked for a `k_<key>` blob the cold run never wrote -> `fallback:missing` for an
artifact that DID exist (just under the cold run's filename). Root cause: the
cheap_key folded in **per-invocation GPU-lock bookkeeping env vars**
(`GPUWRF_GPU_LOCK_TOKEN`/`LABEL`, injected uniquely by `with_gpu_lock.sh`). A
second latent bug: the parallel-prewarm worker keyed off a `ShapeDtypeStruct`
carry while the eager loop keyed off the concrete carry, producing different keys
for the IDENTICAL HLO. Both fixed; GPU two-process load now returns
`source=aot_blob` with byte-identical execution.

## Root cause #1 (THE manager's bug) — `aot_cheap_key.py:global_trace_env_hash`

`global_trace_env_hash()` (`src/gpuwrf/runtime/aot_cheap_key.py`, the
fail-SAFE auto-discovery loop ~`:714`) folds **every** currently-set `GPUWRF_*`
env var (minus a small infra denylist) into the cheap_key. `scripts/with_gpu_lock.sh`
(`:69-74`) exports six `GPUWRF_GPU_LOCK_*` vars on every invocation, two of which
are **UNIQUE per invocation**:

```
GPUWRF_GPU_LOCK_TOKEN = gpuwrf-lock-<pid>-<RANDOM>-<nanos>   # different each run
GPUWRF_GPU_LOCK_LABEL = <--label arg>                       # different each run
```

So the COLD lock-wrapped run and the WARM lock-wrapped run computed DIFFERENT
`global_trace_env_hash` -> different `program_key` -> different `exec_key`
(=`cheap_key`). The warm run looked for `k_<warm-key>` which does not exist; the
cold blob lives at `k_<cold-key>`. Result: `fallback:missing` for files that
"exist" (under the other name). Pure infra bookkeeping with ZERO HLO effect.

Measured (two fresh GPU processes, same canary, lock-wrapped — only the
component-by-component diff differs):

```
cold cheap_key : 24b2ab87...   warm cheap_key : 9c628b28...   (MISMATCH)
DIFF***** global_trace_env_hash   (run1 2e424c59...  vs  run2 4169432a...)
SAME  carry_aval_hash / static_config_hash / fn_identity / source_fingerprint / exec_env
```

**Why CPU tests missed it:** the CPU suite is not wrapped by `with_gpu_lock.sh`,
so the env was identical across its subprocesses and the key was stable. This is
a GPU-gate-only manifestation of an env-capture-too-greedy bug.

### Fix #1

`src/gpuwrf/runtime/aot_cheap_key.py`:
- Added the six `GPUWRF_GPU_LOCK_*` vars to `HLO_AFFECTING_ENV_DENYLIST` (~`:694`).
- Added `HLO_INERT_ENV_PREFIXES = ("GPUWRF_GPU_LOCK_",)` and a prefix check in
  `global_trace_env_hash` (~`:720`) so a FUTURE lock-bookkeeping var under that
  inert namespace cannot silently re-break the warm cache. The fail-safe
  auto-discovery for genuine (unknown) physics knobs is unchanged — only the
  proven-inert lock namespace is excluded.

## Root cause #2 (latent; defeats the prewarm wall-win) — `carry_aval_hash`

The parallel-prewarm worker keys a domain off
`aot_precompile._to_shape_dtype_tree(carry)` (picklable `ShapeDtypeStruct` tree,
no device buffers) while the eager warm loop keys off the CONCRETE runtime carry.
`_to_shape_dtype_tree` is documented to "lower to the identical HLO" — PROVEN on
GPU: both lower to `hlo_sha256 = 376ca795...`. But `carry_aval_hash` hashed the
raw `_leaf_metaty_record` MetaTy strings, which DIFFER between the two:

```
abstract leaf : sharding=None,                committed=None,  is_jax_array=False
concrete leaf : sharding=SingleDeviceSharding,committed=False, is_jax_array=True
```

-> different `carry_aval_hash` -> different cheap_key for the SAME program. So a
prewarm-serialized blob is **unloadable by the eager loop** — the whole point of
the parallel prewarm (write-once-in-child, load-in-parent/fresh) is defeated.

### Fix #2

`src/gpuwrf/runtime/aot_cheap_key.py`:
- Added `_placement_class(rec)`: a coarse, REPRESENTATION-INVARIANT placement
  equivalence class. Abstract (`ShapeDtypeStruct`) AND uncommitted single-device
  default-sharded concrete arrays both map to `"default"` (the cases that lower
  to the same StableHLO for this non-donated single-device jit). A genuinely
  `committed` array -> `"committed:<sharding>"`; a non-single-device sharding ->
  `"sharded:<spec>"` — so the GPT-critic SAFETY intent (never silently reuse a
  blob compiled under a DIFFERENT placement) is preserved.
- `carry_aval_hash` now hashes a canonical projection (`shape`/`dtype`/`weak_type`
  + `placement_class`) instead of the raw process-/representation-variant MetaTy
  strings. `_leaf_metaty_record` still EXPOSES all the raw fields (back-compat /
  introspection; the existing `test_p2_carry_metaty_fields_present` still passes).

Both fixes are identity-preserving (only the lookup-key derivation changed; blob
bytes untouched) and fail-open (any error in `_placement_class` -> `"default"`,
the abstract-tree class, so a cross-process key can never be MORE specific than
the prewarm tree could express).

## Two-process FAST repro (the test the CPU suite was missing)

`proofs/v021/parallel_compile/cheapkey_xproc_repro.py` — PROCESS A serializes the
REAL `_advance_chunk_fori` via the REAL `_serialize_domain_blob` path; a SEPARATE
fresh PROCESS B runs the REAL `load_domain_blob(cheap_key=...)` path with
`GPUWRF_AOT_VERIFY=1` and compares an execution digest.
`cheapkey_xproc_components.py` dumps every key subcomponent for the cross-process
diff that localized the bug.

### GPU result (single-domain canary, under `with_gpu_lock.sh`)

BEFORE fix (the manager's bug, reproduced):
```
A cheap_key = 24b2ab87...   B cheap_key = 9c628b28...
B: loaded=false source=fallback:missing  (k_9c628b28... never written)
```

AFTER fix:
```
A cheap_key = 6e9b8dc7...   EXEC_DIGEST = 6b21c28a...
B cheap_key = 6e9b8dc7...   loaded=true source=aot_blob  EXEC_DIGEST = 6b21c28a...
   (blob is_file()=True, meta.hlo_sha256=376ca795... nonempty, verify confirms,
    BYTE-IDENTICAL execution)
```

Also proven on GPU: prewarm `ShapeDtypeStruct` carry and concrete carry now share
the cheap_key (`f4494af5...` both), and the lowered HLO is identical (`376ca795...`).

## New regression tests (CPU, would have caught both bugs)

`tests/test_aot_cheap_key.py`:
- `test_gpu_lock_env_does_not_fragment_cheap_key` — two FRESH subprocesses with
  DIFFERENT `GPUWRF_GPU_LOCK_TOKEN/LABEL` must compute the SAME cheap_key, and it
  must equal the no-lock key. (Directly reproduces the manager's bug at CPU speed.)
- `test_gpu_lock_env_prefix_is_denylisted` — a NEW `GPUWRF_GPU_LOCK_*` var must
  also be excluded by the prefix guard.
- `test_prewarm_shape_carry_and_concrete_carry_share_cheap_key` — `_to_shape_dtype_tree`
  carry and concrete carry -> SAME cheap_key.
- `test_carry_aval_hash_is_representation_invariant` — same for `carry_aval_hash`.
- `test_placement_class_still_distinguishes_real_placement_changes` — committed /
  non-single-device sharding are NOT collapsed (SAFETY direction preserved).

### Suite result

```
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q \
  tests/test_aot_cheap_key.py tests/test_aot_executable.py tests/test_parallel_compile.py
# 69 passed in 158.63s   (was 64; +5 new, zero regressions)
```

## Remaining risk

- Verified the LOAD MECHANISM cross-process on the single-domain canary on the
  real GPU. The full 3-domain + 9-nest cold->warm GPU re-gate (below) is still the
  manager's release proof — it exercises the prewarm-child path and multiple
  carry-shape variants the canary does not.
- The `with_gpu_lock.sh` env is now inert by design; if a future tool injects a
  per-invocation-unique `GPUWRF_*` var under a NEW prefix, the prefix guard would
  need that prefix added (the auto-discovery is deliberately fail-SAFE/greedy for
  unknown physics knobs, so an inert-but-unique infra var is the one shape that
  needs an explicit allowance). Documented inline.

## EXACT GPU re-gate commands for the manager

3-domain bigswiss then 9-nest canary, default-on AOT + verify, COLD then a FRESH
WARM process. The COLD and WARM runs MUST both be wrapped in `with_gpu_lock.sh`
(that is the configuration that exposed the bug — different lock tokens).

```bash
export GPUWRF_JAX_CACHE_DIR=<DATA_ROOT>/gpuwrf_jax_cache   # shared, version-keyed
export GPUWRF_NESTED_AOT=1                               # default-on cheap-key manifest
export GPUWRF_AOT_VERIFY=1                               # lower-once verify cross-check

# COLD (writes .../aot/<tag>/d0X/k_<key>.{xlaexec,meta}; eager loop + prewarm child)
scripts/with_gpu_lock.sh --label regate-cold-3dom -- \
  <run the 3-domain bigswiss canary, COLD process>

# WARM (brand-new process, SAME cache, DIFFERENT lock token -> must still hit)
scripts/with_gpu_lock.sh --label regate-warm-3dom -- \
  <run the SAME 3-domain bigswiss canary, FRESH process>

# Then the 9-nest:
scripts/with_gpu_lock.sh --label regate-cold-9nest -- <9-nest canary, COLD>
scripts/with_gpu_lock.sh --label regate-warm-9nest -- <9-nest canary, FRESH>

# PASS iff, per domain, the WARM run logs:
#   [gpuwrf:nested-aot] domain=d0X loaded=true source=aot_blob
#   (NOT fallback:missing / fallback:cheap-key-meta-mismatch / fallback:verify-*)
# and the warm wrfout is byte-identical to the cold wrfout.
```

Fast single-domain canary sanity (no full nest; ~minutes, proves the mechanism):

```bash
CACHE=$(mktemp -d)/c
scripts/with_gpu_lock.sh --label xproc-A -- \
  env PYTHONPATH=src GPUWRF_NESTED_AOT=1 \
  python proofs/v021/parallel_compile/cheapkey_xproc_repro.py serialize "$CACHE"
scripts/with_gpu_lock.sh --label xproc-B -- \
  env PYTHONPATH=src GPUWRF_NESTED_AOT=1 GPUWRF_AOT_VERIFY=1 \
  python proofs/v021/parallel_compile/cheapkey_xproc_repro.py load "$CACHE"
# PASS iff B prints  source=aot_blob  and the two EXEC_DIGEST lines match.
```
