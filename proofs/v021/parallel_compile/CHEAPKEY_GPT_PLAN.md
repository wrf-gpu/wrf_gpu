# GPT Independent Plan: AOT Cheap-Key Manifest

Date: 2026-06-25
Branch reviewed: `worker/opus/vnext-parallel-compile`
Mode: CPU/read-only design review plus this plan document. No code changes.

Input gap: `proofs/v021/parallel_compile/CHEAPKEY_INDEPENDENT_PLAN_BRIEF.md`
is absent in this worktree. I proceeded from the manager task text, the current
AOT implementation, and the installed JAX/JAXLIB 0.10.0 source.

## Verdict

Do not rely on a raw hand-hashed "cheap key" as the only identity guard. The
silent-wrong-result risk is real: a missed HLO determinant can load a valid but
wrong serialized executable and execute without an error.

The safe design is a two-level manifest:

1. `program_key`: metadata-only hash over exactly the determinants of the
   lowered StableHLO program. This is the key that must prove 1:1 with
   `hlo_sha256`: within the supported Step-C call contract, one `program_key`
   must never map to two HLO digests, and any real HLO change must be explained
   by a changed determinant.
2. `exec_key`: metadata-only hash over `program_key` plus target and compile
   options. This is the on-disk blob address for a serialized executable. Same
   HLO can legitimately compile to different executables when target, XLA flags,
   PGLE, or compile options differ, so a single HLO-only key is not enough for
   blob identity.

Default-on AOT should be allowed only after `GPUWRF_AOT_VERIFY=1` has passed a
full 9-nest cold-capture -> fresh warm-load gate with zero verification
mismatches. Production default can then keep verify off for speed, but only for
the exact schema, source fingerprint, JAX/JAXLIB version, target, and compile
environment that passed.

## JAX 0.10.0 Cache-Key Investigation

Installed versions:

- `jax 0.10.0`
- `jaxlib 0.10.0`
- Source root: `<USER_HOME>/miniconda3/lib/python3.13/site-packages/jax`

Conclusion: JAX 0.10.0 does not expose a usable pre-lower persistent cache key.

Evidence:

- `jax/_src/cache_key.py:76-83` defines `cache_key.get(module, devices,
  compile_options, backend, ...)`. The first argument is an MLIR `ir.Module`,
  i.e. the computation already had to be lowered.
- `jax/_src/cache_key.py:102-135` hashes the lowered computation, jaxlib
  version, backend version, XLA flags, compile options, accelerator config,
  compression, and a custom hook.
- `jax/_src/cache_key.py:220-225` hashes canonicalized lowered IR bytecode.
  That is exactly the bottleneck we are trying to avoid.
- `jax/_src/compilation_cache.py:363-377` is only a wrapper around
  `cache_key.get(...)`; it also requires the lowered module.
- `jax/_src/compiler.py:404-446` receives an already-lowered `computation` in
  `compile_or_get_cached`, then computes the persistent key, then reads the
  persistent cache.
- `jax/_src/pjit.py:223-281` builds an opaque process-local C++ pjit cache. In
  this installed build, `PjitFunctionCache` exposes only `capacity`, `size`,
  `clear`, and `clear_all`; `JitGlobalCppCacheKeys` is not exposed through
  `xla_client._xla` for project use.
- `jax/_src/pjit.py:293-301` shows `.trace()` and `.lower()` APIs, but `.lower()`
  still calls `jit_trace(...).lower()`.
- `jax/_src/pjit.py:607-628` caches `_infer_params` by argument signature and
  avals, but this is an in-process trace cache, not a persistent executable key.

Possible clever alternative rejected: use JAX's internal pjit argument signature
or `_infer_params` cache as the AOT lookup key. It is not sufficient: it is
private, process-local, not exposed as a stable string, does not include all
compile/executable determinants, and still does tracing work on a miss. It can
inform our determinant list, but it should not be the production manifest key.

## Supported Scope

Cheap-key AOT is only for the Step-C eager-loop `_advance_chunk_fori` path:

- `GPUWRF_ADVANCE_CHUNK_LOOP` resolves to `fori`.
- The call shape is exactly:
  `_advance_chunk_fori(carry, namelist, start, clock_base, n_steps=int,
  cadence=int)`.
- `clock_base` is non-`None` and built by `build_clock_base(namelist)`.
- The function remains a plain `@jax.jit` with no `static_argnames` and no
  donation on this Step-C function.
- Any unsupported mode returns no cheap key and falls back to normal JIT
  lower/compile.

If any of those contract points change, bump the key schema and require the full
proof matrix again.

## Exhaustive Determinant Inventory

The inventory is split by identity level.

### A. `program_key` determinants: StableHLO identity

These are the determinants of `hlo_sha256` for the supported Step-C call.

1. Function and source identity

   - Function module and qualname.
   - The bytecode of `_advance_chunk_fori`.
   - The transitive traced call graph under `_advance_chunk_fori`: dycore,
     physics, boundary, coupler, and static helper code.
   - `gpuwrf.__version__`.
   - A source-tree fingerprint. In a worktree build, use `git HEAD` plus a
     dirty-tree marker; for default-on production, a dirty tree should either be
     folded into the key by content hash or disable cheap-key lookup.

   Current candidate risk: hashing only the top-level function bytecode plus
   `gpuwrf.__version__` can miss an unversioned edit in a traced callee. The
   plan requires a package/source fingerprint or fail-open on dirty/unidentified
   source.

2. JAX trace identity

   - JAX version and JAXLIB version.
   - XLA extension/runtime version where available.
   - `jax_enable_x64`.
   - JAX dtype semantics that can affect tracing or HLO:
     `jax_numpy_dtype_promotion`, dynamic-shapes mode, explicit-x64 mode, and
     default matmul precision.
   - `jax_default_device` or any active default-device context if it changes
     abstract placement or sharding.

3. JIT static configuration

   From JAX 0.10.0 `pjit.py:264-276` and `pjit.py:563-575`, include or assert
   all of:

   - `static_argnums` and `static_argnames` for this jitted function.
   - `donate_argnums` and `donate_argnames`.
   - explicit `device` and `backend`.
   - input and output shardings: treedef plus leaves.
   - input and output layouts: treedef plus leaves.
   - `compiler_options_kvs`.
   - `keep_unused`.
   - `inline`.
   - active mesh/resource environment.

   For current `_advance_chunk_fori`, most are defaults. The key should still
   include a normalized "jit policy" component recording those defaults, so a
   future decorator change cannot silently reuse an old blob.

4. Runtime call structure and dynamic abstract values

   JAX 0.10.0 lowers through `MetaTy` values; see `pjit.py:1093-1126`.
   Therefore the call signature component must include more than shape/dtype:

   - Flattened `(args, kwargs)` structure in JAX tracing order.
   - All static pytree aux embedded in that structure, canonicalized without raw
     treedef equality.
   - For each dynamic leaf: aval shape, dtype, weak_type, named/dynamic shape
     metadata if present, sharding, format/layout object, `_committed`, and
     whether the original leaf is a NumPy array versus a JAX array.
   - `None` versus present optional subtrees.
   - Leading-dimension variants for coupling batches.
   - The full `OperationalCarry` treedef, including Noah-MP and other post-init
     extra carry subtrees.
   - Dynamic namelist children (`tendencies`, `metrics`, `radiation_static`,
     `gwdo_statics`) as dynamic leaves by structure and aval.
   - `start`, `clock_base`, `n_steps`, and `cadence` as dynamic leaves.

   Important: `n_steps` and `cadence` values are not StableHLO determinants in
   the current function because the function converts them to traced int32
   scalars inside the body. Their presence, avals, and weak-type class matter;
   their concrete Python values should not fragment `program_key`. If a future
   change makes them static, the schema must bump and their values must be
   included.

5. `OperationalNamelist` static aux

   Hash `OperationalNamelist.tree_flatten()[1]` by content, not Python `hash()`
   or `repr()`. It includes:

   - `GridSpec` and its array-bearing metadata, including eta levels, terrain,
     projection, vertical metadata, boundary metadata, halo width, staggering,
     and dycore metric provenance/content that can become HLO constants.
   - Scalar dynamics/physics switches: timestep, acoustic substeps, RK order,
     damping, diffusion, advection options, precision mode, physics families,
     boundary cadence/degrade flags, radiation cadence, GWD/radiation options,
     and all similar scalar controls in the aux tuple.
   - `BoundaryConfig` and other static config dataclasses/enums.
   - `_StaticHolder` contents for Noah-MP, Noah classic, slab, Pleim-Xiu, energy
     params, radiation params, land/static tables, and related static bundles.
     Hash the wrapped content, never holder identity.
   - `_DateClockAux` as a sentinel only when `clock_base` is present and the
     traced function does not read the aux date values. If `clock_base=None`,
     date values become static determinants and cheap-key AOT must fail open or
     use a different schema.

6. Trace-time global environment and module-level knobs

   HLO-branching env vars must be captured even if they do not appear in the
   namelist or dynamic avals. Examples found in trace-reachable code:

   - `GPUWRF_FORCE_FP64`
   - `GPUWRF_MOIST_CQW`
   - `GPUWRF_ACOUSTIC_UNROLL`
   - `GPUWRF_W_CORIOLIS`
   - `GPUWRF_THOMAS_UNROLL`
   - `GPUWRF_ADVANCE_W_SAFE_FLOORS`
   - `GPUWRF_THOMPSON_*`
   - `GPUWRF_MP_COLUMN_*`
   - `GPUWRF_MYNN_*`
   - `GPUWRF_RRTMG_*`
   - `GPUWRF_ADVANCE_CHUNK_LOOP`

   Recommended policy: capture every set `GPUWRF_*` variable except a small
   reviewed denylist of path/control/diagnostic variables that cannot branch the
   traced HLO. This is safer than trying to maintain an include list. CI must
   scan for new `os.environ` reads under trace-reachable modules and require
   each new variable to be classified as HLO-affecting or infrastructure-only.

   Note: unset defaults are represented by code version plus absence of the env
   var. If a default changes, the source fingerprint must change.

7. Closed-over constants and generated constants

   JAX may lift Python constants, static arrays, and jaxpr consts during tracing.
   They must be explained by one of the above components:

   - static namelist aux content;
   - source fingerprint/module constants;
   - trace-time env/config;
   - dynamic aval structure.

   The proof harness must report any HLO change not explained by a changed
   component as a missing determinant.

8. Effects and host callbacks

   `_advance_chunk_fori` should be effect-free for AOT. If tracing produces
   ordered effects, unordered effects, host callbacks, or debug callbacks, the
   manifest path should fail open. JAX's persistent key has callback handling
   modes; we should avoid that class entirely for default-on operational AOT.

### B. `exec_key` determinants: serialized executable identity

`exec_key = sha256(schema, program_key, executable_environment_fingerprint)`.
The executable environment must include:

1. Target and topology

   - backend platform;
   - backend platform_version / driver string;
   - device kind;
   - compute capability;
   - accelerator topology/fingerprint when available;
   - device count and assignment semantics relevant to this single-device AOT
     executable.

2. JAX/JAXLIB/XLA compile environment

   JAX's persistent key includes compile options and flags after lowering; see
   `cache_key.py:102-135` and `compiler.py:139-305`. The cheap `exec_key` must
   include a conservative pre-lower approximation:

   - raw `XLA_FLAGS` and `LIBTPU_INIT_ARGS`;
   - `jax_exec_time_optimization_effort`;
   - `jax_memory_fitting_effort`;
   - `jax_optimization_level`;
   - `jax_memory_fitting_level`;
   - `jax_use_shardy_partitioner`;
   - `jax_compiler_enable_remat_pass`;
   - `JAX_DISABLE_MOST_OPTIMIZATIONS`;
   - `jax_xla_profile_version`;
   - `jax_enable_pgle`, `jax_pgle_profiling_runs`,
     `jax_pgle_aggregation_percentile`, and `jax_compilation_cache_expect_pgle`;
   - `jax_persistent_cache_enable_xla_caches`;
   - `compiler_options_kvs` from the jit decorator;
   - any project XLA-autotune mode that mutates `XLA_FLAGS` before backend init.

   For safety, include raw values even if JAX would exclude some dump/debug flags
   from its persistent key. Over-fragmenting executable blobs is acceptable;
   under-fragmenting is not.

3. Serialized blob and call-contract guards

   These are not lookup determinants, but they must be in the manifest and
   checked on load:

   - `blob_sha256`;
   - `AotMeta.hlo_sha256`;
   - `AotMeta.in_avals`;
   - `AotMeta.kept_var_idx`;
   - `AotMeta.out_tree`;
   - target fingerprint saved at serialization;
   - key schema;
   - domain name.

## Manifest Format

Write one manifest next to each blob:

`<cache>/aot/<version_tag>/<domain>/k_<exec_key>.manifest.json`

Required fields:

- `schema`: e.g. `GPUWRF-AOTKEY-v2`.
- `domain`: `d01` ... `d09`.
- `program_key`.
- `exec_key`.
- `hlo_sha256`.
- `jax_persistent_cache_key`: optional, filled after lower/compile when
  available.
- `source_fingerprint`: gpuwrf version plus git commit/content digest.
- `jit_policy_hash`.
- `static_config_hash`.
- `call_signature_hash`.
- `trace_env_hash`.
- `executable_env_hash`.
- `target_fingerprint`.
- `blob_sha256`.
- `aot_meta_path`.
- `xlaexec_path`.
- `created_utc`.
- `verified`: false by default.

Writes must be temp-file plus `fsync` plus atomic rename. Readers must treat
missing, malformed, mismatched, or partially written manifests as a miss and
fall back to normal compile.

## Load Algorithm

1. Check global AOT gate and loop mode. Unsupported mode returns JIT path.
2. Compute `program_key` and `exec_key` from metadata only. No `.lower()`.
3. Read `<domain>/k_<exec_key>.manifest.json` and matching blob/meta.
4. Fail open if any of these checks fail:
   - schema mismatch;
   - domain mismatch;
   - manifest `exec_key` mismatch;
   - manifest `program_key` mismatch;
   - source fingerprint mismatch;
   - target fingerprint mismatch;
   - blob hash mismatch;
   - AotMeta cheap key/schema mismatch;
   - AotMeta target fingerprint mismatch;
   - runtime selected input avals mismatch;
   - load/deserialize/execute raises.
5. In `GPUWRF_AOT_VERIFY=1`, lower once for this `(domain, exec_key)` and
   require:
   - live `hlo_sha256 == manifest.hlo_sha256`;
   - if we can reconstruct it after lower, live JAX persistent cache key equals
     `manifest.jax_persistent_cache_key`;
   - runtime output treedef/avals compatible with meta.
6. If verify passes, mark the key verified for the process and optionally write
   a versioned `.verified` sidecar tied to `schema/source/target/exec_key`.
7. If verify fails, quarantine or ignore the blob for this process and compile
   fresh. This is fail-closed for the suspect blob but fail-open for the model.

## Cold-Miss Algorithm

On a cheap-key miss:

1. Lower the exact runtime variant.
2. Compute `hlo_sha256`.
3. Compile.
4. Serialize the compiled executable.
5. Compute `program_key`, `exec_key`, and all manifest component hashes.
6. If the lowered `hlo_sha256` contradicts an existing manifest for the same
   `program_key`, refuse to write under that key, log a collision, and use the
   normal compiled path only.
7. Atomically write blob, meta, and manifest.
8. Execute the freshly compiled callable.

## Completeness Argument

For the supported Step-C call, JAX constructs the lowered program from:

- the jitted function and its traced callees;
- the parsed `(args, kwargs)` structure and static aux;
- dynamic argument abstract values plus sharding/layout/commitment metadata;
- active JAX config, mesh, jit policy, and trace-time globals;
- target-independent lowering policy.

Those are exactly the `program_key` components above. Therefore, if two calls
have the same `program_key`, JAX should trace and lower the same StableHLO. If a
call lowers to a different HLO under the same `program_key`, the key missed a
determinant and verification must fail before production default-on.

For serialized executable identity, StableHLO is necessary but not sufficient.
The compiled blob also depends on target and compile options. Those are added by
`exec_key`, and the manifest re-checks target fingerprint and blob hash on load.

The proof target should be stated precisely:

- `program_key -> hlo_sha256` is injective over the determinant matrix: no one
  program key maps to multiple HLO digests.
- For each intentional same-program equivalence class, `program_key` and
  `hlo_sha256` are both equal.
- For each intentional different-program case, `hlo_sha256` differs and
  `program_key` differs.
- `exec_key` may be more specific than HLO, because same HLO under different
  target/compile options must not share serialized executables.

## Proof Plan

Add a CPU proof harness, then a GPU gate.

### CPU determinant matrix

For each case, compute `program_key` without lowering, then lower and record
`hlo_sha256`. Assert no collisions.

Must-vary cases expected to change HLO and `program_key`:

- carry leading shape variant `(1, ...)` vs `(2, ...)`;
- extra Noah-MP-like carry subtree present vs absent;
- fp32 vs fp64 dynamic leaf dtype;
- static namelist scalar option changes that branch code;
- GridSpec eta/terrain/static-array content changes;
- `_StaticHolder` wrapped table content changes;
- `GPUWRF_MOIST_CQW`, `GPUWRF_W_CORIOLIS`,
  `GPUWRF_ADVANCE_W_SAFE_FLOORS`, Thompson/MYNN/RRTMG trace knobs;
- `GPUWRF_ADVANCE_CHUNK_LOOP=static_scan` must not produce a key for Step C;
- source fingerprint/schema change;
- x64/config change if it changes HLO.

Must-not-vary cases expected to keep HLO and `program_key`:

- forecast date changes when `clock_base` is present;
- `start_step` value changes with same scalar aval;
- `n_steps` value changes with same scalar aval;
- `cadence` value changes with same scalar aval;
- distinct but value-identical static aux objects;
- process restart with the same source/config/input metadata.

Executable-only cases:

- Same `program_key` but different `XLA_FLAGS` or compile optimization config:
  same or possibly same HLO, but different `exec_key`.
- Same `program_key` but different GPU target/driver/compute capability:
  different `exec_key`, blob must not load.

Negative controls:

- Deliberately omit static aux hash and prove a collision is caught.
- Deliberately omit runtime sharding/committed metadata and prove either a CPU
  fake-device or single-device committed/uncommitted case exposes the risk, or
  document why the production device-commit contract makes it invariant.
- Deliberately omit a trace-time env var and prove the harness catches a
  `program_key -> two HLOs` collision.
- Corrupt blob bytes and prove load rejects by `blob_sha256`.
- Corrupt manifest `hlo_sha256` and prove verify fails closed for that blob.

### Cross-process CPU load proof

1. Process A lowers/compiles/serializes two shape variants under cheap keys.
2. Process B starts fresh with the same cache dir.
3. Monkeypatch or instrument `_advance_chunk_fori.lower` so an accidental warm
   lower fails the test.
4. Process B computes `exec_key`, loads both variants, and executes them.
5. A changed-call process proves genuine mismatches fall back and capture a new
   variant, not load the old blob.

### GPU gate before default-on

1. Cold 9-nest run with AOT on captures every per-domain/per-shape variant.
2. Fresh warm 9-nest run with `GPUWRF_AOT_VERIFY=1`:
   - every variant logs `loaded=true`;
   - every first key verifies `hlo_sha256`;
   - no lower-only fallback except verification lowers;
   - bit-identity against the non-AOT path.
3. Fresh warm 9-nest run with `GPUWRF_AOT_VERIFY=0`:
   - no lower-only lookup calls;
   - every variant logs `loaded=true`;
   - wall-clock shows the intended compile-wall removal.

## Current Candidate Review Notes

The uncommitted branch already contains a candidate `src/gpuwrf/runtime/aot_cheap_key.py`.
It is directionally right: canonical content hashing, `_StaticHolder` content
hashing, `_DateClockAux` sentinel handling, all-set-`GPUWRF_*` env capture with
a denylist, cheap-key addressing, blob hash, and verify-mode are the right
shape.

Do not ship it default-on until these gaps are closed:

1. Runtime signature hashing should include JAX `MetaTy` fields
   (`sharding`, `format`, `_committed`, `is_np_array`) in addition to
   shape/dtype/weak_type. JAX uses these in the lowering path.
2. Source identity should include a transitive source/package fingerprint, not
   only `_advance_chunk_fori` bytecode plus `gpuwrf.__version__`.
3. Split `program_key` and `exec_key`, or clearly document that the single key is
   an executable key and that the 1:1 proof is against a separate program
   component.
4. Include compile-option/JAX-config determinants from `compiler.get_compile_options`.
5. Add the missing `tests/test_aot_cheap_key.py` or equivalent determinant
   matrix; the candidate docstring references it but it is not present in this
   worktree.
6. Make the env denylist auditable with a CI scan for new trace-reachable
   `GPUWRF_*` reads.
7. Keep `GPUWRF_AOT_VERIFY=1` as a mandatory release gate for any new schema,
   source fingerprint, JAX/JAXLIB version, or target family before verify-off
   production use.

## Acceptance Criteria

Cheap-key AOT is ready for default-on only when all are true:

- `program_key -> hlo_sha256` determinant proof passes on CPU with collision
  negative controls.
- Fresh-process CPU warm-load proof loads multiple shape variants without any
  warm `.lower()`.
- GPU cold-capture writes all 9-nest variants.
- GPU warm verify run loads all variants and reports zero verify mismatches.
- GPU warm verify-off run performs no lower-only lookup and preserves bit
  identity.
- Any manifest/key/blob corruption or mismatch fails open to normal compile.
- Docs state that the identity guard is `program_key/hlo_sha256` verification
  plus `exec_key` target/compile-option separation, not trust in Python object
  identity or raw treedef equality.

