# v0.17 Fast First-Compile Fix Review

Status: CODE-ONLY review, no GPU runs. Source inspected: `.wt-rc`
`worker/perf/v017-rc` at `76e36ba1fafac3f1171e9d94e5291299396cd167`.

## Root Cause Recap

The all-7-island nested run is paying many cold compiles of the same huge
timestep program because the compiled entry is shape-specialized.

- `src/gpuwrf/runtime/operational_mode.py:4198` defines `_advance_chunk` as
  `@partial(jax.jit, static_argnames=("n_steps", "cadence"))`. `start_step` is
  traced, so equal-length intervals of one domain reuse one executable, but the
  `OperationalCarry` and `OperationalNamelist` pytrees still include concrete
  array shapes in the JAX cache key.
- `src/gpuwrf/runtime/domain_tree.py:333` calls `_advance_chunk` once per live
  domain shape via `_operational_advance_factory.advance`.
- The canary all-7 case has 8 distinct horizontal shapes:
  d01 94x60, d02 196x94, d03 103x70, d04 64x67, d05 58x55,
  d06/d07 40x40, d08 100x88, d09 76x76. d06 and d07 share one shape.

So a fresh persistent-cache directory pays roughly 8 cold `_advance_chunk`
GPU compiles, plus smaller per-shape physics compiles. The prior investigation
observed roughly 20 minutes per large compile, giving the release-critical
~2 hour first-run tax. The persistent JAX cache fixes repeated runs on the same
machine, but it does not fix a fresh clone with an empty cache.

## Option 1: XLA Compile-Time Flags

### `--xla_gpu_autotune_level=0`

Compile impact: not useful on the measured v0.17 slow compile. The committed
single-domain flag test in `proofs/v017/identity_fast_compile_report.md` measured
the default slow compile at 9m00.7s, and autotune-off at 10m22.7s for the same
module. That rules out autotune as the dominant cost for this family of modules.
I did not run a nested GPU A/B in this review.

Bit identity / executable identity: not acceptable as a release default. Lowering
autotune can choose different GPU kernels and may change performance. It should
be treated as a different executable configuration, not "same executable, faster
compile".

Persistent-cache key effect: changes the key. Installed JAX hashes ambient
`XLA_FLAGS` into the persistent cache key (`jax/_src/cache_key.py:120`,
`347-370`) except for a short diagnostic/cache-location exclusion list
(`cache_key.py:253-278`). `--xla_gpu_autotune_level` is not excluded.

Verdict: reject.

### `--xla_backend_optimization_level=0`,
`--xla_llvm_disable_expensive_passes=true`, or
`JAX_DISABLE_MOST_OPTIMIZATIONS=1`

Compile impact: could reduce compile work in principle, but not measured here
because GPU runs are forbidden and because this is not a numerically inert
release default. JAX's own `JAX_DISABLE_MOST_OPTIMIZATIONS` path sets
`debug_options.xla_backend_optimization_level = 0` and
`debug_options.xla_llvm_disable_expensive_passes = True`
(`jax/_src/compiler.py:241-244`), which is explicitly a different compiler
configuration.

Bit identity / executable identity: not acceptable. These flags disable optimizer
passes and can change fusion, codegen, runtime speed, buffer assignment, and
floating-point operation grouping. They cannot be claimed to produce the same
executable or the same bit path without a full GPU identity gate per flag.

Persistent-cache key effect: changes the key. If passed through `XLA_FLAGS`,
they are hashed directly unless excluded, and they are not excluded. If applied
through JAX compile options, the serialized compile options are hashed
(`cache_key.py:124-131`, `285-339`).

Verdict: reject for release default. Keep only as a developer diagnostic if ever
needed.

### `--xla_gpu_force_compilation_parallelism=4`

Compile impact: already observed to barely help this nested case. CPU usage was
~108%, consistent with one giant module with little exploitable intra-module
parallelism.

Bit identity / executable identity: expected numerically inert because it changes
compile scheduling, not model math. However, it is not a first-compile fix for
this module.

Persistent-cache key effect: yes, it changes the key. The flag is present in
`XLA_FLAGS`, is not listed in JAX's `xla_flags_to_exclude_from_cache_key`, and
therefore hashes into the persistent key. Toggling it creates a separate cache
population. This is cache pollution risk, even if the executable is otherwise
semantically equivalent.

Verdict: reject as the NCAR default. It is safe as an opt-in operator knob, but
not the solution to clone-and-run-fast.

### XLA persistent autotune/kernel caches

Compile impact: at best partial and unproven for this pathology. The single-domain
flag test already shows autotune is not the dominant cost. These caches also do
not help the first-ever compile of the first shape; they could only help later
similar fusions.

Bit identity / executable identity: likely numerically safe when used as a cache
of compiler sub-results, but it still needs a GPU proof before being advertised.

Persistent-cache key effect: risky. JAX excludes only
`--xla_gpu_experimental_autotune_cache_mode` from the XLA flag portion of the
key. Cache directory flags such as `--xla_gpu_per_fusion_autotune_cache_dir` are
not excluded from the `XLA_FLAGS` hash in the installed JAX. JAX's compile-option
path also leaves cache-related debug options in serialized compile options except
for the mode and CUDA path. This can fragment the persistent cache by cache path.

Verdict: not a release-critical first-compile fix.

## Option 2: Reduce Fused-Module Compile Cost

### Split `_advance_chunk` into smaller jitted units

Compile impact: potentially lower per-module compile time, but no clean release
implementation was found. Calling smaller jitted functions from inside
`_advance_chunk` does not guarantee separate persistent executables; under an
outer `jit`/`scan`, XLA can still lower one large compiled module. To force
separate modules, the runtime would need a host-dispatched timestep or phase
loop, such as physics -> dycore -> boundary as separate top-level JIT calls.

Bit identity: possible in principle only if the exact per-step ordering and all
carried scratch fields remain unchanged, but it is not a small patch. It would
change launch/dispatch structure, donation/buffer lifetime, and warm throughput.
The nested driver already blocks between domain advances; adding intra-step host
dispatches could make the warm run much slower and requires a GPU identity and
throughput gate.

Persistent-cache key effect: would create new HLOs and new cache entries, not
reuse the current release cache. It might reduce cold wall if the sum of smaller
compiles is lower, but this is an unproven new execution path.

Verdict: do not implement for v0.17. Make it a dedicated v0.18 experiment if
first-compile time remains a product priority after release.

### Shape-polymorphic `_advance_chunk`

Compile impact: would be ideal if JAX could compile one executable over dynamic
`ny/nx`. In this codebase it is not available as a clean `jax.jit` replacement
for the live GPU runtime. JAX persistent cache keys still include the lowered
program and concrete executable configuration. The current runtime arrays,
stencils, gather/scatter weights, and boundary leaves are shape-specialized.

Bit identity: not clean. The exact `ny/nx` shape controls C-grid field extents,
boundary side lengths, loop bounds, scatter indices, and feedback masks.

Persistent-cache key effect: would be a new HLO/executable if implemented through
export/AOT mechanisms, not a drop-in reuse of current cache entries.

Verdict: reject for v0.17.

### Uniform padding for the 1 km nests

Compile impact: could collapse the seven 1 km leaf shapes to one padded leaf
shape, reducing the heavy shape count from about 8 to about 3
(d01, d02, uniform leaves), or lower if parent domains were also padded. This is
the only structural path that obviously attacks the root cause.

Bit identity: not clean. The model is shape-exact in multiple places:

- `State` sizes boundary leaves as `max(nx + 1, ny + 1)` and staggered fields as
  exact `(nz, ny, nx+1)`, `(nz, ny+1, nx)`, `(nz+1, ny, nx)`
  (`src/gpuwrf/contracts/state.py:35-99`).
- Nest forcedown weights are built from exact parent and child grids
  (`src/gpuwrf/nesting/boundary_construction.py:95-124`), then ring strips are
  sliced/padded to exact boundary leaf side lengths (`boundary_construction.py:140-166`,
  `219-297`).
- Lateral boundary application indexes exact edge and relaxation-zone rows/cols
  (`src/gpuwrf/coupling/boundary_apply.py:166-238`, `463-620`,
  `1110-1165`).
- Feedback weights and masks use exact child and parent extents and reshape
  through `weights.csn * weights.cwe` and `weights.psn * weights.pwe`
  (`src/gpuwrf/coupling/boundary_feedback.py:170-260`, `362-425`).

Padding would therefore need masks in the dycore, boundary construction,
boundary relaxation, diagnostics/output cropping, and feedback gather/scatter.
Any missed mask changes the artificial padded edge, relaxation math, or parent
feedback footprint. That is a real model/runtime sprint, not a release patch.

Persistent-cache key effect: good after a correct rewrite, because one padded
shape would reuse one cache entry. But it would intentionally change the HLO and
invalidate existing cache entries.

Verdict: defer to v0.18 behind a bit-identity gate. Do not apply now.

### Shape-poly only over interior column physics

Compile impact: insufficient. It could reduce some per-shape physics subcompile
costs, but the dominant `_advance_chunk` module still carries full-grid dycore,
boundary, state, and metrics shapes. The root cause is the top-level carry shape,
not only column physics.

Bit identity: maybe possible for isolated column kernels, but it does not solve
the release problem.

Persistent-cache key effect: would create new subprogram cache entries and still
leave `_advance_chunk` shape-specialized.

Verdict: not a first-compile fix.

## Option 3: Clone-And-Run-Fast Strategy

No clean bit-identical code fix exists inside the v0.17 release window. The best
release path is packaging, not model code:

1. Keep the current release executable and persistent-cache key stable. Do not
   add default XLA flags.
2. Produce a release cache bundle from the exact validated environment:
   v0.17-rc commit, jax/jaxlib version, CUDA plugin, GPU backend, `XLA_FLAGS`,
   and canary all-7 input geometry. The current cache directory mechanism is
   already keyed by HLO + backend + flags + compile options, so a matching cache
   hit is the same executable and bit-identical to cold compile.
3. Publish that cache bundle as a release asset or NCAR handoff artifact, with a
   small installer step that unpacks it into the user's
   `JAX_COMPILATION_CACHE_DIR` / `GPUWRF_JAX_CACHE_DIR`.
4. If the environment does not match, fail closed to the honest local cold
   compile rather than pretending the cache is portable.
5. Document the fallback: empty-cache first all-7 run still pays about 2 hours;
   repeated same-machine runs hit the persistent cache.

This is the only path that is simultaneously bit-identical, cache-key-stable,
and credible for "fresh clone" without a risky runtime rewrite. It is not yet
implemented in this review because no validated cache artifact exists here and
GPU validation is explicitly forbidden.

## Recommendation

Recommendation: do not implement a v0.17 model-code fast-compile patch. Ship the
current `worker/perf/v017-rc` executable unchanged, and make NCAR clone-and-run-fast
use a validated prewarmed persistent-cache artifact for the canary all-7 geometry.

Expected first-run compile behavior:

- Empty cache, no release artifact: unchanged, about 2 hours for the all-7 nested
  case.
- Matching prewarmed cache artifact installed: should be the persistent-cache
  read path rather than cold compile. Exact all-7 wall must be measured by the
  manager on GPU; based on existing project cache-hit behavior, the expected
  scale is minutes rather than hours, but this review did not measure it.

Bit-identical: yes for a persistent-cache hit of the same HLO/backend/flags,
because JAX deserializes the same compiled executable. No physics code or XLA
optimization setting changes.

Cache-key-stable: yes if the release launch uses one canonical flag set and the
cache artifact metadata is matched exactly. No default
`--xla_gpu_force_compilation_parallelism`, autotune-level, or optimization-level
flags should be added, because those fragment the cache key.

Manager GPU validation required:

1. Build the v0.17-rc all-7 canary cache from an empty cache and record compile
   time, cache size, jax/jaxlib/CUDA/GPU metadata, and `XLA_FLAGS`.
2. Run the same all-7 case in a fresh process with only the prewarmed cache
   installed and verify no cold `_advance_chunk` compiles occur.
3. Compare outputs cold-cache vs prewarmed-cache bitwise or through the existing
   v0.17 identity gate.
4. Test mismatch behavior by changing one key component, such as a harmless
   XLA flag or jaxlib version, and confirm the installer refuses the cache or
   the run falls back to cold compile with clear messaging.

No `.wt-fastcompile` worktree or `worker/gpt/v017-fastcompile` branch was
created because the clean bit-identical code-fix criterion was not met.
