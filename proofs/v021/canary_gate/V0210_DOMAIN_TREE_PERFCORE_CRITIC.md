# v0.21.0 domain_tree.py Perf-Core Critic (mandatory pre-commit review)

Date: 2026-06-26
Branch: `worker/opus/vnext-parallel-compile`
Base commit: `36f11c2e`
Reviewer: Opus 4.8 adversarial code critic (read-only, CPU-only)
Scope: the uncommitted fused-cascade dispatch diff
  `src/gpuwrf/runtime/domain_tree.py` (+ supporting `tests/test_aot_executable.py`)

## Verdict: SAFE-TO-COMMIT

Per the manager skill, every kernel/perf-core change requires a frontrunner + an
independent critic before commit. This is that critic. The change is a genuine
dispatch/caching-only refactor of the fused-cascade closure: it replaces a single-slot
cached callable with a bounded, aval-signature-keyed dict, fixing the documented
phase-A/phase-B thrash. No computed array, dtype-of-data, or math-selecting control
flow is touched. 28/28 CPU unit tests pass (`tests/test_aot_executable.py`),
including the new keyed-reuse regression test
`test_fused_cascade_cached_calls_are_keyed_by_aval_signature`.

## Findings (evidence = file:line in domain_tree.py unless noted)

1. **NIT / latent, unreachable in-run — `_aval_signature` drops pytree treedef.**
   `:1359-1375` records only `(shape, dtype, weak_type)` per leaf; container structure,
   None-vs-present optional subtrees, and leaf count/order are dropped. Two pytrees with
   different treedefs could in principle share a signature while lowering to different
   programs. **Confirmed safe for this release** because (a) the `OperationalCarry` treedef
   is fixed by run configuration and seeded once at init; the held forcing is the concrete
   tuple (not None) precisely so the scan carry structure is stable, so the treedef does
   not change within a forecast; (b) the only legitimate in-run variation (phase A child
   boundary T=1 -> phase B T=2) is a leading-dim SHAPE change, which the signature DOES
   capture; (c) the AOT `cheap_key` (`aot_cheap_key.py:786-852`, `carry_aval_hash`) is the
   real correctness gate and hashes treedef + n_leaves + placement_class — a strict superset
   of the signature — so the blob originally bound to a signature was selected by the
   stronger key. Optional future hardening (NOT required, tracked for v0.22 hygiene): fold a
   treedef/leaf-count token into the signature so dispatch is self-defendingly equivalent to
   the cheap_key and the latent gap is closed even if a future scheme ever toggled a subtree
   None<->present mid-forecast.

2. **CONFIRMED SAFE — no over-discrimination.** `:1366-1368`; `fused_jit` (`:1415`) declares
   no `static_argnums`, so int `parent_start`/`child_starts` are traced and different values
   share one executable. Only real shape/dtype/weak_type discriminate = exactly the 2
   legitimate phases.

3. **CONFIRMED SAFE — bounded map.** `:1447` `max_cached_calls = 16`; eviction
   `:1478-1480` pops `next(iter(...))` = oldest insertion-order key (deterministic FIFO),
   `pop(key, None)` cannot KeyError, victim materialized before mutation (no
   mutate-during-iteration). Real schedule has 2 signatures, so eviction never fires;
   bounded at 2. Single-threaded host fused loop. (NIT: FIFO not LRU — perf-only, irrelevant
   at 2 << 16.)

4. **CONFIRMED SAFE — fail-open.** Cached-call execute error `:1606-1618` pops ONLY `sig`,
   re-derives cheap_key + load/lower; never returns a wrong/partial result or marks success.
   Compile-exception path `:1534-1546` likewise pops only `sig` then recompiles. No stale
   callable retained.

5. **CONFIRMED SAFE — cheap_key dominates signature (no re-thrash).** Signature MISS computes
   cheap_key then loads/lowers (`:1620-1683`); `carry_aval_hash` is a strict superset of the
   signature, so one cheap_key can cover several signatures (harmless). The original
   single-slot bug (one signature forced to re-lower because two phases shared the slot) is
   fixed; reverse thrash would need same-shape-different-treedef within one run, which is
   run-invariant here.

6. **CONFIRMED SAFE — default / non-fused / AOT-off path unaffected.** `:1599-1600`
   `if not aot_on: return fused_jit(*args)` short-circuits before any signature/dict logic.
   Diff touches only `_build_fused_cascade_program`. Default runtime byte-unchanged.

7. **CONFIRMED SAFE — numerical identity.** Every diff line touches only the cache container,
   signature derivation, status bookkeeping, and which cached callable is invoked. The
   executables themselves are identical objects; only the lookup changed. No clamp, data
   cast, or physics-branch added.

## Manager decision

Commit exactly what the 9-nest gate validates (current code). The finding-1 treedef-token
hardening is optional, latent-only, and unreachable under the structurally-stable in-run
carry; adding it post-gate would diverge shipped code from the proof object and risk a
needless ~70 min re-gate. Tracked as a v0.22 hygiene item (signature/cheap_key parity),
not a v0.21.0 blocker.

## Commands run by the critic
- `git diff HEAD -- src/gpuwrf/runtime/domain_tree.py tests/test_aot_executable.py`
- `PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests/test_aot_executable.py -q` -> 28 passed
