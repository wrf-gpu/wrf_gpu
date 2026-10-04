# Cheap-key AOT manifest: cross-process HLO-digest fix

Branch: `worker/opus/vnext-parallel-compile` (off `56720d5a`)
Worktree: `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile`
CPU-only verification done here; the GPU cold->warm re-gate is for the manager.

## The bug

On the real GPU 3-domain cold->warm gate the warm process computed the SAME
cheap_key as the cold (its looked-for blob paths `.../d0X/k_<key>` exactly matched
the cold's serialized blob filenames), but the warm-load FAILED:

- `loaded=false source=fallback:cheap-key-meta-mismatch` (manager paraphrased it
  `fallback:missing`) -- the blob EXISTS but the load rejects it.
- `loaded=false source=fallback:verify-error error=... missing HLO digest for verify`.

Root cause: the on-disk `meta.hlo_sha256` was **None**. `serialize()` set
`hlo_sha256 = hlo_sha256 or _compiled_hlo_sha256(compiled)` (old aot_executable.py
~line 440). When the caller's `hlo_sha256` arg arrived `None` (the lower-only digest
was not threaded through on that backend), the fallback `_compiled_hlo_sha256(compiled)`
was used -- and that is the WRONG source:

- On **GPU** it returns `None` (you cannot extract the StableHLO text from an
  already-COMPILED PJRT executable).
- On **CPU** it returns a digest, but one derived from the *compiled* HLO text,
  which is a DIFFERENT program representation than the *lowered* StableHLO. So it
  does NOT equal `hlo_sha256_from_lowered(lowered)` -- the digest the load/verify
  path recomputes. (Reproduced here: lowered digest `1a640d73...` vs compiled
  fallback `7d9fcf3e...` for the same program.)

Then on load: (a) the P1-3 enforcement requires nonempty `meta.hlo_sha256`
(`not meta_hlo` -> `fallback:cheap-key-meta-mismatch`); (b) verify-mode
(`GPUWRF_AOT_VERIFY=1`) cross-checks the live lowered digest against the recorded
one -> None (GPU) or a non-matching value (CPU) -> verify-error/quarantine. The cold
IN-PROCESS load worked only because it used the in-MEMORY AotMeta (which had a value);
cross-process (the real B200 case) reads the on-disk meta and broke.

## The fix (file:line)

The persisted `meta.hlo_sha256` must be the lower-only StableHLO digest, derived
from the SAME source (`hlo_sha256_from_lowered`) the load/verify path uses -- never
the compiled-HLO fallback. The lowered object IS available wherever a fresh compile
happens, so thread it into `serialize()`.

- `src/gpuwrf/runtime/aot_executable.py:404` -- `serialize(...)` gains a
  `lowered: Any | None = None` keyword-only param.
- `src/gpuwrf/runtime/aot_executable.py:457-461,468` -- new digest precedence,
  replacing `hlo_sha256 or _compiled_hlo_sha256(compiled)`:
  1. caller-supplied `hlo_sha256` (the authoritative lower-only digest);
  2. `hlo_sha256_from_lowered(lowered)` when `lowered` is passed (re-derives the
     SAME digest even if the caller's capture returned `None`);
  3. `_compiled_hlo_sha256(compiled)` last-resort, ONLY for direct callers/tests
     with neither -- documented as a non-matching, last-resort non-empty value.
- `src/gpuwrf/runtime/aot_precompile.py:613` -- `_serialize_domain_blob(...)` gains
  `lowered: Any | None = None`; threaded to `aotx.serialize(..., lowered=lowered)`
  at `aot_precompile.py:647`.
- `src/gpuwrf/runtime/domain_tree.py:920` -- the eager-loop AOT capture
  (`_compile_capture_and_call`) passes `lowered=lowered` to `_serialize_domain_blob`.
  This is the load-bearing path: in the manager's GPU gate the cold `k_<key>` blobs
  are written by the eager loop, and `lowered` is already in scope there
  (`lowered.compile()` is called on the next line).
- The prewarm worker (`aot_precompile.py:_compile_one_domain_worker`) already passes
  `hlo_sha256=result.hlo_sha256`, and `precompile()` already derives that from
  `hlo_sha256_from_lowered(lowered)` (precedence rung 1) -- so it persists the correct
  digest with no change. The shared `serialize()` precedence covers both paths.

The LOAD path is unchanged and stays lower-free: it only reads `meta.hlo_sha256` from
the pickled meta on disk. No lowering was added to load. Identity-preserving (only the
metadata digest source changed; the blob bytes are untouched) and fail-open (any digest
derivation error degrades to the next rung; a still-None digest just means the cheap-key
load fails OPEN to a fresh compile, as before).

## Why the cross-process load now succeeds

The cold process serializes the eager-loop-compiled blob under `k_<cheap_key>` with
`meta.hlo_sha256 = hlo_sha256_from_lowered(lowered)` -- a nonempty digest, on GPU and
CPU alike. A fresh warm process computes the same cheap_key, finds the blob, reads the
on-disk meta, and the P1-3 contract now passes (`meta.hlo_sha256` is nonempty, equals
the warm process's own `hlo_sha256_from_lowered(live_lowered)`), so the load returns
`source=aot_blob` instead of `fallback:cheap-key-meta-mismatch`. Verify-mode recomputes
the live lowered digest and it matches the recorded one -> confirmed, no verify-error,
no quarantine.

## Tests (CPU)

Added to `tests/test_aot_cheap_key.py`:

- `test_serialize_persists_lowered_hlo_digest_in_meta` -- `serialize(compiled,
  lowered=lowered)` with NO explicit `hlo_sha256` arg persists `meta.hlo_sha256 ==
  hlo_sha256_from_lowered(lowered)`, and asserts the compiled fallback digest differs
  (the fix's premise).
- `test_serialize_domain_blob_threads_lowered_so_on_disk_meta_has_hlo` -- with
  `hlo_sha256=None, lowered=...` (the exact GPU bug condition), the PICKLED on-disk
  meta carries a nonempty `hlo_sha256` == the lowered digest.
- `test_fresh_process_cheap_key_load_succeeds_after_lowered_serialize` -- fresh-process
  -style `load_domain_blob(cheap_key=...)` returns `source=aot_blob` (loaded, not
  `fallback:cheap-key-meta-mismatch`), and the verify-style cross-check
  (`meta_hlo_sha256 == hlo_sha256_from_lowered(live_lowered)`) confirms (no verify-error).

Result:

```
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q \
  tests/test_aot_cheap_key.py tests/test_aot_executable.py tests/test_parallel_compile.py
# 64 passed in 118.25s
```

All green; zero regressions in the three target suites.

## GPU cold->warm re-gate command (for the manager)

3-domain bigswiss, default-on AOT, cold serialize then a FRESH warm process loads via
cheap_key with verify on (no re-lower, byte-identical). Run on the real GPU with the
shared version-keyed cache. Adjust the runner/fixture invocation to the canary harness;
the load-bearing assertions are the env + the `nested-aot` log lines.

```bash
# COLD: fresh version-keyed cache; eager loop compiles + serializes the k_<key> blobs.
export GPUWRF_JAX_CACHE_DIR=<DATA_ROOT>/gpuwrf_jax_cache   # shared, version-keyed
export GPUWRF_NESTED_AOT=1                               # default-on cheap-key manifest
export GPUWRF_AOT_VERIFY=1                               # verify-mode cross-check
# <run the 3-domain bigswiss canary, COLD process>   # writes .../aot/<tag>/d0X/k_<key>.{xlaexec,meta}

# WARM: a brand-new process, SAME cache; must load by cheap_key WITHOUT re-lowering.
# <run the SAME 3-domain bigswiss canary, FRESH process>
# PASS iff per domain: [gpuwrf:nested-aot] domain=d0X loaded=true source=aot_blob
#   (NOT fallback:cheap-key-meta-mismatch, NOT fallback:missing, NOT fallback:verify-error)
# and the warm wrfout is byte-identical to the cold wrfout.
```

Sanity check on the on-disk meta (no GPU needed) -- every cheap-key meta must have a
nonempty hlo_sha256:

```bash
python - <<'PY'
import pickle, pathlib
root = pathlib.Path("<DATA_ROOT>/gpuwrf_jax_cache/aot")
for m in root.rglob("k_*.meta"):
    meta = pickle.loads(m.read_bytes())
    assert meta.hlo_sha256, f"EMPTY hlo_sha256 in {m}"
    print("ok", m.name, meta.hlo_sha256[:12])
print("all cheap-key metas have a nonempty hlo_sha256")
PY
```

## Unresolved risks

- Verified on CPU only; the GPU cold->warm re-gate above is the real-target proof the
  manager must run (the bug only manifests cross-process on the GPU backend).
- The prewarm path is correct by construction (rung-1 digest) but was not exercised
  cross-process here; the same on-disk-meta sanity check covers prewarm-written blobs too.
