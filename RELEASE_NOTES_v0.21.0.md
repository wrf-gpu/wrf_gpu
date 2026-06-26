# Release notes - wrf_gpu v0.21.0

**v0.21.0 is a stability + compile-cache-speed release.** The priority order is
explicit: **STABILITY > IDENTITY > SPEED > MEMORY**.

This release adds a zero-new-regression dycore boundary-stability fix (the all-7
9-domain Canary nest is now finite past the old step-67 divergence window), a
default-on fail-fast finite-state detector, a permanent opt-in steep-terrain GPU
gate, version-keyed compile-cache defaults, XLA autotune-cache fail-open probing,
prewarm pack/unpack tooling, and — the headline compile feature — the **AOT
cheap-key cross-process warm-start**, now **on by default**. After a one-time cold
compile, a fresh process loads the compiled fused GPU executable from disk via a
cheap metadata key and **skips the multi-tens-of-minutes re-lower** the old
persistent cache still paid. The nest runtime default is **fused + AOT**; de-fuse
is an explicit low-host-compile-RAM fallback with documented runtime cost.

> **Compatibility - read first.** The nest default remains the fused runtime
> executable, now with AOT cheap-key warm-start. The de-fuse path is the **same
> bit-identical eager per-domain code path** the `GPUWRF_NESTED_FUSE=0` opt-out
> always used — only the XLA compile partitioning differs, and it has a documented
> runtime cost. `GPUWRF_FINITE_CHECK` is on by default and is observational on
> finite states (it does not mutate values).

> **Honest scope.** v0.21.0's speed win is **compile / warm-start time**, NOT warm
> forecast throughput. Seconds-per-forecast-hour is **unchanged** from v0.20 and
> byte-identical on the default path. v0.21.0 does **not** claim a solved 24-120 h
> forecast-skill gate, full stabilization of the most-extreme 1 km Mont-Blanc
> (~1042 m/cell) terrain, or a long-horizon OOM-proof 9-nest.
>
> **Carried limitations (stated plainly).** (1) The dycore fix stabilizes the 9-nest
> Canary gate-case past its step-67 divergence window, but the most-extreme
> Mont-Blanc (~1042 m/cell) terrain still *relocates* the failure rather than removing
> it — a deep boundary-stability fix is **v0.21.1**. (2) The long-horizon 9-nest can
> still hit a GPU-VRAM OOM around the ~90 min integration horizon (#123,
> mitigated-not-eliminated).

## Validation State

- Version strings are `0.21.0` in `pyproject.toml` and `src/gpuwrf/__init__.py`.
- **Full CPU suite A/B (rigorous, per-file-isolated, baseline vs edited trees):**
  **zero new failures.** The shared pre-existing-debt FAILED set is unchanged from
  the v0.20.2 known-red baseline; the AOT cheap-key path adds no regression.
- **Stability determinant (MEASURED, RTX 5090, 20240901 real case, max_dom=9):** the
  9-nest Canary wrote finite first frames for **all nine domains** and got past the
  step-67 divergence window with no NaN/Inf and no finite-guard abort — the dycore
  boundary fix stabilizes the gate-case.
- **Fused AOT cold→warm gate:** see
  `proofs/v021/canary_gate/V0210_FUSED_AOT_GATE.md` for the fused executable load,
  no-runtime-regression, and bit-identity result.
- **3-domain de-fuse CPU tolerance match (MEASURED):** finite, all outputs, worst T2
  `max_abs 2.96 K / max_rel 1.03%`, all-finite.
- The ≥1 h-finite + all-fields CPU-match Canary gate is measured locally and is **not
  a default v0.21.0 claim**. B200 / fp32 / VRAM work remains v0.21.1+.

Sources: `proofs/v021/blocker_9nest_cheapkey/FIX_REPORT.md`,
`proofs/v021/canary_gate/GATE_RESULTS.md`,
`proofs/v021/canary_gate/HEADLINE_COMPILE_TABLE.md`,
`proofs/v021/canary_gate/V0210_FUSED_AOT_GATE.md`, `proofs/v021/V0210_DEPLOY_STATE.md`.

## The three compile modes

The nest compile default is **fused + AOT**. Three modes:

| Mode | How to select | Host RAM | Cold compile | Warm start | Spawn |
|---|---|---|---|---|---|
| **Fused + AOT (DEFAULT)** | no env | ~60 GB cold compile class | ~50–60 min (1 lower) | **seconds** (fused AOT load, skip re-lower) | none |
| **De-fuse sequential (OPT-IN)** | `GPUWRF_NESTED_DEFUSE_COMPILE=1` or `GPUWRF_NESTED_FUSE=0` | **~20–27 GB** | slower than fused (~70–75 min, 9 separate lowers) | seconds (per-domain AOT) | none |
| **Parallel de-fuse prewarm (OPT-IN)** | de-fuse + `GPUWRF_NESTED_PARALLEL_COMPILE=N` | more (peaked ~65 GB, 3 workers) | faster cold (~25 min prewarm) | seconds (AOT) | spawn workers |

```bash
# DEFAULT — nothing to set: fused + AOT cheap-key warm-start.
export GPUWRF_NESTED_DEFUSE_COMPILE=1     # OPT-IN de-fuse (lower host RAM, slower runtime)
export GPUWRF_NESTED_PARALLEL_COMPILE=4   # OPT-IN parallel prewarm for de-fuse
export GPUWRF_NESTED_PARALLEL_COMPILE=0   # explicit opt-out of parallel prewarm
export GPUWRF_NESTED_FUSE=0               # explicit eager/de-fuse debug path
export GPUWRF_NESTED_AOT=0                # opt out of the AOT warm-start (on by default)
```

The fused default runs **no child processes**. Parallel prewarm is strictly opt-in
and only applies when the de-fuse path is explicitly selected. **De-fuse is a
RAM-for-WALL/runtime trade, not the runtime default** — the fused path compiles
cold faster (one lower) and runs faster; de-fuse lowers host compile RAM.

## AOT cheap-key cross-process warm-start (the headline)

JAX's persistent compile cache stores the *compiled* executable keyed by the *lowered*
HLO, so even a warm hit had to **re-lower** (re-trace) the giant nested module first —
tens of minutes for the 9-nest. v0.21.0 serializes the compiled fused cascade
executable to disk and indexes it by a **cheap key**: a fast hash over the call
metadata, including fused edge geometry, that fully determines the compiled program,
**computed without lowering**. The opt-in de-fuse path uses the same mechanism for
per-domain executables.

The cheap key's source fingerprint is scoped to the **trace-import closure** of the
traced body (the static AST import closure of `gpuwrf.runtime.operational_mode` over
`gpuwrf*` imports), which is a provable **superset** of the trace-reachable source set.
This guarantees it cannot miss an HLO-affecting source edit (the silent-wrong-result
risk) while staying invariant to HLO-irrelevant orchestration edits (the determinism
the earlier blocker lacked). A correctness backstop, `GPUWRF_AOT_VERIFY=1` (lower-once
+ HLO-digest compare, fail-closed, quarantine on mismatch), remains available; the
default is verify-off because a fresh load is **numerically inert** — the cheap key
only *locates* the blob, and the loaded executable is byte-identical to a cold compile.

**MEASURED 9-nest cold→warm re-confirm (RTX 5090, 20240901, max_dom=9, fresh cache):**

- **all 9 domains `loaded=true source=aot_blob` cross-process** (the 9 initial loads
  are contiguous at pipeline entry, no compile/lower between them);
- **0 `fallback:missing`, 0 re-lower** — zero "very slow compile" alarms in the warm
  log → pure cheap-key deserialize;
- **all 9 warm keys byte-match the cold keys on disk** (cross-process determinism);
- **finite integration** (12 root-step groups, 0 NaN / 0 non-finite);
- **warm peak host RSS 16.4 GB**; **load in seconds** vs the cold ~70–75 min compile.

```bash
export GPUWRF_AOT_VERIFY=1   # fail-closed warm-load HLO verification backstop (default off)
```

The serialized blobs are SM/jaxlib/CUDA/flag-specific and version-keyed; any mismatch
(or missing/corrupt blob) fails open to a normal compile.

**Follow-ups (carried, non-blocking):** `_walk`-repr hardening and an import-time env
scanner, tracked for a future patch.

## Stability: fail-fast finite detector

`GPUWRF_FINITE_CHECK` now defaults on for nested forecast boundaries and output
cadence. It fails fast on the first non-finite prognostic value and reports the
`{domain, field, level, step, sim-time, first-bad-index}`. On finite states it is
observational only and does not mutate model values.

```bash
export GPUWRF_FINITE_CHECK=0   # opt out only for explicit max-performance experiments
```

## Stability: dycore boundary mechanism fix

The release fixes the diagnosed steep-terrain mechanism class, not merely the first
downstream symptom. Observed causal chain: a pathological acoustic dry-mass drain in
the boundary zone collapses the small-step mass denominator → `alt = al + alb` crosses
the physical density envelope → `c2a = cpovcv * (pb + p) / alt` becomes singular → the
implicit acoustic `w` solve amplifies the column → pressure/density become unphysical
(and Thompson `Ni` later overflows).

The fix has two parts: (1) limit the acoustic dry-mass drain so one substep cannot
collapse total dry mass in the pathological boundary-zone regime; (2) condition the
`c2a`/`alt` denominator with a positive physical floor instead of the old absolute-value
sign behavior. This is a **mechanism fix**, identity-preserving on the integrated CPU
regression baseline (zero new regressions), and the 9-nest Canary gate-case is now
finite past its step-67 divergence window (all nine domains).

**Carried limitation:** the most-extreme 1 km Mont-Blanc (~1042 m/cell) case still
*relocates* the failure rather than removing it. The deep boundary-stability fix for
the extreme regime is **v0.21.1**.

## Stability: steep-terrain GPU gate

```bash
GPUWRF_RUN_V021_STEEP_TERRAIN_GATE=1 \
scripts/with_gpu_lock.sh --label v021-steep-terrain -- \
  pytest -q tests/test_v021_steep_terrain_stability_gate.py
```

The gate exercises the two-domain Mont-Blanc fixture through the real
`execute_nested_pipeline(...)` path with the finite detector on and both `d01`/`d02`
required finite. It is a local regression gate, not a 24 h or OOM-proof claim.

## Compile cache and autotune

The persistent JAX compile cache now defaults to a version/backend-keyed layout, so a
stale older-release cache is never mistaken for a warm one:

```text
$XDG_CACHE_HOME/gpuwrf/jit/<gpuwrf_version>-jax<JAX>-jaxlib<JAXLIB>-<backend>
$HOME/.cache/gpuwrf/jit/<gpuwrf_version>-jax<JAX>-jaxlib<JAXLIB>-<backend>
```

When `GPUWRF_CACHE` is set, gpuwrf owns `<root>/jit/<version-tag>`. Useful controls:

```bash
export GPUWRF_JAX_CACHE=0
export GPUWRF_CACHE=/path/to/gpuwrf-cache
export GPUWRF_JAX_CACHE_DIR=/explicit/jit
```

The XLA autotune cache is default-on when the executable compile cache is on. Candidate
`--xla_gpu_*` flags are probed in an isolated subprocess first; unsupported flags are
dropped and recorded instead of aborting the main run.

```bash
export GPUWRF_XLA_AUTOTUNE_CACHE=0
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR=/path/to/autotune-cache
```

## Prewarm

Prepared deployments can pack and unpack a warmed cache artifact for the exact
release/backend/JAX combination:

```bash
python -m gpuwrf.runtime.aot_precompile info
python -m gpuwrf.runtime.aot_precompile pack --out gpuwrf-jitcache.tar.gz
python -m gpuwrf.runtime.aot_precompile unpack gpuwrf-jitcache.tar.gz
```

The artifact is tagged by the same `gpuwrf`/JAX/JAXLIB/backend identity as the cache it
was packed from; `unpack` refuses a mismatched tag unless forced, because a mismatched
cache is a guaranteed cold miss and must not be mistaken for a warm run.

## Operational robustness

- Nested namelist scalar handling is more defensive for per-domain values.
- Operational namelists should still use one `key = value` assignment per line.
- Grid-scaled VRAM preflight guidance replaces one fixed threshold for all run sizes:

```bash
export GPUWRF_MIN_FREE_VRAM_GIB=24   # large 1024-scale nested training run
export GPUWRF_MIN_FREE_VRAM_GIB=6    # small 256-scale smoke or stability fixture
```

- Shared-GPU jobs should continue to use the lock helper:

```bash
scripts/with_gpu_lock.sh --label my-gpuwrf-run -- <your command>
```

## Known boundaries

- v0.21.0's speed win is **compile / warm-start time**, not warm forecast throughput.
  Warm s/forecast-hour is unchanged from v0.20 and byte-identical on the default path.
- The most-extreme 1 km Mont-Blanc (~1042 m/cell) terrain is a **carried limitation**:
  the dycore fix stabilizes the standard 9-nest Canary gate-case but relocates the
  failure on the extreme case → **v0.21.1**.
- The ≥1 h-finite + CPU-match Canary gate is a **local** gate, not a default v0.21.0
  claim: in de-fuse mode the single-card 32 GB run can still OOM around the ~90 min
  horizon (#123, mitigated-not-eliminated). B200 / fp32 / VRAM work is v0.21.1+.
- The AOT warm-start ships **verify-off** by default (numerically inert load);
  `GPUWRF_AOT_VERIFY=1` is the fail-closed backstop. Two hardening follow-ups
  (`_walk`-repr; import-time env scanner) are carried.
- No new physics features in v0.21.0 (per `proofs/v021/WRF_V4_FEATURE_AUDIT.md`) — it
  is a compile-cache + stability release.
- Paid large-GPU training output is not claimed here.
