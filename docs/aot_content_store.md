# Shared compile store keyed by program content (experimental, v0.5.0)

**Status: experimental opt-in, off by default.** Use it to save compile time across builds or source trees whose
forecast programs are identical. Do **not** use it to make two builds byte-identical: that does not work yet
(see "Limits").

## Why

The normal compile cache is keyed by the source tree and the configuration. Two builds whose source differs only
in code that does not reach the forecast program (for example a research tool) get different cache keys and compile
twice. A second compile of the same program can also produce a slightly different GPU binary, because XLA's autotuner
chooses by timing; runs from the two binaries then differ by compile noise.

The store keys each compiled program by the lowered program itself (all constants, the GPU kernels without source
line numbers), the effective compile options, the explicit autotune inputs and the target (JAX/jaxlib, GPU model,
driver and CUDA compiler files). A build that lowers the same program finds the stored binary and runs exactly that
binary. Anything it cannot key safely (several devices, sharding, host callbacks, unknown toolchain and others) is
compiled normally.

## Use

```bash
export GPUWRF_AOT_HLO_STORE=/data/gpuwrf_store          # read/write root shared by the builds
export GPUWRF_AOT_HLO_SEEDS=/data/store_a:/data/store_b  # optional read-only roots
gpuwrf run --input-dir CASE --output-dir OUT --domains-from-namelist
```

The variable names do not enter the cache key. Entries are written once (first writer wins) and verified
(record digest, blob size and SHA-256) every time they are served; a damaged entry is refused and the run compiles
normally. `GPUWRF_AOT_HLO_AUTOTUNE_SEED=1` additionally shares XLA's per-fusion autotune decisions; it is not
GPU-qualified.

## Measured (RTX 5090, nine-nest Canary Islands case, v0.5.0 development tree)

- A second build that found its forecast programs in the store was ready after 602 s instead of 1,907 s for the
  cold compile (3.2×). Warm starts with the store are as fast as without it (ratio 0.97).
- Each variant used exactly one binary per forecast program.

## Limits

- **Not byte-identical across builds.** In the test above the second build still recompiled about 4,100 small
  start-up, initialization and output programs that the store does not cover, and the autotuner chose differently
  for 162 of 343 fusions. Its `wrfout` files therefore differ from the first build by compile noise. For A/B
  experiments compare runs of the **same build**, or measure a noise band with a same-build twin.
- The store adds lowering work on a cache miss; programs it refuses compile as before.
- The attestation is a consistency check, not a signature: anyone who can write to a store root can replace entries.
  Share roots only between trusted users.
