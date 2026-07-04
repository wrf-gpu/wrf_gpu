# Release notes — wrf_gpu v0.19.0

**v0.19.0 is the performance+fidelity release for the all-7-island nested
workflow.** It makes the fused nested cascade the default fast path, restores the
fast `_advance_chunk` loop body, and fixes the live-nest terrain/base-state
initialization mismatch that made HGT/MUB/PB/PHB red on d02-d09.

## Headline result

Measured on the reference RTX 5090 workstation, against the canonical 12-rank
CPU-WRF all-7 pilot for the same case:

| Metric | Result |
| --- | ---: |
| GPU warm all-7 `max_dom=9` fused throughput | **713 s/forecast-hour** |
| Best warm fused segment | **683 s/forecast-hour** |
| CPU-WRF 12-rank baseline | **1020 s/forecast-hour** |
| Warm GPU speedup vs CPU | **1.43x** |
| Cold first fused segment, including one-time compile | **7348 s/forecast-hour** |
| Cold first fused segment wall | **about 41 min** |

The one-time fused XLA compile is large and host-memory intensive; on the
reference workstation it needs a corpus-free/RAM-free window and then persists in
the normal JAX compilation cache. Subsequent runs use the cached executable.

## What changed

### 1. Fused nested cascade is default-on

The all-7 nest used to spend too much time crossing the Python/JAX boundary for
serial child-domain advances and boundary construction. v0.19.0 makes the fused
nested cascade the default operational path where the domain tree is fusable.

This is a **fast fusion mode**, not the bitwise identity/debug mode. It is
tolerance-green against CPU-WRF under the established grid comparator.

Explicit opt-outs:

- `GPUWRF_BITWISE=1` selects the eager, non-fused path used for bit-identical
  debug/identity checks.
- `GPUWRF_NESTED_FUSE=0` disables fused nesting directly.

### 2. `_advance_chunk` returns to the fast traced-count `fori_loop`

The v0.18.0 loop-body rewrite from `fori_loop` to `scan` made the warm leaf
advance much slower. v0.19.0 restores the traced-count `fori_loop` form while
keeping the v0.18.3 Thompson compile-bounded fix for `max_dom=9`.

### 3. Live-nest terrain/base-state initialization matches WRF

The nested runtime now follows WRF's terrain/base-state initialization ordering
for the live-nest terrain blend. This fixes the HGT/MUB/PB/PHB red-field class on
nests d02-d09 and makes the speed gate meaningful: the fast path is now both fast
and field-green.

## Fidelity gate

The release gate ran the all-7-island, 9-domain case for the first forecast hour
in fast fusion mode, compared against the canonical CPU-WRF all-7 pilot from the
same initialization.

Result:

- all expected `wrfout` files present for **d01-d09**;
- all fields finite; no NaN/instability;
- established grid-delta atlas comparator: **PASS on all 9 domains**;
- **102 compared numeric fields/domain**, dynamic + static;
- **0 tolerance failures** against the frozen v0.14 manifest.

Fused mode is **tolerance-green**, not bitwise-identical to the eager path by
design. Use one of the opt-outs above when bitwise/eager identity is the target.

## Proof objects

- `proofs/v019/release_prep/gate_summary.json` — sanitized speed/integration
  summary from the all-7 fused release gate.
- `proofs/v019/release_prep/grid_compare_summary.json` — sanitized all-domain
  grid-comparator summary, 9/9 PASS and 0 tolerance failures.
- Raw retained GPU+CPU `wrfout` files and full comparator outputs were preserved
  by the release gate operator for identity-dashboard follow-up.

## Scope and non-claims

- This is a **single-workstation all-7 nested speedup** vs the local 12-rank
  CPU-WRF baseline, not a multi-GPU throughput claim.
- The first fused run has a large one-time compile. Time any production run after
  warming the persistent compilation cache.
- Default fast fusion is not bitwise-vs-eager. The bitwise/eager path remains
  available via `GPUWRF_BITWISE=1` or `GPUWRF_NESTED_FUSE=0`.
