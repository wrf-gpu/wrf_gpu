# v0.19.1 — 24-hour VRAM-stability fix for the nested fast path

Patch release. Fixes a GPU memory leak in the default fused nested cascade that
prevented sustained (24-hour) all-7-island `max_dom=9` runs. No interface,
namelist, numerical, or throughput change.

## The bug

The v0.19.0 fused nested fast path is correct and fast for short runs, but over a
long forecast GPU memory grew steadily — roughly **+1 GB per ~20-minute output
group** — and eventually exhausted the card (~30 GB on a 32 GB GPU) after ~9–10
forecast hours, aborting with a CUDA out-of-memory in a downstream cuSPARSE
(`gtsv2`) workspace allocation. A one-hour gate did not surface it; only a
multi-hour run did.

## Root cause

The recursive nested-domain integration callback (`integrate` in
`runtime/domain_tree.py`) is a nested closure that references **itself** (for the
recursion) and also captures the per-segment carry dictionary. The self-reference
forms a Python reference **cycle** that pins one full per-domain carry set. Each
output-interval segment builds a fresh such closure, so one carry set was retained
per output group. Because the cyclic garbage collector does not trip during the
device-bound integration loop (the loop allocates on the GPU, not the Python
heap), those cycles were never reclaimed mid-run and accumulated until OOM.

This was confirmed empirically (live-array census + referrer-chain analysis): the
retained unit is exactly one nine-domain carry set (~1043 device arrays) per
output group, attributable to the `integrate` closure's self-reference cell.

## Fix

Break the self-reference at the end of the integration call so the closures — and
their captured carry copies — are released by reference counting at return. This
is a **memory-only** change: it does not alter any compiled kernel, the numerics,
or the throughput.

## Validation (canonical all-7-island `max_dom=9`, real case)

- **VRAM:** flat/bounded (~10–15 GB) across nine output groups — the prior steady
  climb is eliminated.
- **Throughput:** warm `721 s/forecast-hour` vs 12-rank CPU `1020 s/forecast-hour`
  (≈1.41× faster) — unchanged from v0.19.0 (no regression).
- **Fidelity:** all nine domains tolerance-green against the CPU reference
  (0 tolerance failures) over the established grid comparison.

## Compatibility

No interface, namelist, or numerical change from v0.19.0. The bit-identical eager
opt-out (`GPUWRF_BITWISE=1` / `GPUWRF_NESTED_FUSE=0`) is unchanged.
