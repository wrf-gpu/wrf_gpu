# Release notes — wrf_gpu v0.18.2

**v0.18.2 is a memory-efficiency patch, not a speed release.** It makes the
measured all-island 1 km nested path fit on the reference single RTX 5090 while
leaving default numerics unchanged and bit-identical.

> **Honest performance framing — read this first.** The AC1_FIT 9/3/1 nested case
> now runs a warm-cache 1-forecast-hour pass on the reference 32 GiB RTX 5090 at
> **18.1 GiB peak VRAM**. Warm steady-state utilization is high
> (**~88% mean**, **~85% of samples >=80%**, **~6% idle**, p95 100%), but the
> **full-run aggregate is ~66%** because it includes the one-time ~81 s domain-load
> + cold-JIT compile prefix. That prefix amortizes toward zero over a real
> multi-hour forecast. The warm forecast-only speed is **~1734 s/forecast-hour**;
> this release fixes VRAM fit and utilization diagnosis, not single-card speed.

## What v0.18.2 delivers

### 1. 1 km nested AC1_FIT fits one reference RTX 5090 (MEASURED)

The measured case is AC1_FIT 9/3/1 nested, d03 = 520x280x45 (~145k columns),
fp64, mp8 Thompson / MYNN / Noah-MP / RRTMG, on the reference RTX 5090 (32 GiB).

| Before v0.18.2 | After v0.18.2 |
|---|---|
| OOM near **31.8/32 GiB**, including a failed **12.72 GiB** contiguous-arena request and recurring **2.09 GiB** allocations. | Warm-cache 1-forecast-hour run **PASS**, all 3 domains ended on the radiation gate, all finite, **18.1 GiB peak VRAM**. |

The 18.1 GiB figure is cross-confirmed by the worker run (**18.34 GiB**) and the
manager demo (**18.10 GiB**). It leaves roughly **14 GiB** of headroom on the
reference 32 GiB card.

Root cause: the failed peak was transient radiation working memory plus a one-time
MYNN cold-start dense-BouLac buffer, **not** persistent State, which is about
2.5 GiB.

### 2. Bit-identical algorithmic memory fix

The fix realizes the previously projected algorithmic VRAM lever:

- RRTMG longwave and shortwave column-tile defaults: **16384 -> 2048**.
- MYNN cold-start dense BouLac initialization: tiled over the production MYNN
  column width.
- d03-scale optional `jnp.max` host-sync metadata reductions: gated off where they
  were unconditional.

Default numerics are unchanged:

- `wrfout` exact comparison: **26/26 fields exact**, `max_abs_diff=0.0`.
- MYNN cold-start identity: `qke` diff **0.0**, `pblh` diff **0.0**.
- Switzerland d01 remains one radiation tile (1764 columns); forecast-only wall
  was **24.15 s** old default vs **23.94 s** new default, so the small-case gate
  did not regress.

### 3. Corrected utilization diagnosis (MEASURED)

The earlier "GPU mostly idle" reading was cold compilation and domain loading, not
the warm steady state. On the nested 1 km path:

| Window | Utilization |
|---|---|
| Warm steady state after one-time load+compile | **~88% mean**, **~85% of samples >=80%**, **~6% idle**, p95 100% |
| Full run including one-time ~81 s domain-load + cold JIT compile | **~66% aggregate** |

The path is compute-bound once warm. The measured warm forecast-only wall is
**~1734 s/forecast-hour**.

### 4. Restored runtime fixtures

`data/fixtures` again includes the Thompson aerosol and cold-collection runtime
tables. Source-only trees that include `src` + `data/fixtures` can import the
Thompson path without needing private local data.

## Scope and non-claims

- **No numerics change.** The default path is bit-identical; no tolerance was
  widened and no field was painted green by changing a gate.
- **No speedup claim.** v0.18.2 is a VRAM-fit and utilization-diagnosis patch.
  The warm forecast-only speed is measured at ~1734 s/forecast-hour for the
  AC1_FIT nested 1 km case.
- **MEASURED scope:** AC1_FIT 9/3/1 nested, d03 520x280x45 (~145k columns), fp64,
  mp8 Thompson / MYNN / Noah-MP / RRTMG, reference RTX 5090 (32 GiB).
- **PROJECTED scope:** any extrapolation to other GPUs, longer runs, other nested
  geometries, or multi-GPU throughput is projected until measured separately.

## Proof objects

- Fix report: `proofs/v018/oom_fix/fix_report.md`
- Root-cause report: `proofs/v018/oom_rootcause/rootcause.md`
- Exact-output compare: `proofs/v018/oom_fix/bit_identity_recheck_compare.json`
- Warm AC1_FIT run: `proofs/v018/oom_fix/real_ac1fit_halfhour_tiled_warm.json`
- Manager demo cross-check: `proofs/v018/oom_fix/v0182_manager_demo_inner.json`
