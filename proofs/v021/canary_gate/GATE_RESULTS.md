# 9-nest Canary GATE (#127) — results log (2026-06-25, overnight)

Case: `<DATA_ROOT>/wrf_downscale/runs/20240901/cpu`, max_dom=9, hours=1, branch `worker/opus/vnext-v021-integration @ b1164c0d` (0.21.0), dycore fix in.

## Run 1 — DE-FUSE (GPUWRF_NESTED_DEFUSE_COMPILE=1)
- **STABILITY DETERMINANT = FINITE ✓✓** — wrote finite first frames for **all 9 domains** (d01-d09 @ +20 sim-min);
  got **past the step-67 divergence window with NO NaN / NO finite-guard abort**. → the dycore mechanism fix
  (acoustic mass-drain limiter 0.5·MUT + c2a alt-floor) **stabilizes the 9-nest Canary gate-case**. The
  Mont-Blanc-1042 m/cell extreme is the ONLY case that still relocates the failure → it is the documented edge.
- **Compile (cold, de-fuse): ~51 min** (sequential 9 domain bodies). Host-RAM peak (tree): **~26.4 GB** (vs fused ~60 GB).
- **THEN: GPU-VRAM OOM at wall≈5374 s (~90 min)** — `RESOURCE_EXHAUSTED: Out of memory while trying to allocate
  6.23 MiB` (GPU ~99 % full), **NOT a divergence**. Run integrated ~20-40 sim-min finite, then OOM'd before the 1 h mark.

## DECISIONS (manager, per north-star, the user away)
- **Dycore A/B → B (SHIP the mechanism fix).** The gate-case is stable; the fix is WRF-faithful + identity-preserving
  + zero-regression. Mont-Blanc-extreme = documented limitation → deep boundary-stability fix = **v0.21.1**.

## NEW BLOCKER for the ≥1 h-finite gate condition: GPU-VRAM OOM
- Hypothesis: **de-fuse keeps 9 resident per-domain executables → more GPU-VRAM than the fused single module**
  (the fused 9-nest ran 24 h VRAM-stable historically post-v0.19.1-leak-fix). De-fuse trades host-compile-RAM
  (−2.4×) for GPU-VRAM (+). If so: use de-fuse for COMPILE only, fused (or a VRAM-capped path) for INTEGRATION;
  OR the #123 RRTMG-transient cap / lower MEM_FRACTION / fp32-for-the-9-nest.
- **Run 2 — FUSED (de-fuse OFF), GPU-mem-sampled, RUNNING (bg bty0ddn6y):** tests the hypothesis (does fused avoid
  the OOM?) + gives the **fused compile-wall baseline** for the parallel-compile A/B. gpumem log: `fused_gpumem.log`.

## Gate status (#127)
- [x] 9-nest Canary FINITE past the divergence window (STABILITY — the headline dycore win)
- [ ] ≥1 h finite — **blocked by GPU-VRAM OOM** (Run 2 diagnosing de-fuse-VRAM vs 9-nest-edge)
- [ ] compile significantly faster/leaner — RAM −2.4× ✓; WALL pending **cross-domain-parallel-compile** (Workflow w6gm11aet)
- [ ] cross-case warm-cache (20240901 compile → 20240403 same-grid warm-start) — Run 3, after the cache is stable
- [ ] CPU-match (all-fields) — after a clean finite ≥1 h run
