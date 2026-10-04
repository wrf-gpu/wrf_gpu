# Compile-efficiency finding — de-fuse is a RAM win, NOT a wall win (2026-06-25)

## Measured (9-nest Canary 20240901, de-fuse cold compile, integration branch b1164c0d)
- **De-fuse cold compile: ~51 min+ and still going** (no first output at 51:44), GPU 0% (CPU-bound XLA lowering),
  8 sequential `jit__advance_chunk_fori` per-domain "very slow compile" passes (~2.5 min each).
- **Peak host-RAM (tree, incl. parallel-compile workers): ~25.4 GB.**
- Fused baseline: ~40-60 min / ~60 GB (operational, not yet freshly measured this cycle).

## Verdict
- **RAM: de-fuse WINS** — ~25 GB vs fused ~60 GB ≈ **2.4× leaner** (per-domain peak << all-9-fused peak). This
  is real and avoids the compile-OOM that contributed to the B200 trouble.
- **WALL: de-fuse does NOT win** — it compiles 9 separate full-physics domain bodies **sequentially**, summing
  to ~the fused wall (or worse). The fused path compiles ONE module; de-fuse compiles 9 → more total compile work.

## Root of the missing wall win
B2 delivered de-fuse (per-domain modules) + parallel-compile (INTRA-module). It did NOT wire **cross-domain
parallel compilation** — compiling the 9 independent de-fuse modules CONCURRENTLY. With that, cold wall ≈
max(one domain body) ~5-6 min instead of Σ(9 bodies) ~50 min ⇒ the ~K× "extreme" speedup the de-fuse was
supposed to enable. The de-fuse already produces 9 independent modules, so parallelizing them is the natural
next step. **This is the missing piece, and the candidate for the real compile-wall headline.**

## What actually fixes the B200 "compile too long" (the user's core concern)
The decisive fix is NOT a faster cold compile — it's **compile ONCE, ship the warm cache, the paid pod
warm-starts in SECONDS** (B1 version-keyed cache + B2 prewarm artifact + #114 cross-case date-blind cache).
Cold ~50 min is paid once (locally/cheap pod); every subsequent run is a cache hit. This is the user's own gate
condition (cross-case warm cache) and must be PROVEN (Run 3: 20240403 same-grid warm-start = seconds).

## Decision for the user (pending)
- Pull **cross-domain parallel compile** into v0.21.0 (delivers the real cold-compile speedup, +timeline) — OR
  ship v0.21.0 with (de-fuse RAM win + warm-cache = the B200 fix) and make cross-domain-parallel-compile the
  **v0.21.1 compile headline**.
- Honest framing for v0.21.0 release notes: stability (dycore mechanism fix + finite detector + steep gate) +
  cache-just-works/warm-cache + de-fuse RAM −2.4×; the cold-compile WALL is only modestly changed (cross-domain
  parallel compile is the follow-up that makes it dramatic).

## Also surfaced
- #115 (GPUWRF_WRF_ROOT not honored for .TBL lookups) — needed a `data/wrf_pristine` symlink to run; portability item.
