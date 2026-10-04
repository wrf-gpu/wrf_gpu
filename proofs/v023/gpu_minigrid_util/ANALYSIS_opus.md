# ANALYSIS — Max GPU utilization for a MINI-GRID Tenerife forecast on one RTX 5090, minimal kernel invasion

**Author:** Opus 4.8 (analyst). **Date:** 2026-06-30. **Type:** architecture/perf ANALYSIS ONLY — no GPU run, no GPU lock, no source edit. Grounded in the real gpuwrf code + documented perf history + first principles.

**Question:** how to maximize GPU throughput (forecast-days per GPU-hour) for a 2-nest 3 km→1 km Tenerife mini-grid over many INDEPENDENT init-days, approaching the 5090 large-grid best-case, with MINIMAL invasion into the fp64 dycore kernels.

**One-line answer:** **Batch B independent same-geometry Tenerife cases on a leading axis `B` and run them through the existing fused cascade via `jax.vmap` over the carry — kernel-invasion NONE (the dycore body is untouched; only the orchestration shell is wrapped), complexity M, ~6–12× throughput at B≈8–16, reaching ~55–75% of large-grid 5090 best-case.** The v0.19 vmap dead-end does **not** recur (it was caused by ragged/heterogeneous shapes; this batching is perfectly homogeneous).

---

## 0. Grounding: what the real code says (cited)

### 0.1 The step is column-physics-dominated and the column physics is embarrassingly parallel
- Measured split (`.agent/decisions/V017-PERFORMANCE-RELEASE-PLAN.md:27`): **dycore = 25–38% of the step (d01 33% / d02 38% / 1 km 25%); physics dominates at 62–75% and grows at scale.**
- The hot column physics already runs as a `lax.scan` over **fixed-size leading-column tiles** with **no inter-column coupling**:
  - RRTMG-LW/SW: `src/gpuwrf/physics/rrtmg_lw.py:566-577`, `rrtmg_sw.py:114-123,401` — `_LW/_SW_COLUMN_TILE_COLS` default **1024**, "runs the kernel over fixed-size column tiles … reshapes outputs back."
  - Thompson: `src/gpuwrf/physics/thompson_column.py:170-181,2154-2173` — "No Thompson op couples columns, so the tiled result is value-identical per column"; tiling "caps the per-step transient to ONE tile" and **engages only when the flattened column count exceeds one tile.**
- **Consequence (the central enabler):** the per-kernel work for a mini-grid (d01 129²≈16.6 k cols, d02 255²≈65 k cols) is small → the SMs are starved and the launch/tile-loop overhead dominates. Adding a batch axis `B` multiplies the column count `B×`, which is *exactly* the dimension XLA maps onto SMs for these column-tiled kernels. Batching **manufactures the scale** the 5090 needs.

### 0.2 The state pytree is a clean SoA — a leading batch axis is structurally trivial
- `src/gpuwrf/contracts/state.py:448-976`: `State` is a flat slot-based SoA pytree; `tree_flatten` returns one array per `__slots__` field, `tree_unflatten` writes them straight back (the identity, `state.py:942-976`). No op in `State` assumes rank: `replace` is field-wise, `bytes()` sums leaves. **A leading `(B, …)` axis on every leaf is a valid pytree of the same treedef.**
- The advance body is `_advance_chunk_fori` → `jax.lax.fori_loop(body)` over `_physics_boundary_step` (`operational_mode.py:5290-5331`). The body takes the carry, a static `namelist`, and traced scalars `start_step`/`n_steps`/`cadence`. It contains **no Python-level branching on traced array values, no `.item()`, no host callback** — radiation gating is the traced predicate `jnp.equal(jnp.mod(step_index, cadence), 0)` (`operational_mode.py:5317`). This is the property `vmap` needs: the program is a pure function of the carry with batch-invariant control flow.

### 0.3 The fused cascade already exists and is the default; it calls `_advance_chunk` per domain
- `domain_tree.py:1591-1620` (`fused_jit`): one `@jax.jit` advances the parent one step, builds each child boundary package, advances each child `parent_grid_ratio` steps — **all in one XLA module**, default-on (`_nested_fuse_default_enabled`, `domain_tree.py:1914-1952`). For Tenerife (d01→d02, one child) the cascade is `parent advance + 1 boundary build + child×3 advance`.
- The host loop (`nested_pipeline.py:1104-1142`) drives this one output-interval at a time; the v0.17 async per-root-step sync (`block_between=False`, `root_sync_cadence=1`) is already the default (`nested_pipeline.py:875-910`), so the host no longer blocks after every domain advance.

### 0.4 AOT cheap-key already keys on shape/dtype — a batch axis is a free cache determinant
- `aot_cheap_key.carry_aval_hash` (`aot_cheap_key.py:786-852`) hashes each leaf's `(shape, dtype, weak_type, placement_class)`. A leading `B` changes every leaf shape `(…) → (B, …)`, so **`B` is automatically part of `program_key`** → distinct `B` ⇒ distinct blob, **one compile per B value**, then warm-loaded forever. No new cache machinery needed; the existing fused cascade serialization (`domain_tree.py:1658-1722`) captures the batched executable under its cheap key.

### 0.5 VRAM budget (measured, for max-B)
- Persistent state per Tenerife case (from `proofs/v022/nest_dycore/mb_native_jit_fixed_2h/proof/nested_pipeline_run.json`, the matching d01 129²/d02 255² geometry): **d01 ≈ 0.126 GB + d02 ≈ 0.391 GB ≈ 0.52 GB/case** (fp64). Plus scratch carry (`*_save` WRF aliases, tendencies) ≈ another ~0.5–1.0×, call it **~1.0 GB persistent+scratch per case**.
- The dominant transient is the **RRTMG g-point radiation transient ~8–9 GiB on the d02 grid** (`nested_pipeline.py:949-954,1066`). Crucially this is **tile-capped at 1024 columns** (§0.1): the transient is ~fixed per *tile*, and the scan reuses one tile's working set. Under batching, the flattened column count becomes `B × ncol`, which simply means **more tiles in the same scan** — the *peak* transient stays ~one-tile-sized, it does not multiply by `B` (the whole reason tiling exists, `thompson_column.py:2164-2173`). This is the single most important VRAM fact: **batching does NOT multiply the 8–9 GiB radiation peak.**

---

## 1. Ranked table

| # | Approach | Kernel-invasion | GPU-util gain | VRAM cost & max B @ 32 GB | Throughput estimate | Bit-identity risk | Other risks |
|---|----------|-----------------|---------------|----------------------------|---------------------|-------------------|-------------|
| **1** | **Batched ensemble — `vmap` leading axis B over the fused cascade (B homogeneous Tenerife cases)** | **NONE** (dycore body untouched; wrap the orchestration shell only) | **High** — `B×` columns/kernel fills SMs + amortizes tile-loop & launch overhead across B | ~1.0 GB/case persistent+scratch; radiation transient tile-capped (does NOT ×B). **max B ≈ 12–18** (state-bound), keep **B≈8–16** for transient headroom | **~6–12×** vs B=1 at B≈8–16 → **~55–75% of large-grid best-case** | **Low** (per-case): `vmap` is per-element; no cross-case reduction/halo. Each lane is the identical program. | Compile-time/RAM grows with the batched HLO; one recompile per distinct B; output writer must de-batch B wrfouts |
| 2 | Concurrent CUDA streams / N XLA executables overlapped (one program, B async dispatches) | NONE | Low–Med — overlaps host gaps & launch latency but each kernel still SM-starved; XLA serializes on one stream by default | ~B × full single-case peak incl. **B × 8–9 GiB radiation** → max B ≈ 2–3 | ~1.3–2× | None (separate executables) | Radiation transient ×B → OOM fast; little occupancy gain (kernels stay tiny) |
| 3 | Multi-process + CUDA MPS (N gpuwrf processes share GPU) | NONE | Med — MPS interleaves N processes' kernels, fills SMs via concurrency | ~N × full peak incl. **N × 8–9 GiB radiation**, no shared arena → max N ≈ 2–3 | ~1.5–2.5× | None (fully isolated processes) | Each process re-pays compile/AOT-load; radiation ×N OOM; no shared cache arena; MPS setup/ops burden |
| 4 | Spatial stitching — pack B grids into one big grid w/ buffer borders | **Moderate** (halo masking, seam handling, boundary re-wiring) | Med–High (one big grid does fill SMs) | Buffer borders waste ~15–30% cells; **seam halos cross cases** | ~3–6× but lossy | **HIGH** — stencil halos contaminate neighbors at seams; needs masking/guard zones = exactly the "no shortcuts/clamps" the project forbids | Complex, wasteful, bit-identity not provable without wide guard bands |
| 5 | Larger d01 to amortize (coarsen less / widen parent) | Minimal (config) | Low | grows single-case VRAM | ~1.1–1.4× | Low | Changes the science product (not the same forecast); doesn't address the core launch-bound problem |
| 6 | Persistent/megakernel (Pallas) for the column block | **Deep** | Med (single-case) | n/a | target ~1.3–2× single-case | Med–High (hand kernel vs fp64 reference) | Explicitly the v0.17 "default-off spike"; huge effort; violates "do not rewrite the dycore"; orthogonal to batching |

---

## 2. WINNER — Approach 1: batched-ensemble `vmap` over the fused cascade

**Why it wins on util-gain ÷ kernel-invasion:**

1. **Zero kernel invasion.** The fp64 dycore body (`_physics_boundary_step`, `_advance_chunk_fori`, all of `dynamics/` and `physics/`) is **not touched**. `vmap` is applied at the orchestration shell — it traces the *same* program with a batch dimension and lets XLA add the leading axis to every op. The body never learns it is batched.
2. **It directly attacks the proven bottleneck.** The mini-grid is launch/occupancy-bound (`proofs/v017/gpu_utilization_investigation.md`: ~1% util de-fused, 50–96% fused but still SM-starved on tiny grids). The step is 62–75% column physics that is *embarrassingly parallel over columns with no inter-column coupling* (§0.1). Adding `B` multiplies the column count `B×` on the exact axis XLA maps to SMs → occupancy and arithmetic-intensity both rise, and the per-kernel launch + tile-loop overhead is amortized across `B`. This is the *one* lever that turns a small grid into a large effective grid without changing the science.
3. **The radiation VRAM ceiling does not multiply by B** (§0.5). Because RRTMG/Thompson are tile-capped at 1024 columns and tile via `lax.scan`, batching just adds more tiles to the same scan — the ~8–9 GiB transient peak stays ~one-tile-sized. Persistent state is the binding constraint (~1 GB/case), so **B≈8–16 fits comfortably in 32 GB**. This is what makes it dominate Approaches 2/3, whose radiation transient *does* scale with B/N and OOM at B≈2–3.
4. **Bit-identity is clean and provable.** `vmap` is per-element: lane `b` of every op reads only lane `b`. There is **no cross-case halo, reduction, or seam** (unlike spatial stitching, Approach 4). The only thing to prove is that no op secretly reduces across the leading axis (none should: all reductions in the dycore/physics are spatial within a case). Per-case output is therefore bit-identical to the standalone run, satisfying the brief's hard constraint and the project "no masking/clamps/self-compares" rule.
5. **AOT/cache is free** (§0.4). `B` rides into `carry_aval_hash` as a shape determinant, so the existing cheap-key manifest serializes/loads the batched executable with no new code — one compile per `B`, warm thereafter.

**Why the v0.19 dead-end does NOT apply (the key reconciliation):**

The v0.19 result — `vmap` leaf-batching is **2.5× SLOWER** — is real and documented (`.agent/decisions/VERSION-SPRINT-LEDGER.md:243-245`; `v0192-efficiency-roadmap.md:34-38`; `V022-ROADMAP.md:51-52`). Its **mechanism was ragged/heterogeneous shapes**: it batched the 7 *different-shaped* nested LEAVES (d03–d09, down to 39×39) **within one nest**. The ledger is explicit: *"the 7 leaves are NOT same-shape"* and *"pad to the largest domain → wasted compute → perf loss."* `vmap` over heterogeneous shapes forces padding to the bounding box, so every lane computes mostly-masked tiny 39×39 stencils → occupancy collapse + arithmetic waste → 2.5× slower. The adopted alternative was **sequential bucket-fusion** (one program per same-shape bucket), *not* `vmap`.

The Tenerife problem is the **opposite structurally**: B cases of **identical geometry** (same d01 129², same d02 255², same dt, same nz) differing only in **initial conditions (different init-days)**. The leading batch axis is **perfectly homogeneous — zero padding, zero ragging, zero masking, no inter-case coupling.** Every lane does full, useful, identical-shape work. The 2.5× regression came entirely from the padding/occupancy pathology that homogeneous batching does not have. The brief's hypothesis is correct: the dead-end is a heterogeneous-shape artifact and **does not recur here.** (It *would* recur if someone tried to `vmap` d01 and d02 together — different shapes — so the batch axis must be added *per domain*, see §3.)

---

## 3. Implementation sketch (minimal-invasion diff shape)

**Principle:** batch *within a domain* (homogeneous), never *across domains* (heterogeneous → the v0.19 trap). The fused cascade already advances each domain with its own shapes; we batch each `_advance_chunk` call and each boundary build over the leading axis.

### Where the batch axis enters
- **State construction / loader.** In `nested_pipeline._load_domains` (`nested_pipeline.py:390-601`), build B per-domain carries (one per init-day) and **stack them on a new leading axis** per leaf: `jax.tree.map(lambda *xs: jnp.stack(xs), *carries_b)`. The static `namelist`, `grid`, `metrics`, and the precomputed `NestForceWeights` are **shared (un-batched)** across cases (same geometry) — they stay as closed-over statics, exactly as today. Only the **prognostic carry** gets the `B` axis.
- **The fused cascade body.** In `_build_fused_cascade_program` (`domain_tree.py:1554-1620`), wrap the two compute primitives in `vmap` over the carry axis only:
  - parent advance: `jax.vmap(_advance_chunk, in_axes=(0, None, None, None), out_axes=0)` — carry batched, `namelist`/`start`/`clock_base` broadcast.
  - boundary build: `jax.vmap(build_child_boundary_package, in_axes=(0, 0, None), …)` — child state and parent state batched, weights shared.
  - child advance: same `vmap` shape as the parent.
  The cascade's parent/child structure, ordering, and the `@jax.jit` wrapper are unchanged; only the two inner calls gain a `vmap`. **This is the entire numerical-path change and it does not enter any kernel** — `vmap` rewrites the *caller*, XLA adds the axis to the *callee*.
- **`start_step` / `clock_base` are shared.** All B cases advance the same step indices (same forecast length/cadence), so `start_step`, `n_steps`, `cadence`, and the traced `clock_base` are **batch-invariant** and broadcast (`in_axes=None`). The per-case *date* differences are already date-blind in the compiled HLO (#91/#114: dates ride in the traced `clock_base`/`_DateClockAux` sentinel — `aot_cheap_key.py:48-52,186-189`), so **one compiled program serves all B init-days** and the date does not fragment the cache. (If radiation cosz/solar geometry must differ per case by date, that enters via batched `clock_base` leaves — still a clean batched input, still one program.)

### Host-orchestration / output changes
- The segmented host loop (`nested_pipeline.py:1104-1142`) is **structurally unchanged** — it still drives one output interval per call and syncs per segment. The carries it threads are now batched; `result.states` leaves are `(B, …)`.
- **Output writer** (`_PerDomainWrfoutWriter`, `nested_pipeline.py:688-837`) must **de-batch**: loop `b in range(B)`, slice each leaf `leaf[b]`, and write `wrfout_<domain>_<case_b>_<valid_time>`. The `AsyncWrfoutWriter` already overlaps the write; submitting B payloads per output group is a small loop. The finite guard (`assert_state_finite_at_boundary`) runs once on the batched leaf (NaN anywhere in `(B,…)` fails) — cheap.
- **Per-case independence guarantee.** Add a CPU unit gate: run case `j` standalone and as lane `j` of a `B=2` batch; assert leaf-wise bit-identity. This is the bit-identity proof object (no GPU needed for the structural part; the on-GPU confirm is one short A/B when the lock frees).

### AOT/cache implications
- `B` enters `carry_aval_hash` (`aot_cheap_key.py:786-852`) as a leaf-shape determinant → the batched cascade serializes under a `B`-specific cheap key automatically (`domain_tree.py:1658-1722`). **One cold compile per distinct B**, warm-loaded after. Choosing a **single fixed B** (e.g. 16) and padding the last partial batch with a discarded dummy case keeps it to **exactly one compiled program** for the whole campaign. Compile RAM rises with the batched HLO (the fused cascade already holds parent+child; ×B widens buffers) — measure, and if it stresses the cold compile, the de-fuse compile-RAM lever (`GPUWRF_NESTED_DEFUSE_COMPILE`) and the existing tile caps are the mitigations.

### Minimal-invasion summary
Two `vmap` wraps inside `_build_fused_cascade_program`, a stack at load time, a de-batch loop in the writer, and one unit gate. **No edit to `_advance_chunk_fori`, `_physics_boundary_step`, the dycore, or any physics module.** The dycore stays bit-identical fp64.

---

## 4. Complexity estimate — **M (medium sprint)** + top-3 risks

**Sizing rationale:** the numerical change is two `vmap` wraps (small), but the surrounding plumbing — batched loader, de-batching output, a fixed-B padding policy, the per-case bit-identity gate, and a compile-RAM/VRAM A/B at the chosen B — is real integration work across `nested_pipeline.py` + `domain_tree.py`. Not S (more than a config flip), not L (no kernel surgery, no new subsystem).

**Top-3 risks:**
1. **Compile-time / RAM blowup of the batched fused HLO.** The fused cascade already lowers parent+child into one module; ×B widens every buffer and may push cold-compile RAM/time up. *Mitigation:* pick the largest B that compiles within budget; reuse the AOT warm-load (compile once, run the whole campaign warm); fall back to the de-fuse compile-RAM lever if cold compile stresses host RAM.
2. **A hidden batch-coupling op breaks per-case independence (bit-identity trap).** If any op in the traced path reduces or broadcasts across what becomes the leading axis (e.g. a global `jnp.mean` over the wrong axis, or an allocator/RNG seed shared per call), lanes contaminate each other. *Mitigation:* the `B=2`-lane-vs-standalone bit-identity unit gate catches this deterministically before any production run; `vmap`'s axis discipline makes it unlikely, but it must be *proven* not assumed.
3. **VRAM ceiling lower than hoped if the radiation transient is not as tile-bounded as documented.** The 8–9 GiB transient is tile-capped *in principle* (§0.5), but if any radiation/diagnostic temp materializes the full `(B×ncol)` working set before tiling, peak VRAM jumps. *Mitigation:* start at B=4, watch peak VRAM, climb to B=8/12/16 with the existing preflight VRAM guard; the radiation tile-cap env (`GPUWRF_RRTMG_*_COLUMN_TILE_COLS`) can be tightened if needed (bit-identical).

---

## 5. How close to large-grid 5090 best-case? (quantitative, with assumptions)

**Model.** Let single-case wall be `T1` per forecast-interval, split into:
- `H` = residual host-bound / launch-overhead fraction that batching *amortizes* across B (it becomes `H/B` per case), and
- `C` = on-device compute that scales with work. On the mini-grid most kernels are **SM-starved**, so device time is dominated by under-occupied launches whose *throughput rises sub-linearly with B until SMs saturate*, then becomes ~linear (true compute).

**Assumptions (sourced):**
- Mini-grid is launch/occupancy-bound: fused util on tiny grids is partial and SM-starved (`proofs/v017/gpu_utilization_investigation.md`; brief states ~1% de-fuse / 50–96% fused but SM-idle). Estimate effective SM occupancy at B=1 ≈ **15–30%** of the 5090's large-grid best-case throughput-per-watt.
- d02 (255²≈65 k cols) reaches ~one radiation tile (1024) ×64 tiles; d01 (16.6 k cols) ≈16 tiles. The 5090 (128 SMs, large register/occupancy budget) saturates around **several ×10⁵ active columns**. So B≈8–16 pushes the effective column count from ~65 k toward ~0.5–1.0 M, into the saturating regime.
- Persistent state ~1 GB/case ⇒ **B≈12–18 state-bound**; radiation transient ~fixed (tile-capped) ⇒ not the binding limit; pick **B=8–16**.

**Estimate.**
- **Occupancy fill:** going from ~15–30% to ~70–90% effective occupancy is a **~3–5×** device-throughput gain on the column-physics 62–75% of the step.
- **Launch/host amortization:** the per-kernel launch + tile-loop + host-orchestration overhead drops ~`1/B`; on a launch-bound mini-grid this contributes another **~1.5–2.5×** at B≈8–16.
- **Combined (not naively multiplied — they overlap):** **~6–12× throughput (cases/GPU-hour) vs B=1 at B≈8–16.**
- **As % of large-grid 5090 best-case:** the large-grid case is the regime where the 5090 is already SM-saturated and beats CPU ~1.3–2.7× (`V017-PERFORMANCE-RELEASE-PLAN.md:29`). Batching the mini-grid to saturation recovers most of that efficiency but pays (a) a residual non-amortizable host fraction (segment syncs, output D2H, boundary builds between cascades) and (b) the fact that a 1 km nest's arithmetic intensity is intrinsically below a large single domain's. Net estimate: **~55–75% of the large-grid best-case throughput-per-GPU-hour.** It does **not** reach 100% — the irreducible host fraction (per-segment block, hourly wrfout D2H for B cases, the eager boundary build between fused cascades) and the nest's lower arithmetic intensity cap it. But it converts a **2.3×-slower-than-CPU** mini-grid (single-domain 129² launch-bound, `V017-PERFORMANCE-RELEASE-PLAN.md`) into a solid GPU win and multiplies training-days/GPU-hour by ~an order of magnitude.

**Sensitivity:** the dominant uncertainty is where SM saturation lands and the residual host fraction `H`. If `H` is larger than assumed (output D2H for B cases is heavy), the launch-amortization term shrinks and the realistic gain is the lower end (~6×, ~55%). A B=4 → B=8 → B=16 throughput-vs-VRAM sweep (one short GPU A/B when the lock frees) pins the knee precisely.

---

## 6. Honest failure modes

1. **Compile-RAM / cold-wall regression at high B.** The batched fused HLO could push cold-compile RAM/time beyond budget (the fused cascade is already the compile-heavy path). If so, B is capped *by compile*, not by VRAM, and the gain is smaller. The AOT warm-load (compile-once-per-B) is the saving grace, but the *first* compile of a large B could be painful on a 32 GB host. **Disappoints if:** B must be kept ≤4 to compile, giving only ~3–4×.
2. **Residual host-bound serial orchestration batching can't hide.** Per-segment `block_until_ready` (`nested_pipeline.py:1121`), the eager boundary build between cascades (it *is* inside `fused_jit`, good), and **B× output D2H + NetCDF writes per output group** are real host work that does not shrink with B. At large B the output-write fraction can dominate. **Disappoints if:** output is the new bottleneck — mitigate with the async writer and compact training-subset output (`GPUWRF_TRAINING_OUTPUT_SUBSET`), already available.
3. **VRAM ceiling lower than the tile-cap model predicts.** If a radiation/diagnostic temp materializes the full `(B×ncol)` set before the scan tiles it (a fusion-boundary surprise), peak VRAM jumps ~`B×` and OOMs at small B — collapsing the whole advantage. The documented tile-cap (§0.5) makes this unlikely for the forecast path, but the **M9 diagnostic** path (separate `jax.jit`, `operational_mode.py:5396-5414`) and the writer diagnostics must be checked for batch-aware tiling. **Disappoints if:** diagnostics force B≤3.
4. **A latent cross-case coupling op silently contaminates lanes.** Any shared RNG, any reduction over the wrong axis, or an allocator/aliasing bug (the `*_save` WRF scratch aliases state leaves — `_operational_feedback` note, `domain_tree.py:2033-2049`) could leak across lanes. **This is the one correctness risk.** The `B=2`-lane-vs-standalone bit-identity gate is mandatory and must pass before any production data is generated — if it fails, the approach is blocked until the coupling is found (it would not be a "shortcut/clamp" fix; it must be a real per-lane-independence fix).
5. **Heterogeneity creep.** The whole win depends on every case sharing *exact* geometry/dt/nz/physics options. If the campaign mixes init-days that trigger different conditional state leaves (e.g. a hail-MP day vs a non-hail day → different `State` leaf set, `state.py:60-189`) the treedef diverges and `vmap` breaks / the cache fragments. **Disappoints if:** the case list is not pre-screened to a single physics/leaf configuration. *Mitigation:* freeze one namelist/MP option for the whole Tenerife campaign (it is one island, one config — natural).

---

## Appendix — why NOT the alternatives (one line each)
- **Streams (2) / MPS (3):** zero kernel-fill — the kernels stay tiny and SM-starved; and the **radiation transient scales with B/N → OOM at B≈2–3**. They overlap *gaps* but don't *fill SMs*; strictly weaker than batching, which does both.
- **Spatial stitching (4):** the only one with **HIGH bit-identity risk** — stencil halos cross case seams, requiring guard bands/masking that the project forbids and that can't be proven equivalent. Also wastes 15–30% on buffers. Strictly dominated by `vmap` (which fills SMs with *zero* seam contamination).
- **Larger d01 (5):** changes the science product, doesn't fix launch-boundness.
- **Megakernel/Pallas (6):** deep kernel surgery, violates "do not rewrite the dycore," and is orthogonal — it speeds a single case; batching speeds the *campaign*. If ever done, it composes *under* batching.
