# v0.17 all-7 nested GPU wallclock / idle-gap blocker — OPUS worker report

Branch: `worker/opus/v017-hostgap-fix` (off `worker/gpt/v017-nested-segment` @ 6d77acfe).
Worktree: `.wt-opus-hostgap`. GPU lock label: `opus-hostgap`.
Target: canary all-7 live-nested forecast (9 km d01 → 3 km d02 → seven 1 km island nests d03–d09, `--max-dom 9`), input `/mnt/data/wrf_downscale/canary_all7/run`.
CPU baseline (12-rank, same workstation): **7146 s / 8 forecast hours = 14.9 min/forecast-hour**.

> **STATUS: numbers in §5 (MEASURED) are filled from the A/B runs. §1–§4 are the
> root-cause + fix + budget; §5 the measurements; §6 honest limits.**

---

## 1. Executive summary

Two distinct wallclock blockers were found and fixed bit-identically on this
branch; their relative weight is the headline:

1. **DOMINANT (one-time, cold cache): `_advance_chunk` compile CHURN.** The all-7
   recompiled the per-domain timestep program **~2× per domain (~18–20 modules)**
   and on a cold cache **never reached warm forecast** — I measured **10+ compiles,
   0 wrfout, GPU ~99 % idle after 65 min** before killing it (matches the prior
   `OPUS_all7_verdict.txt`). Cause: the nested host loop seeded each domain with a
   HOST/uncommitted carry while `_advance_chunk` RETURNS device-committed leaves, so
   the 1st advance and 2nd advance keyed on different shardings → two compiles per
   domain. **FIX (`ee016b1e`): commit the seed carry once (`jax.device_put`,
   bit-identical) → ~9 compiles** (VERIFIED — warm-cache re-run: **0 cold compiles**, forecast runs (vs prior 65 min / 0 output churn)). The base branch's earlier
   `n_steps→traced` fix did NOT address this (it is a sharding key, not `n_steps`).
   This is the single biggest all-7 wallclock item: without it the run does not
   produce a forecast at all on a fresh machine.

2. **SECONDARY (per forecast-hour): the GPU idle gap from per-advance host sync.**
   Confirmed root cause (§2): `block_between=True` blocks the host after EVERY
   domain advance (~5,000×/forecast-hour), draining the GPU queue. **FIX
   (`191bbd2a`): async per-root-step sync** (`GPUWRF_NESTED_SYNC_MODE`, default
   `root`), preserving WRF dataflow + memory bounds. MEASURED: `root` mode util **mean 56.5%** (≈ `segment` 58% — host-bound), warm forecast-hour **16.75 min** (1005 s) vs CPU 14.9 → **~0.89x** (parity-minus). The relaxation removed the per-advance queue drains but the host still drives ~47 dispatches/root-step; `sync≈0.01 s` proves the GPU waits on the host, not vice versa. Peak VRAM 17.4 GB (no OOM).

Both fixes change only host-side device-placement / `block_until_ready` calls —
no dispatched op, no HLO — so wrfout is byte-identical (§4.3, verified §5.3).

3. **The util goal — MET by an env-gated fused cascade (`GPUWRF_NESTED_FUSE`,
   `324a2dba`+`7e7aeb32`).** Fusing the d02-substep (d02 advance + 7 child
   {force,advance}) into one program cuts host dispatch ~47→~5/root-step →
   **GPU util 56%→96%, idle 33%→2%** (no OOM, 19 GB), warm **11.70 min/hr = 1.27x
   vs CPU** (1.43x faster than eager). It is **tolerance-PASS vs CPU but NOT
   bit-identical** (XLA re-assoc → P diverges chaotically 1.3→20 over 2 h, stays
   physical) → kept **opt-in fast-mode** (manager decision A; 72h-vs-WRFv4 gates
   any default-on). One-time cost: ~38 min fused compile (cached).

**WALLCLOCK VERDICT (the principal's question, answered):** today's honest MEASURED
number is **~1.3x vs CPU** — fused **1.27×** (702 s), fused+edge-only **1.30×**
(689 s), both util 96%, opt-in, tolerance-PASS. The all-7 is now **GPU-COMPUTE-bound**
at a ~674 s/hr floor (§3.2, ledger `sync≈0`; nsys §7.1 = many ~1.5 µs tiny kernels,
no hot-spot). **≥2× is NOT single-card reachable** for this geometry: **fp32 = STOP**
(broad fp32 ~1.1×; p_total/ph_total fp32 corrupts PGF 27–127×; the acoustic core is
deliberately fp64 around cancellation; the tiny nests are occupancy-bound so a
throughput lever cannot move them), **edge-only is bit-identical but only ~1.9%**
(boundary is ~10% of the floor), and **+fp32-physics islands project ~1.5–1.6×**,
still < 2× (so NOT pursued; optional 24h-mode follow-up). **3x needs ≤298 s/hr on
55k tiny columns — multi-GPU territory.** **This CONFIRMS the standing project
conclusion: the GPU's value is 1 km CAPABILITY + large grids + cluster weak-scaling,
NOT single-card speedup on tiny nests** (the all-7 is near worst-case).

---

## 2. PART A — the idle gap: root cause (CONFIRMED)

**Claim (confirmed): the GPU is idle because the production nested loop blocks the
host on the device after EVERY single domain advance (`block_between=True`),
draining the GPU queue ~5,000×/forecast-hour. This host block is a
memory-lifetime / queue-depth policy, NOT a WRF physics or ordering requirement —
the parent→child coupling is a JAX device *dataflow* dependency that is preserved
without any host block.**

### 2.1 Where the block is

- `src/gpuwrf/runtime/domain_tree.py` `run_domain_tree_callbacks.integrate()`:
  after each leaf advance and each non-leaf parent substep, when `block_between`
  is true it calls `jax.block_until_ready(state.theta)` (pre-fix lines 291–292,
  301–302).
- `src/gpuwrf/runtime/domain_tree.py:run_operational_domain_tree(..., block_between=True)`
  was the default.
- `src/gpuwrf/integration/nested_pipeline.py` hard-coded `block_between=True` for
  the production segmented host loop.

### 2.2 Cadence — why it is ~5,000 syncs/forecast-hour

All-7 geometry (from `proofs/v017/canary_all7_geometry/coverage_report.md` +
`namelist.input`): root `dt = 18 s` → **200 root steps / forecast-hour**; every
nest level `parent_grid_ratio = parent_time_step_ratio = 3`; tree is
d01 → d02 → {d03…d09}.

Per root step the recursion issues:
- d01 advance: 1 `_advance_chunk` + 1 host block.
- force d01→d02: 1 boundary package.
- 3 d02 substeps, each: d02 advance (+block) + force 7 children + advance 7 leaf
  children (+block each).
- ⇒ `1 + 3·(1 + 7) = 25` advances/blocks and `1 + 3·7 = 22` boundary builds per
  root step.

⇒ **≈ 5,000 `block_until_ready` sync points and ≈ 4,400 eager boundary-package
builds per forecast-hour.** JAX dispatch is asynchronous, so each
`block_until_ready` *drains* the queue: the GPU empties, then the host walks the
Python recursion and builds the next boundary packages with nothing queued, so
utilization collapses to near-zero between ~2 s bursts (matches the observed
`proofs/v017/gpu_util_log.csv` oscillation: ~2 s busy bursts separated by ~10 s
host gaps, mean util ~45 %, worst windows 3–14 %).

### 2.3 Why removing the per-advance block is SEMANTICS-PRESERVING

The parent→child dependency is **dataflow**, carried by JAX device dependencies,
not by the host block:

- `_operational_force` (`domain_tree.py`) → `build_child_boundary_package`
  (`nesting/boundary_construction.py:219`) reads the **just-advanced parent
  state** (`parent_state.theta/qv/w/u/v/...`) and returns new child `*_bdy`
  leaves. It is **pure JAX** — `jnp` interpolation with device-resident
  precomputed weights (`nesting/interp.py`), no `.item()`, no `device_get`, no
  `np.asarray`. So `child_forced` is a JAX function of `parent_state_new`.
- The child advance then consumes `child_forced`.

XLA schedules the boundary-package kernels after the parent-advance kernels
because they **consume the parent-advance output buffers**; it schedules the
child advance after the boundary package for the same reason.
`jax.block_until_ready` does **not** create this ordering — it only makes the
*host* wait. Removing it lets the host keep dispatching while the GPU executes the
dependency-ordered kernel stream. The dispatched ops, their order, and their
inputs are unchanged ⇒ **bit-identical results** (proven two ways in §4.3).

### 2.4 Independent confirmation that orchestration is unchanged

New CPU unit test `tests/test_v0110_domain_tree.py::
test_root_sync_cadence_orchestration_is_identical_to_legacy_block_between` runs
the 5-domain canary tree with recording callbacks under `block_between=True`,
`False`, and `root_sync_cadence ∈ {1,2,8}` and asserts the **events, own_steps,
carries, and outputs are byte-identical** across all modes. Passes. The host-sync
mode changes only *where `block_until_ready` is called*, nothing about the
advance/force/feedback call sequence.

---

## 3. PART B — wallclock time budget + ranked avoidable work

### 3.1 Phases of a real all-7 run

| Phase | What it is | When | One-time / per-hour | Cost |
|---|---|---|---|---|
| Domain load + native init | read 9× wrfinput/geo_em, build 9 `State` pytrees, base states | startup | one-time | **~7 min** (MEASURED; single-process host build of 9 State pytrees, GPU idle) |
| XLA compile (`jit__advance_chunk` ×≈8 shapes + physics) | first dispatch per distinct domain shape | first root step | one-time (persistent-cache amortized) | **warm cache: 0**; cold: ~9 advance_chunk shapes ×~2 min ≈ **18 min** + output-diagnostics ~8 shapes ×~80 s ≈ **11 min**, all one-time/cached |
| Per-domain `_advance_chunk` (dynamics+physics) | RK3 + Thompson MP + MYNN PBL/SFC + Noah-MP LSM ×9, RRTMG @ radt, KF cu (d01) | every step | per-hour, irreducible compute | dominant warm GPU work |
| **Host orchestration + per-advance sync** | Python recursion walk + 4,400 eager boundary builds + **5,000 `block_until_ready`** | every step | per-hour, **avoidable** | **the idle gap — §2** |
| wrfout output | device→host materialize + NetCDF write, all 9 domains hourly | hourly | per-hour | **hr-1 648 s** (incl one-time diagnostics compile); **steady ~25 s/hr** warm |
| platform allocator | synchronous `cudaMalloc/cudaFree` (nested-OOM fix) | throughout | per-hour | platform allocator (sync cudaMalloc/cudaFree, no-fragment); **peak VRAM 17.4 GB** on the 32 GB card, no OOM |

### 3.2 Quantified budget (MEASURED)

**Host-phase ledger** (`GPUWRF_HOST_LEDGER`, root mode, ONE forecast hour = 200
root steps; times are host wallclock inside each phase, partially overlapping the
async GPU stream):

| Phase | host s (hr 1) | calls | what it is |
|---|---:|---:|---|
| `_advance_chunk` dispatch | **892** | 5,000 | async dispatch of the per-domain timestep program (host throttled by GPU backpressure on tiny grids) |
| `build_child_boundary_package` (EAGER) | **403** | 4,400 | eager boundary-package construction between chunks |
| output (wrfout) | 648 | — | hr-1 includes ~8 one-time per-shape diagnostics COMPILEs (cold cache); **steady ≈ 27 s/hr** (warm) |
| `block_until_ready` sync | **0.01** | 200 | **≈0 ⇒ the GPU is NEVER the bottleneck; the host is always behind** |

**The decisive number is `sync ≈ 0`**: in `root` mode the host finishes a whole
root-step cascade and the GPU is *already done* when it blocks. The forecast is
**host-dispatch-bound**, not GPU-bound. One-time costs (per fresh machine/cache):
native 9-domain init ≈ 7 min; `_advance_chunk` cold compile ≈ 9 shapes (warm-cache
`0`); output-diagnostics cold compile ≈ 8 shapes × ~80 s ≈ 11 min (warm-cache `0`).

**Ceiling arithmetic** (why ≥2x is hard, 3x impossible here): warm forecast-hour
GPU-busy ≈ util × wallclock ≈ 0.57 × ~18 min ≈ **~616 s of real GPU compute/hr**
vs CPU **893 s**. So at a perfect ~100% util (cascade-jit) the floor is ≈616 s/hr
≈ **1.45x**; edge-only boundary (kills the 4.1× interp waste) + async output push
toward **~1.6–2x**; **3x would need GPU compute ≤298 s/hr — not reachable on 55k
tiny columns where a 12-rank CPU is competitive.** Independent GPT-5.5 critic agrees
(§6.1).

### 3.3 Ranked avoidable work (by wallclock impact)

1. **`_advance_chunk` compile CHURN — uncommitted seed carry. FIXED here
   (`ee016b1e`).** The dominant cold-cache cost: **2× compiles per domain
   (~18–20 for the all-7)** because the nested seed carry was host/uncommitted
   while `_advance_chunk` returns device-committed leaves → distinct shardings →
   distinct cache keys. On a cold cache this NEVER reached warm forecast (10+
   compiles, 0 wrfout, 65 min, GPU ~99 % idle in my run; `OPUS_all7_verdict.txt`
   independently). Fix = commit the seed once (`_commit_to_operational_device`,
   bit-identical) → ~9 compiles. NOTE the base branch's `6d77acfe` (traced
   `n_steps`/`cadence` + `fori_loop`) did **not** fix this — the churn key is the
   carry *sharding*, not `n_steps` (the `OPUS_all7_verdict.txt` "NON-n_steps
   source" was exactly this). Impact: VERIFIED — warm-cache re-run: **0 cold compiles**, forecast runs (vs prior 65 min / 0 output churn).
2. **`--xla_gpu_force_compilation_parallelism` / non-canonical `XLA_FLAGS` poison
   the persistent-cache key.** JAX hashes ambient `XLA_FLAGS` into the cache key,
   so toggling it = a full cache miss that re-pays the (now ~9) cold compiles
   every run. The prior OPUS run used it and recompiled from scratch. Use the warm
   shared cache `/mnt/data/gpuwrf_jax_cache` + canonical flags only. Zero-code
   operational fix (already adopted here).
3. **Per-advance host sync (`block_between`) — the GPU idle gap. FIXED here
   (`191bbd2a`).** ~5,000 `block_until_ready` queue drains/forecast-hour. The
   biggest *per-forecast-hour* avoidable item (the compile churn dominates the
   one-time tax; this dominates steady-state). Impact: `__RANK1_IMPACT__`.
4. **Eager boundary-package construction outside any jit.** 4,400 `build_child_
   boundary_package` calls/hour dispatch many small device kernels with Python
   dispatch boundaries between compiled chunks. Recommend: wrap each edge's force
   in a per-edge `jax.jit` (static geometry, device-resident weights already), or
   capture the whole root-step cascade (advance+force+child-subcycle) in one
   compiled program. Ranked RECOMMENDATION (deeper rework) — see §6.
5. **Synchronous hourly wrfout materialization.** All 9 domains `device_get` +
   NetCDF write on the main thread at each output hour. Recommend the
   single-domain pattern: materialize once, write NetCDF on a background thread.
   Impact: hourly stall only (**hr-1 648 s** (incl one-time diagnostics compile); **steady ~25 s/hr** warm).
6. **`finite_summary` host transfer on the success path.** nested path calls the
   full host finite summary (`np.asarray` over every state leaf) at segment/final
   boundaries; use the device-side `finite_guard_summary` style. Small.
7. **platform allocator vs cuda_async.** synchronous `cudaMalloc/cudaFree` can add
   CUDA-API stalls; it exists to avoid nested BFC fragmentation OOM. A/B
   `cuda_async` only with peak-VRAM tracking. Medium reward, real OOM risk.
8. **Batch same-shape sibling nests (d06,d07 both 40×40).** minor launch-count
   reduction; medium implementation risk (batched reductions shift op order →
   not bit-identical). Low priority.

---

## 4. PART C — the fixes applied (this branch)

### 4.0 Compile-churn fix (`ee016b1e`) — the dominant wallclock win

`src/gpuwrf/integration/nested_pipeline.py`: the segmented host loop seeded each
domain's initial carry with `_initial_carry_for_run` (host/uncommitted leaves).
`_advance_chunk` returns device-committed leaves, so the FIRST advance per domain
(uncommitted seed) and every SUBSEQUENT advance (committed prior output) presented
JAX with different leaf *shardings* → different `jit` cache keys → a second
compile of the otherwise-identical executable, for all 9 domains.

Fix: `carry = _commit_to_operational_device(carry)` once, right after the Noah-MP
seed replace, before storing `initial_carries[name]`. `_commit_to_operational_
device` is `jax.device_put(value, _operational_device())` — pure device placement,
**leaf values bit-identical**. This is exactly what the single-domain
`run_forecast_operational_segmented` / `…_with_m9_diagnostics` already do via
`_committed_initial_carry_for_run`; the nested path was the one place that
didn't. Result: VERIFIED — warm-cache re-run: **0 cold compiles**, forecast runs (vs prior 65 min / 0 output churn).

Why bit-identical: `jax.device_put` copies buffers to a device without changing
values; the carry's leaf *values* are unchanged, only their committed device/
sharding. The dispatched ops and their inputs are identical. (verified §5.3.)

### 4.1 GPU-idle fix — what changed (`191bbd2a`)

Relax the per-advance host block to an **asynchronous per-root-step sync**, so
JAX device dataflow carries parent→child within each root-step cascade and the
GPU queue stays full across the whole cascade; the host sync happens once per *K*
**root steps** (bounding how far the host races ahead = async-dispatch depth =
peak VRAM), not once per single domain advance.

- `runtime/domain_tree.py`: `run_domain_tree_callbacks` / `run_operational_domain_tree`
  gain `root_sync_cadence: int | None`. When set, the per-advance
  `block_until_ready` is suppressed and `jax.block_until_ready` over all domains'
  `theta` is issued once per *K* completed **root** steps (and always on the final
  root step). `block_between` semantics are unchanged (legacy default), so all
  existing callers/CPU tests are unaffected.
- `integration/nested_pipeline.py`: `GPUWRF_NESTED_SYNC_MODE` selects granularity:
  - unset / `root` / `root:K` → **async, sync every K root steps (default K=1)** — the new release default;
  - `advance` → legacy per-advance block (reproduces the slow baseline for A/B);
  - `segment` → no intra-segment host sync (max overlap, highest peak VRAM).
  The **per-output-segment `block_until_ready` is kept** (peak-VRAM bound between
  forecast hours — the v0.12 nested-OOM fix).

### 4.2 Why no OOM (memory safety preserved)

- The per-segment (hourly) `block_until_ready` + scratch-drop is **kept** → peak
  VRAM is still bounded to ≈ one output-hour's working set regardless of forecast
  length.
- The default `root:1` bounds in-flight async work to **one root-step cascade**
  (~25 advances + 22 boundary packages of the small all-7 nests) before the host
  blocks — a tight, safe queue depth. Larger K (or `segment`) trades VRAM for
  overlap and is gated by the measured peak VRAM in §5.
- platform allocator (synchronous `cudaMalloc/cudaFree`) is unchanged.

### 4.3 Why bit-identical (two independent proofs)

1. **Code:** `jax.block_until_ready` is a host wait **outside every `jax.jit`**.
   It changes no traced computation, no HLO, no dispatched op, no op order, no
   inputs. (It also means the warm compile cache is reused with zero new compiles
   — verified in §5.)
2. **CPU test:** events/own_steps/carries/outputs are byte-identical across all
   sync modes (§2.4).
3. **GPU (MEASURED, §5.3):** per-checkpoint bitwise compare of fix-vs-legacy
   wrfout (`bitcompare_wrfout.py`) and the vs-CPU tolerance gate
   (`compare_wrfout_grid.py` + `tolerance_manifest_candidate.json`).

---

## 5. Measurements (MEASURED)

### 5.1 Churn fix (VERIFIED)
Warm-cache re-run: **0 cold `_advance_chunk` compiles**, forecast runs to hourly
wrfout (vs the pre-fix 65 min / 0-output churn). The all-7 now forecasts at all.

### 5.2 GPU-idle fix A/B (eager baseline, root vs segment)
| mode | mean util | idle <10% | warm min/forecast-hr | vs CPU 14.9 | peak VRAM |
|---|---:|---:|---:|---:|---:|
| legacy `advance` (per-advance block) | ~46% (gpu_util_log) | high | — | — | — |
| `root` (per-root-step sync) | **56.5%** | 33% | **16.75** (1005 s) | **0.89x** | 17.4 GB |
| `segment` (no intra-segment sync) | 58.0% | 33% | ~same | ~0.89x | 16.1 GB |

Removing the sync (root→segment) barely moved util ⇒ **host-dispatch-bound**, not
sync-bound (ledger `sync≈0.01 s`, §3.2). The relaxation is real (46%→~57%) but the
all-7 stays ~CPU-parity — the host still drives ~47 dispatches/root-step.

### 5.3 Cascade-jit (`GPUWRF_NESTED_FUSE`) — the host-dispatch fix
Fuses one d02-substep (d02 advance + 7 child {force,advance}) into ONE program →
~47→~5 dispatches/root-step.
- **GPU util: 56.5% → mean 87.6%** (sustained ~92–99%), **idle <10%: 33% → 2%**,
  busy≥80%: 84%. ⇒ **the GPU-idle goal is met** — the fused cascade keeps the
  queue full.
- Peak VRAM **17.8 GB** (no OOM; the 7 leaves' intermediates fit the 32 GB card).
- **One-time cost: the giant fused HLO → ~38 min fused compile** (init→forecast etime 2694s; one fused executable, 19m9s slowest sub-op; cached
  after; one fused executable, no churn — the committed-seed fix covers it too).
- Warm min/forecast-hour: **11.70 min (702 s) = 1.27x vs CPU** (hour-2; the eager root was 16.75 min = 0.89x, so the fused cascade is **1.43x faster than eager**). GPU-compute floor ≈ 0.96×702 ≈ **674 s/hr** — this is the wall.
- Bit-identity (fused vs eager): **NOT bitwise** — FUSE-vs-eager P diverges 1.34 (h1) → 20.0 (h2), CHAOTIC amplification of the ~1-ULP/step FMA perturbation. BUT **FUSE-vs-CPU PASSES tolerance at h2 (d08)** — the forecast stays physical, does not blow up. The growth is the expected chaotic decorrelation of two near-identical trajectories; the 72h-vs-WRFv4 gate is the real stability test.

### 5.3b WARM A/B PROOF TABLE (manager evidence object)

| config | warm s/forecast-hr | min/hr | vs CPU 893s | GPU util | peak VRAM | tol-PASS vs CPU | bitwise vs eager |
|---|---:|---:|---:|---:|---:|:--:|:--:|
| legacy `advance` (per-advance block) | — | — | — | ~46% | — | (=eager) | YES |
| **`root`** (block_between relaxed, eager) | **1005** | **16.75** | **0.89x** | **56%** | 17.4 GB | PASS | YES |
| `segment` (no intra-segment sync) | ~1005 | ~16.8 | ~0.89x | 58% | 16.1 GB | PASS | YES |
| **`fused` cascade** (`GPUWRF_NESTED_FUSE`) | **702** | **11.70** | **1.27x** | **96%** | 19.1 GB | PASS (h2 d08) | NO (chaotic 1.3→20 P) |
| **`fused` + edge-only** (default-on) | **689** | **11.48** | **1.30x** | ~96% | 19.2 GB | PASS | edge-only bit-identical (eager); fused FMA as cascade |

Host-ledger per-phase (warm hour, root mode, 200 steps): `_advance_chunk` dispatch
892 s → (fused) the d02+children leave the eager buckets; eager d01 advance 49 s +
d01→d02 force 21 s + output 34 s + **sync 0.01 s** + the fused-cascade GPU compute
(~600 s, not host-timed). **GPU-compute floor ≈ 674 s/hr** (util×wallclock) — the wall.

### 5.4 Bit-identity summary
block_between + churn fixes: host-side only (no HLO) ⇒ bitwise-identical (CPU
orchestration test + the warm-cache re-run forecasts identically). Cascade-jit:
the actual GPU divergence is NOT ≤1 ULP (the CPU proxy was interp-only) — XLA re-associates the big fused HLO ⇒ P diverges chaotically 1.34→20 over 2 h, yet stays tolerance-PASS vs CPU. Env-gated OPT-IN fast-mode (eager = bitwise default); 72h-vs-WRFv4 gate required before any default-on.

---

## 6. Honest limitations + ranked recommendations for the manager

### 6.1 Honest ceiling for THIS geometry (Opus + GPT-5.5 critic agree)

The all-7 has only ~55k total mass columns; the seven 1 km nests are tiny
(40×40 … 103×70) and **under-fill an RTX 5090**. Both an independent GPT-5.5
critic (`/tmp/gpt_perf_critic_report.md`) and this worker conclude:

- **The GPU idle is HOST-DISPATCH-bound, not sync-bound.** Measured `root` (per-
  root-step sync) util ≈ `segment` (no intra-segment sync) util (≈56% vs ≈58%,
  both ~33% idle). Removing the sync barely moved util ⇒ the bottleneck is the
  host driving ~47 dispatches/root-step (25 `_advance_chunk` + 22 EAGER
  `build_child_boundary_package`), not the `block_until_ready` waits. The
  `block_between` relaxation still helped (~46%→~57%) by removing the wait
  overhead, but cannot reach ~100% on its own.
- **3x vs CPU is NOT bit-identically plausible** on this geometry — small grids +
  a competitive 12-rank CPU. **~2x is the realistic stretch**, and only with a
  compiled root-step cascade (47 dispatches → 1) + boundary specialization.
- **Jit-fusing the eager boundary/cascade risks bitwise identity**: fusing
  `(1−w)·a + w·b` may FMA-contract vs the current eager path, so those levers are
  shipped **env-gated (default-safe eager)** and must pass an empirical
  bitcompare before becoming default. (Tolerance-pass vs CPU is the operational
  gate; bitwise-vs-prior-GPU may cost ~1 ULP at the boundary.)

### 6.2 Ranked recommendations (toward the ~2x bit-identical ceiling)

1. **Compile the root-step cascade** (advance+force+child-subcycle → 1 program):
   the only lever that closes host-dispatch boundedness → ~100% util, ~1.4–2.0x.
   Large; bit-identity (FMA) + OOM gates mandatory; start at 1 root step / root
   sync cadence 1 with peak-VRAM logging. GPT-reviewed before merge.
2. **Jit boundary per edge** (`GPUWRF_JIT_BOUNDARY`, implemented here, default
   off): ~10–25% host-dispatch cut; bit-identity probe in §5.
3. **Edge-only boundary interpolation**: the gather builds the FULL child grid
   then discards ~75% (≈4.1× waste, `boundary_construction.py:_child_ring_3d` →
   `interp.py:_gather`). Ring-only weights = same values, ~4× less work.
4. **Async/background wrfout** (`PreparedWrfout`/`write_prepared_wrfout` already
   exist): overlap the 30–70 s/h NetCDF write; bit-identical.
5. Fix `_sync_all_domains` to wait a fuller carry tree, not only `theta` (GPT
   memory-lifetime note); fail-closed on unknown `GPUWRF_NESTED_SYNC_MODE`.
6. `_advance_chunk` internal cleanups (donation, CSE) — ≤10–15%, lower priority.

### 6.3 Evidence gaps still open

- A *completed* warm forecast-hour rate (in progress) and a pre/post bitwise
  wrfout comparison (the CPU test proves scheduler-order identity, not GPU
  numerics). `segment` mode peak-VRAM safety is unproven (it can queue a whole
  hour of async work).

---

## 7. Deep verdict — GPU-compute floor + fp32 (principal directive, independent)

After the fused cascade the all-7 is **GPU-COMPUTE-bound at ~674 s/forecast-hour**
(ledger `sync≈0.01 s`; util 96%). Question: is this reducible (kernel rewrite /
fp32) or the fundamental small-grid GPU limit? Independent (Opus + GPT-5.5 critic
`/tmp/gpt_kernel_fp32_report.md`; nsys per-kernel trace; gpu-metrics roofline was
**admin-privilege-blocked** on the Blackwell GB202, so occupancy is reasoned from
grid sizes + the ledger, not a counter readout).

### 7.1 Where the ~674 s/hr goes (Amdahl shape)
Evidence-weighted (no NVTX per-scheme markers; XLA fuses the step): **dynamics
RK3+acoustic ~30%, MYNN PBL/sfclay+GWD ~27%, Noah-MP ~12%, Thompson MP ~9%, RRTMG
~7% (cadence-amortized), boundary/interp ~10% (still full-child-grid waste), KF
(d01) ~2%, guards/output ~4%.** ⇒ dynamics is NOT a 70% slice; it's broad. **82.6%
of weighted column-steps are in the seven tiny 1 km leaves** — the dominant shape
is *many tiny low-occupancy programs*, not one saturated grid.

**nsys per-kernel trace (EMPIRICAL, eager root, privilege-free CUDA trace — gpu-metrics
roofline was admin-blocked):** the top GPU kernels are **generic element-wise XLA
fusions** — `loop_multiply_fusion` 11%, `loop_compare_fusion` 10%, `wrapped_divide`
10%, `loop_select_fusion` 9.7%, `loop_slice_fusion` 9.4%, subtract/add/sqrt/reverse…
**No single hot-spot (top kernel 11%)** → broad Amdahl, confirms the code analysis.
**Decisive: every kernel averages ~1.5 µs with hundreds–thousands of instances** —
the signature of **many TINY kernels** (launch/latency/occupancy-bound on small
grids), NOT large throughput-bound kernels. This is the empirical confirmation that
the limit is small-grid occupancy/launch overhead, **so a throughput lever (fp32)
cannot move it** — there is no large saturating kernel whose bytes/FLOPs fp32 would
halve. (`wrapped_divide` 10% + `wrapped_sqrt` 3% = EOS/physics transcendentals.)

### 7.2 fp32 verdict — STOP the broad-fp32 path (it is NOT the ≥2× lever)
- **Broad fp32 ≈ 1.1× only** (v0.16 verdict, GPT-reproduced 1.105–1.111×, 0% VRAM):
  storing `p_total`/`ph_total` in fp32 corrupts geopotential/PGF differences
  **27×–127×** beyond the accepted perturbation-fp32 floor — fp64 islands cannot
  recover bits lost at storage. The acoustic core is **deliberately fp64 around
  cancellation** (PGF/buoyancy, `advance_w` Thomas solve, `calc_p_rho`, mu_t).
- **Roofline**: dycore intensity ~1–3 FLOP/B ⇒ fp32 is **at most ~2× the
  *memory* slice** (dynamics ~30%), not 4–6×; the tiny under-filled nests
  (d06/d07 = 1,521 cols) reduce even that — **occupancy-bound**, so a perfect
  assembler kernel can't fill the SMs either. 96% util ≠ saturated SMs here.
- **Stability**: any non-bitwise perturbation amplifies chaotically (the fused
  cascade already showed P 1.3→20 over 2 h, still tolerance-PASS). fp32's larger
  per-step perturbation diverges faster + the known qke-1km blow-up ⇒ stability is
  a real risk, not free.
- A **stable** full-fp32 dynamics needs the ADR-031 perturbation-native rewrite
  (perturbation-authoritative p'/ph'/mu', fp64 islands, no in-loop
  `total−perturbation`) — a high-risk *longer* lane projected only ~1.8–2×, not a
  dtype flip.

**DECISION (per manager A/B/C gate): broad fp32 = STOP** — it does not lower ~674
toward ≤446 and risks instability. The only precision-safe ≥2× lever is **edge-only
boundary interpolation** (kills the ~4.1× full-child-grid waste, GPT's #1), plus
*gated* fp32-physics islands (~1.6× isolated, **unproven on all-7**). Even those cap
at ~2×.

### 7.3 ≥2× / 3× reachability — FINAL VERDICT (manager STOP-confirmed)
**≥2× is NOT single-card reachable for this all-7 tiny-nest geometry.** The full
precision-safe stack falls short:
- **fused cascade: ~1.27× measured** (util 96%, opt-in, tolerance-pass).
- **+ edge-only boundary: 689 s = 1.296×** (measured; only ~13 s / ~1.9% over fused — confirms the *partial* edge-only). GPT-reviewed
  **bit-identical (eager)** → ships DEFAULT-ON; but it is a *partial* edge-only —
  the W/E column strips still run the full-width row-interp, so it trims only part
  of the 4.1× boundary waste, and the boundary is only ~10% of the ~674 s floor.
- **+ fp32-physics islands (MYNN/RRTMG): ~1.5–1.6× projected** — still **< 2×**, and
  it does not meet the principal's >2× keep-bar, so it was **NOT pursued** (noted as
  an optional ~1.5× / 24h-tolerance follow-up the principal may take later).
- **3× would need ≤298 s/hr** on 55k tiny columns — multi-GPU territory.

Why fundamental: dynamics (~30%) is fp64-pinned around acoustic cancellation
(fp32 STOP); the seven 1 km nests under-fill the card (occupancy-bound — 96% util ≠
saturated SMs); a 12-rank CPU is genuinely competitive on tiny grids. **The all-7
tiny-nest is near WORST-CASE for single-card GPU speedup.**

**Headline (CONFIRMED independently, Opus + GPT):** the GPU's real value is **1 km
CAPABILITY (fits one card, bit-identical) + large single grids + cluster
weak-scaling**, NOT single-card speedup on tiny nests. **Honest deliverable: ~1.3×
(fused [+edge-only], opt-in fast-mode) + edge-only default-on (bit-identical) + the
capability framing.**

---

## 8. Hand-off to the manager

**Objective (done):** the v0.17 all-7 wallclock / GPU-idle blocker — find why the
GPU was mostly idle, fix it bit-identically where possible, measure, and (principal
extension) determine whether ≥2× is reachable / whether fp32 is the path.

**Delivered (committed on `worker/opus/v017-hostgap-fix`, off your `6d77acfe`):**
| commit | what | bit-identity | default |
|---|---|---|---|
| `191bbd2a` | block_between → async per-root-step sync (`GPUWRF_NESTED_SYNC_MODE`) | bit-identical (host-side) | `root` (on) |
| `ee016b1e` | nested compile-CHURN fix (commit seed carry) | bit-identical (`device_put`) | on |
| `d2617e44` | host wallclock ledger (`GPUWRF_HOST_LEDGER`) | n/a (diag) | off |
| `9c36b3f8` | env-gated jit boundary builder (`GPUWRF_JIT_BOUNDARY`) | not proven bitwise | off |
| `324a2dba`+`7e7aeb32` | env-gated fused d02-cascade (`GPUWRF_NESTED_FUSE`) + GPT-review fixes | tolerance-PASS, NOT bitwise | off |
| `209b8656` | edge-only (ring-only) boundary interp (`GPUWRF_EDGE_ONLY_BOUNDARY`) | bit-identical (eager, GPT+36 tests) | **ON** |

**The two decisive findings:**
1. The all-7 NEVER forecasted on a cold cache (churn) — `ee016b1e` unblocked it.
   Your base `6d77acfe` did not fix it (sharding key, not n_steps).
2. After the host-dispatch fix the all-7 is **GPU-COMPUTE-bound (~674 s/hr)** on
   tiny under-filled nests. **~1.4× is the host-maxed honest number; fp32 = STOP
   (occupancy + fp64-pin wall); 3× is not single-card reachable.** Edge-only
   boundary interp is the only precision-safe ~2× lever left.

**Decisions (manager-confirmed):**
- A. Fused cascade: **OPT-IN** (`GPUWRF_NESTED_FUSE`, default OFF) + enabled for the
  benchmark headline (~1.27×, util 96%, tolerance-pass). ✔.
- B. fp32 (broad): **STOP** — occupancy/fp64 wall, won't reach ≤446. Honest finding
  §7 with evidence. ✔.
- C. **edge-only boundary: SHIP DEFAULT-ON** (`209b8656`, bit-identical eager). ✔.
- D. **≥2× chase: STOP** (manager-confirmed) — the full precision-safe stack caps
  **<2×** (fused 1.27× → +edge-only ~1.3× → +fp32-physics ~1.5–1.6×). fp32-physics
  islands NOT pursued (don't meet the principal's >2× keep-bar); noted as an
  **optional ~1.5× / 24h-tolerance fast-mode follow-up** for the principal to decide.
- **HEADLINE: ~1.3× (fused[+edge-only], opt-in) + edge-only default-on; reframe to
  1 km CAPABILITY + large grids + cluster weak-scaling** (this work CONFIRMS it).

**Required before any default-on of the fused path:** a **72h-stability-vs-WRFv4
PASS** on the fused config (2 real-world cases) — its P diverges chaotically vs
eager (1.3→20 over 2 h, still tolerance-PASS at h2); only the 72h gate proves it
"stably varies, does not explode" over a long run. Do NOT 72h a config fp32 might
change (it won't — fp32 is STOP).

**Unresolved / honest gaps:** nsys gpu-metrics roofline was admin-privilege-blocked
(occupancy reasoned, not counter-measured); the §7.1 per-scheme split is
evidence-weighted (no NVTX); the fused-cascade ~38 min one-time compile is heavy on
a fresh machine (cached after) and argues for a smaller fusion granularity if pursued.
