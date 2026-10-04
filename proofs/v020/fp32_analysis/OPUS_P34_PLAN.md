# OPUS — v0.20.0 POINTS 3 & 4: STRUCTURAL speedup (host/launch/HBM) + timestep/cadence

**Author:** Opus 4.8 (analyst #1 of 2; peer is GPT). Independent. **Date:** 2026-06-21.
**Status:** PLANNING ONLY — no GPU runs, no source edits.
**Builds on** (does not re-derive): `OPUS_FP32_PLAN.md` + `GPT_FP32_PLAN.md` — both concluded the
nested all-7 max_dom=9 step is **launch-count / HBM / occupancy bound (STRUCTURAL), not
fp64-ALU bound**, so precision alone caps at ~1.1–1.4× on the nest.
**Source/evidence read:** `runtime/domain_tree.py` (run_domain_tree_callbacks, the fused
cascade `_build_fused_cascade_program`, `root_sync_cadence`), `nesting/boundary_construction.py`
(`build_child_boundary_package`), `runtime/operational_mode.py` (`_advance_chunk`, the RK/acoustic
substep scan at :2111, the RK stage descriptors at :2949-2955), `integration/nested_pipeline.py`
(`_nested_sync_mode_from_env`, radiation cadence), `proofs/v018/maxdom9_speedup/profile_opus.md`
(nsys: launches/syncs/allocator), `proofs/v019/release_prep/gate_summary.json` (current event
census), the all-7 namelist `proofs/v018/maxdom9_speedup/namelist.h1.input`.

---

## 0. TL;DR

**Points 3 & 4 are the REAL near-term levers for nest wall-time** (the fp32 reports proved
precision is not, on this benchmark). They split cleanly:

- **POINT 3 (structural) attacks the launch/HBM/host barrier.** The fused cascade (v0.19,
  default-on) ALREADY moved the d02-subtree boundary builds on-device, so the cheap remaining
  host levers (pooling allocator, deeper async sync, wider fusion, command-buffer capture) are
  **bounded ~1.2–1.5× combined** — they trim the host/sync fraction but cannot fix the
  *occupancy* wall. **The one structural lever with a real multiplier is the fused acoustic +
  vertical-implicit-solve kernel** (the [S2]-identified "next step"): it collapses the ~12 k
  tiny intra-domain kernels/step that dominate the launch-bound, occupancy-bound regime — and
  it is the lever that *compounds with fp32* (fewer kernels AND halved HBM). High effort,
  highest reward.

- **POINT 4 (algorithmic) is the cheapest *real* GPU-compute saving — and it is hiding in
  plain sight.** The all-7 namelist runs **d01 `time_step=18 s` at `dx=9 km` — only 2×dx(km),
  vs WRF's up-to-6×dx guideline** — and `acoustic_substeps=10` (per full step: 1+5+10 = 16
  acoustic substeps; WRF-typical `time_step_sound≈4` → ~7). Because every nest dt is locked to
  `parent_dt/ratio`, **raising d01 dt and/or cutting the acoustic substeps reduces step count
  near-proportionally across ALL 9 domains** — a direct **1.3–2× (dt) × 1.1–1.4× (substeps)**
  candidate, stability/CFL-gated. This is config/algorithmic, not a rewrite, and it *reduces
  the very launch count point 3 attacks structurally* — the two are deeply synergistic.

**Honest combined trajectory** (each gated on the no-blow-up + 24–120 h wind/temp/cloud skill
policy, measured not promised): cheap point-3 host levers ~1.2–1.5× × point-4 timestep/substep
~1.4–2.2× ≈ **~1.7–3× before the fused acoustic kernel or fp32**, with the fused kernel + fp32
(points 1 & the structural part of 3) carrying it toward the larger multiple at DRAM-bound
scale. **Crucially, every point-4 step reduces fp32 accumulation drift (fewer steps) — so
points 1, 3, 4 reinforce rather than compete.**

**The current regime (the lens for all sizing below):** v0.19.1 warm all-7 = **713 s/fc-h**
(seg-2 683), 1.43× vs CPU 1020. Per fc-h: **5025 advances + 4422 forces + 0 feedback (one-way)
+ 27 outputs** (`gate_summary.json`). This is essentially the **v0.17 fused regime** (689–702
s/fc-h), which `profile_opus.md` §2.1 measured as **host-dispatch-bound at ~56 % GPU util** —
i.e. the GPU is *idle-waiting* a meaningful fraction, on top of being occupancy-bound on the
tiny leaves. (The v0.18.3 "GPU-compute-bound, 91 % busy, 296 k single-thread kernels" state was
the *broken* scan-pathology regime; fusion fixed it, returning us to host+occupancy bound.)

---

## 1. POINT 3 — cut host-orchestration: on-device boundary-builds + sync + kernel fusion

### 3.1 Strategic analysis + the speedup model

The per-fc-h orchestration is **5025 advances + 4422 forces**. The brief frames the 4422 as
"host force/boundary-builds," but the source shows the nuance that determines the lever:

- **`build_child_boundary_package` (`boundary_construction.py:360`) is on-device JAX**
  (ring-gathers `ring3d/ring2d`, `_fit`, `two_time` stacking — all `jnp`; the only host ops are
  `int(shape)` reads, traced once). So the "boundary build" is NOT host numpy compute.
- **The fused cascade (`domain_tree.py:446-490`) already inlines the d02→{d03..d09} boundary
  builds + child advances into ONE jitted device program** (`_build_fused_cascade_program`:
  parent `_advance_chunk` → for each of 7 leaves `build_child_boundary_package` →
  `_advance_chunk`). For the fused subtree the force is *on-device already*. The 4422 "force"
  entries are mostly **host-side event-logging** (`events.append`), not host compute.

So the real residual host cost is: (a) the **per-root-step Python dispatch + bookkeeping**
(~9447 `events/own_steps` appends/fc-h — cheap per call, but unbounded RAM, efficiency-roadmap
item 7); (b) the **d01 root advance + d01→d02 force**, which are OUTSIDE the d02-subtree fused
program (2-level tree: only the flat d02-subtree is fused); (c) the **allocator + sync churn**
(`profile_opus.md` §2.3: platform allocator = synchronizing `cuMemFree` per transient, 56 k
alloc + 56 k free + 36 k `cuStreamSynchronize` per window); (d) the **GPU idle-wait** from
host-dispatch-bound execution (~56 % util in the v0.17-equivalent regime).

**The speedup model (tied to the launch/HBM/occupancy finding):** on a host-dispatch-bound +
occupancy-bound nest, there are two distinct headrooms:

1. **Host/sync headroom (bounded ~1.2–1.5×):** the GPU is idle ~44 % of wall (v0.17 util 56 %).
   Removing host/sync/allocator overhead and deepening the async queue recovers *up to* that
   idle fraction — but no more, because once the GPU is the bottleneck the occupancy wall binds.
2. **Occupancy/launch headroom (the real multiplier, high effort):** the ~12 k tiny dependent
   kernels/step ([S2]) on L2-resident leaves are launch-latency- and parallelism-starved.
   **Fewer, larger kernels** (fusing the acoustic substep body + vertical solve) raise
   work-per-launch AND occupancy — this is the only lever that moves the structural wall, and
   it is exactly the [S2] "next step — structural, not a dtype tweak."

### 3.2 Ranked point-3 sub-levers (reward / effort)

| # | Lever | Attacks | Est. speedup | Effort | Reward/Effort | Risk |
|---|---|---|---|---|---|---|
| **L1** | **Pooling allocator** `platform`→`cuda_async`/BFC | per-transient synchronizing `cuMemFree` + alloc churn (113 k/window) | **1.05–1.25×** | **LOW** (env flag + VRAM re-verify) | **HIGHEST** | 1 km-nest OOM regression — must re-verify AC1_FIT fits |
| **L2** | **Deeper async sync** `root:1`→`root:K` / `segment` | GPU idle-wait between root cascades; queue depth | **1.03–1.12×** | **VERY LOW** (env, already plumbed) | **HIGH** | peak VRAM rises with queue depth (now headroom post leak-fix) |
| **L3** | **Wider fusion**: fold d01→d02 into the cascade + bound/stream the host event lists | the 2 un-fused root host steps/root-step + item-7 host-RAM growth | **1.02–1.06×** | MEDIUM (fused program is currently flat-1-level) | MEDIUM | correctness of 2-level fused force ordering |
| **L4** | **Command-buffer / CUDA-graph capture** of the per-root-step cascade | `cuLaunchKernelEx` + 36 k `cuStreamSync` host-API overhead (helps the *host-bound* fraction) | **1.05–1.15×** | MEDIUM (banked recipe exists) | MEDIUM (conditional) | numerics-changing fusions must be EXCLUDED; needs stable shapes |
| **L5** | **Fused acoustic + vertical-implicit-solve kernel** (the structural multiplier) | the ~12 k intra-domain kernels/step: `loop_multiply_fusion`, `input_reduce_fusion`, PCR tridiagonal | **1.5–3× on the dycore portion** | **HIGH** (XLA fusion tuning → Pallas) | **HIGHEST reward** | Pallas maintenance; was v0.15-deferred; must keep all-7 green |

**Sequencing logic:** L1+L2 are near-free flag wins (do first, together — they recover the
host-idle fraction). L4 pairs with them (capture the now-stable fused cascade to cut host-API
overhead — and note WHY it works now vs the old no-op: see 3.5). L3 is a moderate structural
tidy. **L5 is the real lever** but high-effort; sequence it after the cheap wins bank their
~1.2–1.5×, and design it to compound with fp32 (point 1) and the reduced step count (point 4).

### 3.3 Step-by-step incremental plan (each independently testable + reversible)

**S0 — Baseline + instrument (no code).** Capture a CLEAN v0.19.1 warm nsys of the all-7 (the
existing `profile_opus.md` is the *broken* v0.18.3 state — we need the current one): GPU util %,
host-gap/sync per root step, launches/step per domain, allocator alloc/free/sync counts,
top-kernel families. Add NVTX ranges around each domain advance + a host ledger. This settles
"how host-bound is v0.19.1 NOW" and sizes L1–L4 precisely. **Gate:** none (measurement).

**S1 — Allocator (L1).** Set `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (or BFC) for the
production nested run. **Gate:** warm s/fc-h improves vs 713; **AC1_FIT 1 km nest still fits
(no OOM)**; VRAM flat over ≥3 output groups (the item-1 regression guard); all 9 domains
tolerance-/skill-green; bit-identical wrfout (allocator changes nothing dispatched). Reversible
(env flag).

**S2 — Async sync depth (L2).** A/B `GPUWRF_NESTED_SYNC_MODE=root:2`, `root:3`, `segment`
against `root:1`. **Gate:** s/fc-h improves; peak VRAM stays within budget (the binding limit —
deeper queue = more in-flight transients); byte-identical wrfout (sync is a host wait only,
changes no dispatched op — stated in `_nested_sync_mode_from_env` docstring). Pick the deepest
cadence that fits VRAM. Reversible.

**S3 — Command-buffer capture (L4).** Apply the banked recipe
(`KERNEL-OPTIMIZATION-FINDINGS-FINAL.md` §4): `--xla_gpu_enable_command_buffer=FUSION,CUBLAS,
CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE` + `--xla_gpu_command_buffer_update_mode=
NEVER_UPDATE`, **EXCLUDE** `DYNAMIC_SLICE_FUSION,DYNAMIC_SLICE_COPY_FUSION` (those CHANGE
numerics — 99/168 leaf hashes). **Gate:** s/fc-h improves on the host-bound fraction; **wrfout
bit-identical** (the excluded fusions are the only numerics-changers); no compile-cache
explosion. If wall-neutral (GPU already saturated after S1/S2), STOP — it's the v0.15 device-
bound no-op and not worth the complexity. Reversible (flag).

**S4 — Wider fusion + host-list bounding (L3).** Extend `_build_fused_cascade_program` to fold
the d01→d02 root force + d01 advance into the cascade (2-level fused program), and bound/stream
the `events`/`outputs` host lists (item 7). **Gate:** fewer host round-trips/root-step (NVTX
ledger); bit-identical; all-7 green; no host-RAM growth over 24 h.

**S5 — Fused acoustic + vertical-solve kernel (L5, the structural multiplier).** Two staged
sub-approaches, cheapest first:
- **S5a (XLA-only):** raise `_acoustic_unroll()` on the substep scan (`operational_mode.py:2111`)
  and restructure the substep body so `acoustic_substep_core` fuses into fewer, larger kernels;
  audit the HLO for the `loop_multiply_fusion`/`input_reduce_fusion` kernel-count collapse.
  Cheap, reversible, but bounded by what XLA will fuse (the [S2] finding: XLA left ~12 k
  launches even in mixed mode → XLA-only may not collapse enough).
- **S5b (Pallas megakernel):** hand-fuse the acoustic substep + the PCR/Thomas vertical solve
  into one per-column in-register kernel (the deferred v0.15 "S2" lever, reopened because we now
  have a concrete launch-bound target + the fused-cascade infra). The fp64 cancellation brackets
  live in **registers** (never HBM) — so this is also the enabler for the fp32 plan's island
  strategy (point 1). **Gate:** launches/step drop materially (NVTX/nsys); the dominant kernels'
  wall drops (the [S2] failure signature gone); all 9 domains bit-/skill-green; warm s/fc-h
  improves. Keep behind a flag with the eager path as fallback (ADR-class change).

### 3.4 Validation / probes / tools (measure launch/HBM/occupancy + nested skill)

- **Launch-count delta:** nsys `cuda_api_sum` (`cuLaunchKernelEx` count) + `cuda_gpu_kern_sum`
  (kernel instances) per step, before/after each lever. The headline metric for L4/L5.
- **HBM / occupancy proxy:** HW counters are blocked (`ERR_NVGPUCTRPERM`, `profile_opus.md` §1),
  so use: (a) CUDA-timeline kernel durations + grid dims; (b) the **L2-residency model**
  (per-domain working set vs 96 MiB L2) to predict where fp32/fusion can help; (c) GPU-util %
  from `nvidia-smi` sampling (the `run_sampled_nested.sh` harness already exists in
  `proofs/v019/vram_fix/`).
- **Host-gap / sync:** `cuStreamSynchronize` time + count; GPU-active-union vs wall (the
  `profile_opus.md` §2.1 method) → the host-bound fraction that L1/L2/L4 can recover.
- **VRAM regression guard:** `jax.live_arrays()` census FLAT after the first output group
  (efficiency-roadmap item 1) — binding for L1/L2 (deeper queue / pooling must not leak or OOM).
- **Correctness:** L1–L4 are **dispatch-only / wall-only** → **bit-identical wrfout** is the
  gate (`scripts/compare_wrfout_grid.py`, all 9 domains). L5 changes fusion boundaries → tiered
  field tolerance + the point-1 **24–120 h wind/temp/cloud skill + no-blow-up** policy on the
  nested all-7.
- **The decisive probe order:** S0 baseline → each lever A/B (warm, compile-excluded) → re-nsys
  to confirm the launch/host metric actually moved (don't trust wall alone — the [S2] lesson is
  that walls can move for the wrong reason or not move despite HLO changes).

### 3.5 Roadblocks + ideas

- **Command-buffer was a no-op before — when does it help now?** The v0.15 A/B (and the
  v0.18.3 graphs-on-vs-off A/B, `proofs/v018/fused_restore/ab_d03_*`) found capture **wall-
  neutral** because the step was *device-bound*. The current v0.19.1 nest is back to
  **host-dispatch-bound (~56 % util)** — so capturing the per-root-step cascade removes the
  `cuLaunchKernelEx`/`cuStreamSync` host-API overhead that *is* now on the critical path. The
  prerequisite: **stable launch topology** (the fused program has fixed per-domain shapes → it
  is capturable; the old broken dynamic-carry scan was not). **Idea:** capture the fused cascade
  program specifically (it's the stable, repeating unit), not the whole daily wrapper.
- **Pallas megakernel risk (L5b):** v0.15 deferred it on "the dynamic XLA scan scales to any
  grid" grounds — valid for *large* grids, but the all-7 is the *tiny* launch-bound case where a
  megakernel's launch-collapse is exactly the win. **Idea:** scope L5b as a *nest-mode* kernel
  (small fixed nz=44 columns), keeping the dynamic XLA path as the large-grid default — i.e. a
  per-regime kernel, not a global replacement. This matches the v0.15 "conditional future path."
- **Allocator OOM (L1):** `cuda_async`/BFC pools memory → can raise peak. **Idea:** pair with
  the point-1 fp32 VRAM reduction so the pool fits; gate hard on AC1_FIT.
- **2-level fused force (L3):** folding d01 in must preserve the exact force/advance ordering
  WRF's `module_integrate` mandates. **Idea:** reuse the existing event-ledger identity test
  (`tests/test_v0110_domain_tree.py`) to prove scheduler equivalence on CPU before GPU.

### 3.6 How nested all-7 stays fully functional

- L1–L4 change **no dispatched op** (allocator, host-sync, command-buffer capture, host
  bookkeeping) → **bit-identical wrfout** is the binding gate; the all-7 is unaffected by
  construction. The fused cascade's fail-closed safety (root/feedback/non-flat → eager) is
  untouched.
- L5 is behind a flag with the eager/XLA path as fallback; every domain d01–d09 must stay
  finite + skill-green, and d03 1 km is the steep-terrain sentinel. No domain may silently fall
  back to a slow path without the report naming it.
- The v0.19.1 closure-cycle leak fix (`domain_tree.py:406`) and the VRAM regression guard are
  preserved across all levers (L1/L2 explicitly re-verify VRAM-flat).

---

## 2. POINT 4 — larger timestep + physics-call cadence (algorithmic, stability-gated)

### 4.1 Strategic analysis + expected speedup (the conservative-timestep finding)

The all-7 namelist (`namelist.h1.input`): `time_step=18` s, `dx = 9000, 3000, 1000×7` m,
`parent_grid_ratio = parent_time_step_ratio = 1, 3, 3, …`. So the dt ladder is **d01=18 s
(9 km), d02=6 s (3 km), d03–d09=2 s (1 km)**, and `acoustic_substeps=10` (per full RK3 step:
1+5+10 = **16 acoustic substeps**, `operational_mode.py:2949-2955`).

Two conservatism signals, both reducible and both near-proportional wins because **every nest
dt is locked to `parent_dt/ratio`** (cut d01 dt → all 9 domains' step counts fall together):

**(A) The model timestep is ~2–3× more conservative than WRF guidance.**
- WRF rule of thumb: stable `dt(s) ≈ 5–6 × dx(km)` for the coarse domain. d01 at dx=9 km → WRF
  allows **~45–54 s**; the config uses **18 s = 2×dx(km)**. That is a **2.5–3× headroom signal**
  on the *advective/gravity-wave* CFL — IF the real max winds + gravity-wave speeds on this
  Canary domain leave that slack (steep terrain can erode it; must MEASURE per domain).
- Because the ratio chain is fixed, d01 18→36 s (still only 4×dx, conservative) **halves every
  domain's step count → ~2× fewer advances/forces/launches** across the whole cascade. Even
  18→24 s is ~1.33×.
- **This is the single cheapest *real* GPU-compute saving in the whole roadmap** — it is a
  namelist/config change, not a rewrite, and it shrinks the exact 5025-advance/4422-force/
  ~12k-launch count that point 3 attacks structurally.

**(B) The acoustic substep count is on the high end.** `acoustic_substeps=10` → 16 substeps/
step; WRF-typical `time_step_sound≈4` → ~7 substeps/step. The acoustic substep dt for d01 is
18/10 = 1.8 s → horizontal sound CFL `c_s·dt_a/dx ≈ 340·1.8/9000 ≈ 0.07` (and similarly ~0.07
for the finer nests by construction) — **far below the ~0.5–1 the split-explicit scheme
tolerates**, i.e. a strong headroom signal. Cutting n_sound 10→6 is **~1.4× fewer acoustic
substeps** (the dominant inner-loop work), gated on acoustic stability. (Caveat: the dycore-
rewrite history had stability sensitivity; n_sound=10 may be a deliberate safety margin — so
this is gated, not assumed.)

**Honest point-4 budget:** dt 18→24–36 s = **1.3–2×**; n_sound 10→6 = **~1.1–1.4×** on the
dycore fraction. Combined, plausibly **~1.4–2.2×** — *if* the CFL/stability headroom is real on
this steep-terrain nest. This is measured per-domain, gated hard on the policy, never assumed.

**(C) Physics-call cadence.** Audit findings:
- **Radiation:** `radt` target = 1800 s (`nested_pipeline.py:70,182`; `radiation_cadence_steps =
  round(1800/dt)`). This is WRF-standard (radt=30 min) and radiation is only ~7 % of the step,
  already cadenced — **little headroom**; lengthening radt risks the diurnal-cycle skill. Low
  priority.
- **Microphysics (Thompson):** runs **every step** — WRF-standard; cannot reduce without skill
  loss. (And the fp32 reports already showed Thompson is launch/bandwidth-bound, not a precision
  lever; here the point is its CADENCE is correct, not a lever.)
- **Cumulus:** at dx ≤ 4 km, WRF runs explicit convection (cu_physics OFF). **Audit:** confirm
  cu is OFF on d02–d09 (it should be) — if any fine nest calls cumulus, that's a free removal.
- **PBL/surface:** every step (WRF-standard). No cadence headroom.
- **Net:** the cadence lever is small EXCEPT the dt/n_sound levers above, which are the real
  point-4 prize. Lengthening dt *automatically* lengthens the radiation step count target
  (cadence = 1800/dt), so dt and radiation co-move correctly.

**Synergy with points 1 & 3:** fewer steps/substeps → (i) fewer kernel launches → compounds
with point 3's structural fusion; (ii) **fewer fp32 accumulation steps → less drift → fp32 is
SAFER** (point 1). So point 4 is not a competing lever — it strengthens both.

### 4.2 Step-by-step incremental plan

**T0 — Per-domain CFL headroom probe (no skill run).** For each domain, from a warm state,
measure the actual max horizontal wind, max vertical-implicit constraint, and gravity-wave
speed → compute the realized advective + acoustic CFL at the current dt/n_sound. This *predicts*
the headroom (and where steep terrain erodes it) before any expensive forecast. **Gate:** none
(measurement); output the per-domain CFL margin table.

**T1 — Acoustic substep reduction (n_sound 10→8→6).** Step n_sound down one level at a time
(keep the RK1=1, RK2=n//2, RK3=n cadence). **Gate (per level):** the point-1 stability ladder
(1 substep → 20 steps → 1 h → 24 h → 120 h) on the all-7; **no blow-up** (esp. d03 1 km steep
terrain, the qke sentinel); **wind/temp/cloud 24–120 h skill preserved** within the point-1
bands. Revert one level on any fail. Reversible (namelist field).

**T2 — Timestep increase (d01 18→24→30→36 s, ratios fixed).** Raise d01 dt one level at a time;
the nest dt's follow via the ratio. **Gate (per level):** the stability ladder + skill bands on
all 9 domains; watch the finest nest (d03 1 km, dt 2→2.67→3.3→4 s) for the first CFL failure
(steep terrain + high winds bite there first). **The binding constraint is the finest, steepest
nest, not d01.** Revert one level on any fail. Reversible.

**T3 — Decouple time_step_ratio from grid_ratio (if T2 binds on the fine nest).** If the fine
nests can't take the proportional dt but the coarse can, WRF allows
`parent_time_step_ratio > parent_grid_ratio` for selected nests (finer time subcycling only
where needed). This keeps the coarse-domain dt win while protecting the steep 1 km nests.
**Gate:** as T2, per domain. Higher complexity (per-domain dt bookkeeping in the cascade).

**T4 — Cumulus-cadence audit.** Confirm cu_physics OFF on dx ≤ 4 km nests; remove any
unnecessary call. **Gate:** bit-identical/skill-green (removing a no-op is free; removing an
active call needs the skill gate).

### 4.3 Validation / probes / tools

- **CFL probe (T0):** a per-domain diagnostic computing realized Courant numbers (advective +
  acoustic) from a warm state — the predictor for T1/T2 headroom and the early-abort signal.
- **Stability ladder (the hard floor):** reuse the point-1 ladder. Abort on any NaN/Inf in a
  non-carve-out field, super-linear divergence, or the d03 qke non-finite signature. Larger dt /
  fewer substeps are **stability-first** changes — the ladder is the primary gate, not tolerance.
- **Skill gate (binding):** 24/48/72/96/120 h wind/temp/cloud on the all-7 vs CPU-WRF / fp64-GPU
  baseline, within the point-1 bands. Larger dt slightly changes the discrete solution (it is NOT
  bit-identical) — so this is a **skill-equivalence** gate, exactly the relaxed policy. Cumulative
  precip + QVAPOR get the carve-out (drift OK, no blow-up).
- **Speed proof:** warm s/fc-h at each dt/n_sound level + the launch-count drop (nsys) — confirm
  the step-count reduction translated to wall (it should, near-proportionally, since the nest is
  launch-bound). Report per-domain.
- **Tools:** the namelist fields are already plumbed (`acoustic_substeps`, `dt_s` via
  `time_step`, `parent_time_step_ratio`); T0/T1/T2 need only config sweeps + the ladder harness —
  **no source rewrite**, which is what makes point 4 cheap.

### 4.4 Roadblocks + ideas

- **Steep-terrain CFL erosion (the main risk):** the 1 km Canary nests over Tenerife have large
  vertical velocities + terrain-following coordinate stiffness; the realized CFL margin may be
  much tighter than the 2×dx headroom suggests, and the d03 qke non-finite history (point-1
  §K-STAB) shows this nest is the fragile one. **Idea:** T3 (decouple time_step_ratio) protects
  exactly this — take the dt win on d01/d02, keep the 1 km nests at a safe dt.
- **Vertically-implicit solve + epssm:** `epssm=0.5` (off-centering) damps acoustic modes; a
  larger dt may need a slightly larger epssm for stability (WRF couples these). **Idea:** treat
  epssm as a paired knob with dt — but increasing epssm adds damping (skill cost), so gate on
  skill, not just stability. Prefer the smallest epssm that holds.
- **Larger dt changes the solution (not bit-identical):** unlike point-3 host levers, point 4 is
  a genuine numerical change. **Idea:** lean on the relaxed policy — the gate is 24–120 h
  wind/temp/cloud skill, not WRFv4 bit-parity; a dt that preserves skill is acceptable even if
  per-cell fields differ. This is precisely the regime the user authorized.
- **n_sound and dt interact:** the acoustic substep dt = dt/n_sound; raising dt while cutting
  n_sound raises the acoustic CFL twice over. **Idea:** sweep them jointly (T0 predicts the
  combined acoustic CFL); don't change both blindly in one step — one knob per ladder run so the
  failure is attributable.

### 4.5 How nested all-7 stays fully functional

- Point 4 is the all-7 benchmark *itself* run at a coarser dt / fewer substeps — every gate is
  ON the nested all-7, every domain d01–d09 must stay finite + skill-green, and the binding
  constraint is the finest steep nest (d03 1 km). No level ships unless all 9 pass.
- Every change is a **single namelist field**, fully reversible, and the default stays the
  current 18 s / n_sound=10 until a level passes the ladder + skill gate.
- T3 (decoupled time-step ratio) is the safety valve that keeps the fragile 1 km nests stable
  while still banking the coarse-domain dt win — so the benchmark never has to choose between
  speed and the steep-terrain nests staying functional.

---

## 3. SYNERGY, COMBINED BUDGET, AND RECOMMENDATION

**The three v0.20 lever-families compose multiplicatively and reinforce each other:**

| Lever family | Mechanism | Honest range (nest, measured-not-promised) | Risk/effort |
|---|---|---|---|
| **Point 3 cheap host (L1+L2+L4)** | recover the ~44 % GPU idle-wait (host/sync/allocator/graph) | **1.2–1.5×** | low; bit-identical |
| **Point 4 dt + n_sound (T1+T2)** | fewer steps/substeps across all 9 domains | **1.4–2.2×** | medium; CFL/skill-gated |
| **Point 3 fused acoustic kernel (L5)** | collapse the ~12 k intra-domain launches; raise occupancy | **1.5–3× on dycore** | high; Pallas |
| **Point 1 fp32** (separate report) | VRAM + HBM at DRAM-bound scale; compounds with L5 | ~1.1–1.4× nest / 2–3.5× at scale | high; numerics |

**Reinforcement (not competition):** point 4 *reduces the launch count* point 3 fuses, and
*reduces the step count* that drives fp32 drift (point 1). The fused acoustic kernel (L5) is
*also* the structural enabler for fp32's in-register islands (point 1). So the right v0.20
program runs them as one coordinated sprint, cheapest-and-safest-first.

**Recommended order (measure-then-commit, every step gated on no-blow-up + 24–120 h
wind/temp/cloud skill, all-7 functional):**

1. **S0 + T0 — instrument** (clean v0.19.1 nsys + per-domain CFL probe). Settles how host-bound
   we are now and where the timestep headroom is. ~1 GPU session, no risk.
2. **S1 + S2 — allocator + async sync depth** (cheap host wins, bit-identical, ~1.2–1.4×).
3. **T1 + T2 — n_sound then dt** (the cheapest *real* compute saving, ~1.4–2.2×, ladder-gated;
   T3 if the 1 km nests bind). This is likely the **biggest near-term wall-time win for the
   least effort** — pursue aggressively but gate hard.
4. **S3/S4 — command-buffer + wider fusion** (recover residual host-API overhead).
5. **L5 — fused acoustic + vertical-solve kernel** (the structural multiplier; high effort;
   design to compound with fp32). 
6. **Point 1 fp32** lands the VRAM/capability + DRAM-bound-scale speed (separate report).

**My honest expectation:** the **point-4 timestep/substep levers are the surprise high-value,
low-effort win** (the config is measurably conservative vs WRF guidance, and the nest is
launch-bound so fewer steps ≈ proportional wall) — *if* the steep 1 km nests have CFL headroom,
which T0 settles cheaply. The **point-3 cheap host levers are real but bounded** (~1.2–1.5×, the
GPU-idle fraction). The **fused acoustic kernel (L5) is the only structural multiplier** and is
where the bulk of the remaining nest speedup lives, but it is the high-effort item and should be
sequenced after the cheap wins prove out. Combined, points 3 + 4 (before fp32 and before L5)
plausibly reach **~1.7–3× on the all-7 nest** — and unlike fp32, this is wall-time on the
*actual benchmark*, not only at DRAM-bound scale.

---

*Appendix — anchors (verified 2026-06-21): `runtime/domain_tree.py:227` (run_domain_tree_callbacks),
`:240/:290-309` (root_sync_cadence / async sync), `:406` (closure-leak fix), `:446-490`
(_build_fused_cascade_program), `:519` (_fusable_parent fail-closed); `nesting/boundary_construction.py:360`
(build_child_boundary_package, on-device); `runtime/operational_mode.py:2111` (acoustic substep scan,
`unroll=_acoustic_unroll()`), `:2949-2955` (RK stage descriptors, acoustic_substeps=10), `:366/:406`
(dt_s=10 / radiation_cadence_steps=60 defaults); `integration/nested_pipeline.py:70,182`
(radt=1800 s target), `:727 _nested_sync_mode_from_env` (sync modes root/advance/segment);
`proofs/v019/release_prep/gate_summary.json` (5025 adv / 4422 force / 713 s-fc-h); `namelist.h1.input`
(time_step=18, dx 9/3/1 km, ratios 1/3/3); `proofs/v018/maxdom9_speedup/profile_opus.md` (nsys host/alloc/sync);
`KERNEL-OPTIMIZATION-FINDINGS-FINAL.md` §4 (command-buffer recipe + EXCLUDE list). [S2] =
`.agent/reviews/2026-06-13-gpt-fp32-s2-result.md` (12 k launches/step, "structural next step").*
