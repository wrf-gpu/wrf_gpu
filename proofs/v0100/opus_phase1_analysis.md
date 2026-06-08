# Opus-MAX v0.10.0 Phase-1 Kernel-Optimization Analysis (INDEPENDENT)

**Analyst:** Opus 4.8 MAX kernel-optimization analyst
**Code analyzed:** v0.9.0 kernel @ `REPO
**Mode:** analysis only — NO code changes; NO GPU runs (existing artifacts only). taskset 0-3.
**Calibration:** principal exit-gate = near-theoretical-optimum, ALL inefficiencies noted (ideally removed), <1% gains out of scope, every not-removed item needs >=5 failed attempts OR a written not-worth-it justification.

> **Workload anchor:** Canary d02 (159×66×44 = 461,736 mass cells), fp64 (`force_fp64=True` operationally — `daily_pipeline.py:234`), dt=10s, 10 acoustic substeps (16/step across RK1+RK2+RK3), RRTMG radiation at cadence 180. Warmed per-step = **42.6–45.5 ms** (gate, `segscan_24h.json`) / **26.9 ms** (light-mem no-radiation marginal, `roofline_costanalysis.json`). Dycore-only = 16.9 ms.

---

## 0. EXECUTIVE SUMMARY (the 5 levers)

1. **Acoustic substep is the launch-bound core (~6,450 of the ~6,890 micro-kernels/step).** nsys proves it: `loop_add_fusion_9` = **116,136 instances** (~3,226/step), `loop_multiply_fusion_1` = 47,520 (~1,320/step), `loop_subtract_fusion_1` = 46,440 (~1,290/step), `loop_select_fusion_3` = 22,176 (~616/step), all ~780–1,050 ns. Plus **129,641 Device-to-Device memcpys** (~3,600/step) = the `concatenate`/`jnp.pad(edge)`/scan-carry shuffles. **Lever = fuse across the substep `lax.scan` boundary** (scan `unroll`, shrink the carry, kill the pad/concatenate face-pairs). This is the #1 lever, ~30–45% of the step.
2. **Thompson microphysics ~20 ms = ~half the coupled step**, of which ~85% is sedimentation (4 species × 64-iteration `lax.scan`, each body doing 2 `concatenate` shifts). The static `NSED_MAX=64` runs ~5× more substeps than the typical per-column `nstep≈8–12`. Lever = bit-identical `unroll` (already default 2) + tighter/data-aware substep bound + species batching of the flux-shift.
3. **Precision is sequenced AFTER fusion, not now.** `force_fp64=True` makes the whole step fp64; the gated-fp32 matrix (`precision.py`) is fully dormant operationally. fp32 measured ~1.00× *today* because the step is launch-bound and the fp64-acoustic-island boundary converts cancel the byte saving. Once levers 1–2 make phases bandwidth-bound, gated-fp32 on the non-acoustic memory-bound fields (theta/u/v/q advection inputs, Thompson, MYNN) becomes a real second-wave lever. **Do fusion first.**
4. **Per-step whole-state precision pass + redundant stage-invariant rebuilds.** `_enforce_operational_precision` casts all ~60 state leaves every step (`operational_mode.py:2197`); `dry_cqw` is rebuilt twice per stage (`:889` and `:1283`) and `AcousticCoreState` carries ~60 fields through the substep scan, most of them stage-constant. Small individually, real in aggregate as launch/carry-copy reduction.
5. **End-to-end ceiling ≈ 8–11× vs clean CPU-WRF (83 s/fc-hr); ≥10× is reachable but conditional, and ONLY via launch-count reduction (fusion), never precision.** Realistic post-Phase-1 coupled step ~**16–24 ms** (from 42.6), i.e. coupled ~1.8–2.6× warmed. The cuSPARSE PCR tridiag (2 ms) and the bandwidth floor (~3 ms dycore) are the hard floors.

`top_lever=acoustic_substep_scan_fusion · est_ceiling=~8-11x · ranked_items=14`

---

## 1. WHERE THE TIME GOES (per-phase, cited)

Per-phase isolated warmed cost (`phase_breakdown.json`, `compute_cycle_analysis.md:53-65`), coupled ~42.6 ms gate / 26.9 ms light:

| Phase | min ms | % of ~27ms light step | GB | ms/HBM-floor | calls/step | bound by |
|---|---|---|---|---|---|---|
| **Thompson microphysics** | **20.0** | (≈half of 42.6 coupled) | 1.77 | **20×** | 1 | launch/bw (85% sedimentation) |
| surface layer | 4.0 | — | 0.61 | 12× | 1 | launch |
| MYNN PBL | 2.9 | — | 1.30 | 4× | 1 | bw/launch |
| **vertical Thomas (cuSPARSE PCR)** | ~2.0 (nsys 62.8ms/36) | — | 0.060 | 29× | 2×16 | **already optimal** |
| advection tendencies | 1.0 | — | 0.66 | 2.7× | 3 | bw |
| small_step_prep | 1.0 | — | 0.76 | 2.3× | 3 | launch |
| calc_p_rho EOS | 1.0 | — | 0.037 | **48×** | 3 + 16/substep | launch |
| boundary apply | 1.0 | — | 0.87 | 2.1× | 1 | bw |
| flux-adv augment | 0.68 | — | 0.79 | 1.5× | 3 | bw |
| calc_coef_w | 0.34 | — | 0.076 | 8× | 3 | launch |
| halo apply | 0.52 | — | 0.56 | ~8 | **NO-OP** (see §5.6) |

**Dycore roofline (`roofline_costonly.json:81-91`):** 5.66 GB/step, 2.26 GFLOP/step, AI **0.40 FLOP/byte** (fp64 ridge 0.915, fp32 ridge 58.5), 18.7% HBM peak, 8.2% fp64 peak, HBM floor **3.16 ms** → wall 16.9 ms = **5.3× launch tax**.

**Kernel-level (`nsys_warmed_step_stats_cuda_gpu_kern_sum.csv`, 36 steps):**
- Genuine dominant kernels: `loop_add_fusion_9` 116,136 inst / 90.6ms; `loop_multiply_fusion_1` 47,520 / 49.2ms; `loop_subtract_fusion_1` 46,440 / 44.2ms; `loop_select_fusion_3` 22,176 / 27.8ms. **Sum ≈ 232k inst ≈ 6,450/step** of the ~6,890 — the acoustic substep arithmetic.
- `pcrGtsvBatchSharedMemKernelLoop<double>` 360 inst / 62.8ms = ~1.74 ms/step (the tridiag — already a parallel cuSPARSE PCR, NOT a lever).
- **Memory ops (`cuda_gpu_sum.csv`):** Device-to-Device memcpy **129,641 inst / 105ms = ~2.9 ms/step (~3,600/step)** — the dominant memory cost; H2D/D2H counts are autotuning/staging only (see §5.5).
- Autotuning artifacts (vanish at ≥200-step warmup, NOT real): `RedzoneAllocatorKernelImpl` 35.9%, `DelayKernel`, `xla_fp_comparison`.

---

## 2. RANKED OPTIMIZATION TABLE

Ranked by **gain × (1/effort) × (1/risk)**. Gain % is of the **coupled** step unless noted. S/M/L effort; risk = kernel-stability risk.

| # | Inefficiency | Location (file:func) | Est. gain | Effort | Risk | Category | Concrete fix |
|---|---|---|---|---|---|---|---|
| **1** | **Acoustic substep `lax.scan` not fused across iterations** — 16 substeps/step emit ~6,450 tiny dependent ~1µs kernels + ~3,600 D2D memcpys; XLA won't fuse across the scan boundary | `operational_mode.py:_acoustic_scan` (`:1318-1328` `jax.lax.scan(body,...)` no `unroll`); body = `acoustic.py:acoustic_substep_core` | **15–25%** (dycore 1.22× measured at unroll=4 → coupled prorated) | S (env flag exists) → M (bake-in) | med | fusion/launch | Set `GPUWRF_ACOUSTIC_UNROLL` default 2 (not 4: milder ~3× compile + smaller program → avoids the unroll=4 coupled OOM, `fusion_results.md:103-127`). Round-off only (rel ~1e-15, `unroll_ab_verdict.json`). Re-run the 24h coupled gate on a free GPU before flipping default. |
| **2** | **`AcousticCoreState` carry bloat** — ~60-field pytree threaded through the substep scan; ≥30 are STAGE-CONSTANT (dnw/fnm/fnp/rdnw/c1h/c2h/all msf*/cf1-3/c1f/c2f/rdn/a/alpha/gamma/ht/phb/p_base/ph_base etc.) yet copied every substep | `acoustic.py:AcousticCoreState` (`:99-257`); built in `operational_mode.py:_acoustic_core_state_from_prep` (`:906`) | **5–12%** (cuts a large share of the ~3,600 D2D carry memcpys + scan plumbing) | M | med | mem/launch | Split the scan carry into (mutable prognostics u/v/w/mu*/theta*/ph/p/ww/t_2ave ≈ 14 fields) vs **closure-captured stage constants** (everything else). XLA then threads only the evolving fields; constants become free reads. Round-off neutral (no math change). |
| **3** | **Thompson sedimentation over-resolves substeps** — 4 species × static `NSED_MAX=64` `lax.scan`, each body = 2 `concatenate` flux-shifts; typical column `nstep≈8–12` so ~5× of the 64 iterations are masked no-ops still launching kernels | `thompson_column.py:_sed_one_species` (`:1145-1213`, scan `:1209`); `_sedimentation` (`:1291`) | **8–15%** (Thompson is ~half the step; sedimentation ~85% of it) | M | med | compute/launch | (a) keep `GPUWRF_THOMPSON_SED_UNROLL=2` (already default, bit-identical, +5% shipped). (b) Lower `NSED_MAX` to a data-justified bound (e.g. 16–24, still >headroom over WRF nstep~8–12) — **flag: WRF-faithfulness** — columns needing nstep>cap silently clip (already current behaviour at 64); validate surface precip vs the precipitating Thompson oracle (`PRECIP_ORACLE_AND_IMPLICIT_SED.md`). (c) batch the 4 species' flux-`concatenate` into one shaped op. |
| **4** | **Per-step whole-state precision cast** — `.astype` over all ~60 `STATE_FIELD_ORDER` leaves every step; under `force_fp64` it's a fp64→fp64 no-op (DCE-able) but still a full elementwise pass family per non-DCE'd field | `operational_mode.py:_enforce_operational_precision` (`:498-515`), called `:2197` every step | **1–3%** | S | low | compute/launch | Skip the cast when the field is already the target dtype (trace-time `if dtype==target: continue`). Under `force_fp64` this removes the whole pass. Bit-identical. |
| **5** | **`dry_cqw` rebuilt twice per RK stage** + `zeros_like`/`ones_like` scratch rebuilt in `_acoustic_core_state` legacy helper | `operational_mode.py:_acoustic_scan` (`:1283`) AND `_acoustic_core_state_from_prep`→`:889`; legacy `_acoustic_core_state:876-902` | **<1–2%** | S | low | compute | Build `cqw_field` once per stage, pass into both the coef build and the core state. (Note: the operational path uses `_from_prep`, so the `:889` build is the live one; the `:1283` build is the second.) Bit-identical. |
| **6** | **`jnp.pad(mode="edge")` face-pairs in advance_uv** — 10 `_x/_y_face_pair_3d` pad calls per substep on the largest 3D fields (ph,p,p_base,al,alt,php ×2 axes); pads materialize a padded copy = extra HBM + a memory-op kernel each | `acoustic.py:_x_face_pair_3d`/`_y_face_pair_3d` (`:310-327`), used in `advance_uv_wrf` (`:416-473`) | **2–5%** | M | med | mem/fusion | Replace edge-pad-then-slice with direct slice + explicit boundary handling (`jnp.concatenate` of edge rows, or roll). Avoids the full padded-array materialization. Validate idealized gates (round-off). |
| **7** | **`.at[].set()` scatter in dpn build** — `_x/_y_face_pressure_dpn` build a zeros array then 2–3 `dynamic-update-slice` sets per substep | `acoustic.py:_x_face_pressure_dpn`/`_y_face_pressure_dpn` (`:330-369`) | **1–3%** | M | med | mem | Build with `jnp.concatenate([bottom, interior, top])` instead of zeros+scatter — removes the `loop_dynamic_update_slice_fusion` ops. Bit-identical (same values, different lowering). |
| **8** | **Thompson `_fill_down` + sed use `jnp.moveaxis` transposes** — nsys shows `loop_*_transpose_fusion` (`loop_and_select_transpose_fusion`, `loop_transpose_fusion_7`, `loop_divide_multiply_subtract_transpose_fusion`); pure memory shuffles, no compute | `thompson_column.py:_fill_down` (`:1069-1083` moveaxis ×2), physics column reshapes | **1–3%** | M | med | algo/mem | Keep the vertical axis in the kernel-native layout end-to-end (avoid the moveaxis round-trip) so XLA needn't transpose. Validate Thompson oracle (bit-identical if layout-only). |
| **9** | **fp32 storage on non-acoustic bandwidth-bound fields — SEQUENCED AFTER 1–3,6,7** | `precision.py:PRECISION_MATRIX` (gated fields already marked); blocked by `daily_pipeline.py:234 force_fp64=True` | **0% now → 10–25% on the bw-bound fraction AFTER fusion** | L | high | precision/mem | Once 1–3 make phases bandwidth-bound, drop `force_fp64` for the gated fields (theta/u/v/q advection inputs, Thompson hydrometeors, MYNN bulk). **fp64 acoustic island stays fp64** (mandatory). Re-run ALL skill/conservation/24h gates under the mixed mode (P1-8). High risk: boundary converts can re-cancel — measure. |
| **10** | **`pg_buoy_w` / `rhs_ph` / `diagnose_pressure_al_alt` rebuilt per RK stage** (correct cadence) but `diagnose_pressure_al_alt` is called twice (`_horizontal_pressure_gradient_tendencies` and `_acoustic_core_state_from_prep`) on overlapping inputs | `operational_mode.py:795-814` and `:951` | **<1–2%** | S | low | compute | Compute `pressure/al/alt` once per stage, reuse for both the PGF tendency and the buoyancy `grid_p`. Bit-identical. |
| **11** | **Four independent sediment scans not overlapped optimally** — comment says batched was slower, but the 4 share dz/rho and the flux-shift `concatenate` pattern | `thompson_column.py:_sedimentation` (`:1325-1328`) | **1–3%** | M | med | compute | Re-test a single (species, level) batched scan body NOW that `unroll` is on — the earlier "slower" verdict predates SED_UNROLL. Measure; keep whichever wins. Bit-identical. |
| **12** | **`run_forecast_operational` Python while-loop emits one scan PER radiation interval → compile scales with length** (96 scans at 24h); the single-scan/segmented entries fix this but the *default* `run_forecast_operational` is the compile-blowup one | `operational_mode.py:run_forecast_operational` (`:2538-2569`) | warmed ~0%; **compile/usability win** | S | low | launch | Make `run_forecast_operational_segmented` (or `_single_scan`, validated `single_scan_equiv.json`) the operational default. Removes per-interval dispatch; one compile. Bit-identical (segmented) / round-off (single-scan cond). |
| **13** | **CUDA command-buffer flag is a NET LOSS on coupled** — do NOT bake `--xla_gpu_graph_min_graph_size=1` (1.71× dynamics-only but **0.83–0.87× coupled**, `fusion_confirm_results.md:40-48`) | XLA launch env | avoid −15..−21% regression | S | low | launch | NEGATIVE lever — explicitly leave OFF for the coupled operational path. Only helps the (non-operational) dynamics-only config. |
| **14** | **`small_step_prep` / `calc_p_rho` EOS at 48× HBM floor** — tiny-data, launch-bound; called 3×/stage + 16×/substep | `core/small_step_prep.py`, `core/calc_p_rho.py`; `calc_p_rho_step` in substep `acoustic.py:765` | folded into #1/#2 | M | med | launch | These fuse for free once the substep scan is fused (#1). No separate action; listed for completeness. |

---

## 3. RECOMMENDED SEQUENCING

**Phased, not all-at-once.** The levers have a hard dependency chain (fusion must precede precision), and the kernel-stability bar requires gating each fp64-core-touching change independently.

**WAVE A — launch-count reduction (precision-invariant, the >1% bulk). Do these together, validate as one gate pass:**
1. Lever #2 (carry split — biggest structural, round-off-neutral) + Lever #1 (acoustic `unroll=2` default).
2. Lever #6 + #7 (kill the pad/scatter face-pair memory ops in the substep).
3. Lever #4 + #5 + #10 (cheap stage-invariant / no-op-cast removals — bit-identical, low risk, bank them first as quick wins).
4. Lever #3 + #8 + #11 (Thompson sedimentation: keep SED_UNROLL=2, retest batched scan + NSED_MAX bound + remove transposes).
- **Gate after Wave A:** idealized warm-bubble + Straka close gates (the fp64-core round-off vet) + the **24h coupled stability run on a FREE GPU** (the unroll=4 OOM was contention-only; unroll=2 should fit) + d02 24h skill no-regression + conservation budget. Re-profile (nsys, ≥200-step warmup) to confirm the kernel-count drop and to settle the autotuning artifacts out.

**WAVE B — make the step bandwidth-bound, THEN unlock precision:**
5. Lever #12 (segmented/single-scan as operational default — compile/usability, do anytime).
6. Lever #9 (gated-fp32 on non-acoustic bw-bound fields) — **only after Wave A**, because fp32 is 0% while launch-bound and only pays once the phases are bandwidth-bound. Highest risk; needs the full skill+conservation+24h gate suite under the mixed-precision mode (P1-8), and must measure that the fp64-island boundary converts don't re-cancel the saving.

**What unlocks what:** Wave A (fusion) is the prerequisite for Wave B (precision). Lever #1 fuses across substeps → fewer kernels for any later graph-batching AND shifts the bottleneck from launch→bandwidth, which is the precondition that makes Lever #9 (fp32 byte-halving) actually pay. Lever #13 (command buffers) stays OFF throughout (coupled regression).

**Do NOT** chain these as many tiny sprints — coupling bugs hide between them and each fp64-core touch needs the same gate suite. One Wave-A sprint, one gate, one Wave-B sprint, one gate.

---

## 4. THEORETICAL-OPTIMUM ESTIMATE (one RTX 5090)

**Hard floors (cannot be beaten without fidelity loss):**
- Dycore HBM bandwidth floor = **3.16 ms** (`roofline_costonly.json:89`; 5.66 GB / 1.792 TB/s).
- cuSPARSE PCR tridiag = **~2.0 ms** (already optimal, `nsys`).
- Thompson sedimentation is intrinsically a dependent upwind chain; even fully fused its data-movement floor is ~1 ms (`phase_breakdown.json`: 0.99 ms HBM floor) but the dependent-substep latency keeps it above that.
- fp64 compute floor = 1.38 ms (never binding; AI 0.40).

**Realistic post-Wave-A coupled per-step:** from 42.6 ms (gate) →
- dycore 16.9 → **~6–9 ms** (close most of the 5.3× launch tax; irreducible dependent stencils + the 3.16 ms bandwidth floor + 2 ms tridiag prevent reaching 3 ms).
- Thompson 20 → **~10–14 ms** (sedimentation fusion + tighter substep; the 85% sedimentation chain is dependent so it stays the single largest phase).
- surface+MYNN+boundary ~8 → **~5–7 ms**.
- → **coupled ~16–24 ms** = **~1.8–2.6× warmed** over the current 42.6 ms.

**Post-Wave-B (fp32 on bw-bound fields, IF the boundary converts don't cancel):** an additional ~10–20% on the bandwidth-bound fraction → coupled possibly **~14–20 ms**.

**Speedup vs CPU-WRF (`speedup_denominator.md`: clean 83 / realistic 123 s/fc-hr; dt=10s, 360 steps/fc-hr):**
- Current: 42.6 ms/step × 360 = 15.3 s/fc-hr → **5.4× clean / 8.0× realistic**.
- Post-Wave-A: ~20 ms/step → 7.2 s/fc-hr → **~11.5× clean / ~17× realistic** (optimistic end); conservatively ~22 ms → **~10.5× clean**.
- **≥10× clean is reachable but CONDITIONAL** on getting the dycore to ~7 ms AND Thompson to ~11 ms — both hard, both require the gates to stay green. It is NOT guaranteed, and it is achievable ONLY via launch-count reduction (Wave A), with precision (Wave B) as a secondary multiplier. The cuSPARSE PCR + bandwidth + dependent-sedimentation floors cap the absolute ceiling around coupled ~14–16 ms (~13–15× clean) — beyond that needs algorithmic fidelity tradeoffs (rejected: implicit sedimentation +47% precip, `PRECIP_ORACLE_AND_IMPLICIT_SED.md`).

**What closes the gap to the floor:** Lever #1+#2 (substep fusion + carry shrink) is the single biggest gap-closer (the 5.3× launch tax). The residual gap to the 3.16 ms bandwidth floor after fusion is irreducible dependent-stencil latency.

---

## 5. NOTES / VERIFICATIONS / OUT-OF-SCOPE (<1%, noted not actioned)

**5.1 <1% / out-of-scope items (NOTED, not actioned):**
- `halo_spec(namelist.grid)` rebuilt per stage (`_rk_scan_step:1601,1617,1636`) — Python trace-time only, **free at runtime**.
- `_with_save_family` `zeros_like` on muave each call — one small 2D alloc, DCE-friendly.
- `_theta_mass_weights` / theta limiter overhead — the limiter is now a non-load-bearing identity on physical theta (`:709-744`); the finite-checks are cheap masks. <1%.
- `jnp.argmax`/`_first_limited_cell_xyz` diagnostic in the limiter (`:595-609`) — runs every step even when no cell limited; tiny, but pure diagnostic → could gate behind a debug flag. <1%.
- Boundary `_finite_or_origin` guard family (`:2184-2196`) — ~10 small `jnp.where` masks/step; <1% and they ARE the safety net (keep).
- `lu_index` INT32 cast and 2D surface field casts in the per-step precision pass — trivial.

**5.2 Confirmed NOT levers (already optimal / would regress):**
- **Vertical implicit w/φ solve** = cuSPARSE batched PCR (`pcrGtsvBatch*<double>`), ~2 ms — already a parallel solver. A hand-fused solver changes the fp64 reduction order → not provably safe; deferred, low priority.
- **CUDA command-buffer graph flag** = NET LOSS on coupled (0.83–0.87×) — leave OFF (Lever #13).
- **Implicit (backward-Euler) sedimentation** = 2.25× kernel but +47% surface precip vs WRF oracle = FIDELITY-REJECTED. Keep gated OFF.
- **fp32 dynamics / fp32 Thompson TODAY** = ~1.00× while launch-bound (re-eval in Wave B only).

**5.3 WRF-fidelity flags (explicit tradeoffs called out):**
- Lever #3b (lower `NSED_MAX`): a tighter static substep cap silently clips columns whose WRF `nstep` exceeds it (same class as the existing 64-cap). Must be validated against the precipitating Thompson oracle; flagged as a potential fidelity tradeoff.
- Lever #9 (gated-fp32): WRF operational default is largely single-precision, so fp32 on non-acoustic fields is arguably MORE WRF-faithful, but it changes the validated v0.9.0 fp64 trajectory → re-gate fully.
- The acoustic surface-w decoupled-wind feed (`acoustic.py:658-679`) is a KNOWN, documented fidelity deviation (stability tradeoff) — NOT a perf item; do not touch for speed.

**5.4 Device-residency VERIFIED CLEAN:** `fusion_transfer_audit.json` → **0 in-loop H2D/D2H bytes**; the nsys H2D (3,212) / D2H (4,872) instance counts are one-time input/output staging + autotuning, not per-timestep transfers. No host-sync lever exists. `donate_argnums=(0,)` already set on the public entries (in-place carry). No `.item()`/host-callback in the hot path (debug=False verified across `_physics_boundary_step`, `_advance_chunk`, `run_forecast_operational*`).

**5.5 debug=False verified end-to-end:** the only `jax.debug.print` is gated behind `if debug:` (`:1644`); operational entries call with `debug=False`. No snapshot/sanitizer/assert in the compiled path. The limiter diagnostics path (`_with_limiter_diagnostics`) is a SEPARATE entry, not on `run_forecast_operational`. Clean.

**5.6 `apply_halo` is a single-GPU NO-OP** (`halo.py:28-32`, returns state) — so the phase_breakdown "halo apply 0.52ms ×8" is isolated-jit wrapper overhead, NOT real per-step cost; the redundant calls in `_rk_scan_step` are identity (DCE'd). Downgraded to non-issue. (Future multi-GPU will make this real; out of scope for single-GPU v0.10.0.)

**5.7 RECOMMENDED FRESH PROFILING (serialize on the single GPU — manager to schedule, NOT run here):**
1. **nsys with ≥200-step warmup** on the CURRENT v0.9.0 to settle autotuning (`RedzoneAllocator`/`DelayKernel`/`xla_fp_comparison` → ~0) for a clean kernel-count baseline before Wave A.
2. **`ncu` per-kernel occupancy** on `loop_add_fusion_9` / the D2D memcpys to confirm they're the substep-scan boundary (attribution).
3. **Wave-A A/B re-profile** (unroll=2, carry-split) to measure the actual kernel-count drop + the **24h coupled stability gate on a FREE GPU** (the prior unroll=4 OOM was contention-only).
4. **Fresh 28-rank CPU-WRF wallclock** for the final speedup claim (the 83/123 s/fc-hr denominator is from two L2 runs; re-measure on a free box).

---

## 6. PROVENANCE
All numbers cited to: `proofs/perf/compute_cycle_analysis.md`, `roofline_costonly.json`, `phase_breakdown.json`, `nsys_warmed_step_stats_{cuda_gpu_kern_sum,cuda_gpu_sum}.csv`, `unroll_ab_verdict.json`, `fusion_results.md`, `fusion_confirm_results.md`, `fusion_transfer_audit.json`, `segscan_24h.json`, `warmed_timing.json`, `speedup_denominator.md`, `publish/runtime_optimization_analysis.md`, `publish/GPU_PORT_GAPS_TODO.md` (P1-8). Code: `operational_mode.py`, `dynamics/core/acoustic.py`, `coupling/scan_adapters.py`, `coupling/physics_couplers.py`, `physics/thompson_column.py`, `contracts/precision.py`, `contracts/halo.py`, `integration/daily_pipeline.py`, `runtime/operational_state.py`. Estimates marked "est." / "~"; all others artifact-cited.
