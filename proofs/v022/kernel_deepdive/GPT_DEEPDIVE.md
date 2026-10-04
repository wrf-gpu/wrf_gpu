# v0.22 kernel-performance deep-dive - GPT report

Date: 2026-06-26
Scope: read-only, CPU-only analysis. No GPU commands run. No repo files edited.

## Executive verdict

I do not find a hidden lossless "one more 2x" kernel lever in the current v0.22
state. The ruled-out briefing is internally consistent with the later proof set:
the default runtime is device-bound and latency-bound, with roughly 10k-18k tiny
dependent kernels in the 128x128 anchor, and v0.21's fused AOT work fixed
cold/warm startup rather than warm kernel throughput.

The last credible BIG runtime win is still #96/P4: dt and acoustic-substep cadence,
because it cuts whole timesteps/substeps and therefore cuts kernel count. It must
run as a CFL-first ladder, not as a blind config bump.

The last credible SMALL lossless kernel wins are bounded: BouLac O(nz) or
fp32-BouLac if the full-pipeline compile/stall pathology is solved, WRF-faithful
positive bldt/cudt cadence if a case asks for it, and a few low-single-digit
skip/audit items. Everything else either does not cut kernel count in the measured
profile or collapses into a §3 dead end.

The NEW non-lossless category is the only place where another material speed mode
could be built. It needs an explicit "operational-relaxed" acceptance tier before
implementation: multi-case 24/72/120 h skill bands, hard stability/conservation
guards, no silent clamps/fallbacks, and the user approval per lever.

Counts for manager summary: 4 big / 10 small / 9 non-lossless.

## Evidence read

- `/tmp/v022_kernelperf_deepdive_gpt.txt`
- `.agent/decisions/KERNEL-PERF-RULEDOUT-BRIEFING-v022.md`
- `.agent/decisions/KERNEL-OPTIMIZATION-FINDINGS-FINAL.md`
- `.agent/decisions/V0200-ROADMAP.md`
- `.agent/decisions/V022-ROADMAP.md`
- `.agent/decisions/BOTTLENECK-CROSS-MODEL-SYNTHESIS.md`
- `.agent/sprints/2026-06-13-v015-fable-kernel-final/CONTRACT.md`
- `proofs/perf/v015/kernel_characterization.md`
- `proofs/perf/v015/{boulac_nz_optimization,fp32_definitive_verdict}.json`
- `proofs/perf/v015/viability/VIABILITY_VERDICT.json`
- `proofs/v020/lowhang/COMBINED_SPEEDUP.md`
- `proofs/v020/fp32_integration/FP32_INTEGRATION_REPORT.md`
- `proofs/v020/instrument/REPORT.md`
- `proofs/v020/s4/S4_1KM_CAPABILITY_DEMO.md`
- `proofs/v021/canary_gate/V0210_FUSED_AOT_GATE.md`
- `proofs/v021/canary_gate/HEADLINE_COMPILE_TABLE.md`
- `proofs/v021/blocker_9nest_cheapkey/FIX_REPORT.md`
- `README.md`, `RELEASE_NOTES_v0.21.0.md`, `docs/namelist-compatibility.md`, `docs/KNOWN_ISSUES.md`
- Static source skim: `operational_mode.py`, `nested_pipeline.py`, `physics_couplers.py`, `noahmp_surface_hook.py`, `noahmp_coupler.py`, `runtime/domain_tree.py`

## Big / strategic lossless or WRF-faithful candidates

| idea | category | expected effect on latency-bound profile | risk | fidelity traded | gate needed | §3 cross-check |
|---|---|---:|---|---|---|---|
| P4 dt / n_sound CFL-first ladder | big | Real runtime win because it cuts timestep/substep count. Planning band from v0.20: likely 1.2-1.8x, stretch ~2.2x if dt and n_sound both pass. This is the only open lever that directly cuts many core kernels. | High stability risk on steep 1 km nests, vertical CFL, boundary phase, qke/PBL feedback. | None if kept inside WRF-faithful time-step/substep semantics and skill gates. | GPU ladder: dt first with baseline-like acoustic dt, then n_sound cuts only on passing dt; 6 h then 24/72/120 h all-7; per-domain CFL, boundary rings, conservation, restart/output. | Not §3 because §4 explicitly names P4 as open. It is not command-buffer capture, fp32 precision-only, or megakernel work. |
| R0 fused compile memory-fitting flag sweep | big, compile/ops | No warm-step kernel win by design; potential big compile-RAM win if fused cold compile peak drops from ~60 GB class toward ~25-35 GB while keeping fused runtime and AOT blob. | Low correctness risk if runtime HLO/blob is unchanged; medium risk of no setting helping or compile wall increasing. | None. | Cold compile wall/RSS, serialized blob validity, warm-load seconds, byte/tolerance identity, s/step == fused within 5%. | Not §3 because §3.15 explicitly leaves exactly this fused-compile memory-fitting flag sweep open. |
| Larger coherent grids / deployment scale | big, deployment | Not a code speedup. Current honest gain is capability and scale: ~1.6-2.7x vs CPU at 1 km, centered ~2x, not 6-10x. It amortizes fixed intercept but per-cell saturation is refuted. | Low technical risk, high communication risk if oversold. | None. | Canary benchmark at target grids with CPU denominator, warm timings, VRAM/RSS, transfer audit. | Not §3.16 because this uses the corrected ~2x band, not the refuted 6-10x per-cell claim. |
| Multi-GPU / sharding | big, separate milestone | Real scale path for larger grids and throughput. It can cut per-GPU wall/capacity pressure, but is not a v0.22 "missed kernel tweak". | High architecture and halo correctness risk. | None if numerically gated; likely tiered identity due decomposition/order. | ADR, halo/savepoint tests, transfer audit, multi-GPU profile, CPU-WRF/tiered/skill gates. | §3.17 says deferred, not missed. Include as possible strategic path only, not a v0.22 quick win. |

## Smaller but real lossless candidates

| idea | category | expected effect on latency-bound profile | risk | fidelity traded | gate needed | §3 cross-check |
|---|---|---:|---|---|---|---|
| BouLac O(nz) full-pipeline compile fix | small | Isolated proof: 1.30x MYNN step, ~7.5 ms saved, -387 MB. Full-step if landed likely <=1.05-1.10x depending current share. Cuts work and memory, not just flops. | Medium. Current O(nz) implementation is correct in isolation but triggers XLA slow compile/no-completion in the full pipeline. | None; oracle says machine-epsilon vs WRF. | Fix via fusion boundary/custom-call/fusion-friendly formulation; then full-pipeline compile, tiered field gate, nsys wall/kernels, VRAM. | Not §3 because §4 names BouLac O(nz) open. It is not the ruled-out loop-hoist or tiling speed path. |
| fp32-BouLac re-test after v0.20/v0.21 precision/cache changes | small | Historical isolated MYNN win ~8 ms and full 128x anchor 108 ms vs 123 ms in one A/B, but v0.15 full pipeline NO-GO due XLA compile/stall. If stall is gone, likely low-single-digit to ~1.1x. | Medium-high until the stall is disproven. | Tiered identity, not strict bitwise. No operational-relaxed trade if field gate passes. | Fresh full-pipeline compile/run stability, tiered gate, warm timing, HLO dtype/converts. | Not §3.12 because this is BouLac-specific, not "fp32 precision-only unchanged topology speeds the nest". §4 explicitly keeps fp32-BouLac open to re-check. |
| Honor positive `bldt` / `cudt` cadence when configured | small, case-dependent | If a namelist requests positive PBL/cumulus interval, current docs say port warns and runs PBL/cumulus every dynamics step. Honoring cadence would skip whole PBL/cumulus calls and cut kernel count. Default cu=0 and bldt=0 cases see no win. | Medium because held tendencies/carry semantics must match WRF. | None when matching the selected WRF namelist cadence. | Cadence-semantics proof, WRF fixture/case with bldt/cudt>0, skill/conservation, nsys kernel count. | Not §3 because this is a not-yet-implemented WRF cadence semantic, not a host/capture/precision/unroll item. |
| Nighttime all-domain SW radiation skip | small | Radiation is already held-cadence, so this only affects radiation-call steps. If every column has zero/negative coszen, skip SW solver and return zero SW heating; possible night-half SW reduction, but small end-to-end. | Low-medium. Need exact-zero equivalence at night and twilight guard. | None if bit/tiered identical to SW solver for all-night intervals. | Oracle against RRTMG SW at coszen<=0, all-night real case, field gate, profile. | Not §3.7 implicit sedimentation; this preserves radiation physics when the solar source is zero. Not command buffer. |
| Noah-MP land-index compaction / ocean bypass | small, case-dependent | Canary domains have lots of ocean. Current adapter runs sfclay all columns and Noah-MP then blends by `where(is_land, ...)`. Static land-index compaction could skip Noah-MP over water, reducing land-surface kernel wall/transients. It may not cut kernel count if gather/scatter overhead dominates. | Medium-high. Static shape per domain, land carry packing, output diagnostics, and scatter stability. | None if land columns are identical and water path stays sfclay. | CPU/GPU identity on land/water masks, profile on ocean-heavy domains, VRAM/RSS. | Not §3 column tiling; this skips physically inactive land work over water, not adding tiles. |
| XLA redzone/debug allocator audit | small | `nsys_post` shows `RedzoneAllocatorKernelImpl` as a nontrivial top kernel family in one profile. If production flags accidentally enable redzone/checking, removing it could save low-single-digit ms/step and kernel count. If it is profiler-only/no longer present, zero. | Low. Main risk is chasing a profiling artifact. | None. | A/B clean profile with production env, confirm kernel family disappears and identity unchanged. | Not §3 because this is a new environment/profiler hygiene audit, not graph capture or hoisting. |
| Wrapped slice/transpose/layout cleanup | small | Kernel characterization flagged layout/wrapped-transpose as under-measured and bounded small. Potential <1-3% if it removes repeated tiny kernels without materializing big arrays. | Medium. Layout changes can perturb fusion and identity. | None if bit/tiered green. | Top-20 kernel diff, HLO before/after, tiered gate. | Not §3.4 hoist because this is layout/slice elimination, not materializing stage constants. |
| Acoustic scan carry split retest with corrected cache-hit protocol | small | Code comment says a carry split was bit-identical but revalidation was confounded by cache-miss timing. If it reduces scan carry traffic or D2D copies without extra materialization, small wall win possible. | Medium. Prior measurement inconclusive, and carry changes can worsen XLA. | None if bit/tiered green. | Warm cache-hit A/B, D2D/kernel count, tiered gate. | Not §3.5 scan-unroll knobs; this is carry structure, not unroll. Also not §3.4 hoisting. |
| Boundary/ring specialization for domains without specified/nested boundary work | small | Many boundary paths are statically gated, but any residual ring target interpolation/pin work in non-applicable domains is pure overhead. Static specialization can remove kernels in idealized/periodic or child domains. | Low-medium. Must avoid duplicating compile keys excessively. | None. | Static HLO diff by domain type, identity, profile on all-7. | Not §3 because it removes unused static branches, not command buffers or unrolls. |
| Low-noise remeasure of root-sync cadence (`root:3`) | small, nest-host only | v0.20 lowhang saw a possible root:3 lead at H2 but an H1 outlier; not settled. It affects host sync cadence more than the §1 device-bound 128x profile, so not a core kernel win. | Low. | None. | Corpus-paused A/B on all-7, VRAM, output identity. | Not §3.1 because it is sync cadence, not whole-step graph capture. Also likely not relevant to the settled device-bound anchor. |

## Non-lossless candidates requiring an operational-relaxed tier

These are not shippable silently. Each needs the user approval and a new acceptance
tier: multi-case 24/72/120 h skill vs fp64-GPU/CPU-WRF/obs/AEMET/AIFS/ERA,
hard no-blow-up, dry-mass/water/cumulative budgets, boundary-ring growth,
restart equivalence, no new clamps, no hidden fallback, and profiler proof.

| idea | category | expected effect on latency-bound profile | risk | fidelity traded | gate needed | §3 cross-check |
|---|---|---:|---|---|---|---|
| Fast-mode PBL/surface/Noah cadence | non-lossless | Potentially material on all-7 if PBL/surface/land are called every dynamics step. Cuts whole physics calls and kernel count. | High for T2/U10/V10, low cloud, qke/PBL feedback. | Operational-relaxed; not WRF-trueness if default WRF would call every step. | bldt-like ladder: every 2/3/5 steps; held tendencies/fluxes; 24/72/120 h skill by regime. | Not §3 because non-lossless §5 explicitly opens physics call-frequency reduction. |
| Wider radiation cadence beyond WRF radt | non-lossless | Radiation is already WRF-held cadence, so only larger radt saves. Smaller core effect than PBL because radiation is infrequent, but can reduce RRTMG transient/wall. | Medium-high for diurnal T2/cloud and terrain radiation. | Operational-relaxed. | radt ladder, daytime/night/cloud regimes, radiation flux diagnostics, skill and energy budget. | §3.7 rejected implicit sedimentation for fidelity, but §5 newly opens non-lossless physics cadence. This is radiation, not sedimentation. |
| Larger dt / fewer root steps beyond strict ladder | non-lossless | Directly cuts total step count, so it is one of the few possible big wall wins. | Very high: CFL, terrain channels, pressure/geopotential boundary growth, restart/output alignment. | Operational-relaxed. | Separate from P4 lossless ladder; fail-fast CFL/stability then multi-case skill. | Not §3 because §5 explicitly opens larger acoustic dt / fewer substeps as non-lossless. |
| More aggressive n_sound reduction / adaptive per-domain n_sound | non-lossless | Cuts acoustic substep kernels. If dycore/acoustic dominates a domain, meaningful; if physics dominates, smaller. | High due acoustic stability and phase errors. | Operational-relaxed. | Per-domain n_sound ladder, acoustic CFL, pressure/dry-mass/geopotential growth, skill. | Not §3.5 scan-unroll knobs; this reduces substep count, not unrolls the same count. |
| Wider precision relaxation for slow fields | non-lossless | Can reduce VRAM and maybe wall on large grids. On single-card tiny/nest profile, broad fp32 speed is refuted/inconclusive. Candidate fields: surface/boundary/radiation/cloud optics, selected diagnostics, non-conservation accumulators only with guard bands. | Medium-high. fp32-on-nest is documented as capability-only and can be slower/OOM. | Operational-relaxed or tiered, field-dependent. | Per-field dtype matrix, HLO dtype audit, 24/72/120 h skill, conservation and limiter hit counts. | Must not be "global fp32" (§3.11) or "precision-only nest speedup" (§3.12). This is narrower and accepted only for capability/fast-mode evidence. |
| Microphysics active-column / species threshold skip | non-lossless | If hydrometeor species are tiny/absent, skip Thompson substeps/species and cut kernels/wall in dry regimes. Helps only if implemented as static/compact active work, not masks that still launch all kernels. | High for precip/cloud fidelity, accumulated precip, positivity. | Operational-relaxed. | Dry/moist/precip cases, precip object metrics, water budget, threshold sensitivity. | Not §3.7 implicit sedimentation unless it changes sedimentation physics. If it does, mark as same family and only evaluate under §5 non-lossless. |
| Vertical-level reduction / vertical coarsening fast mode | non-lossless | Big potential: less vertical work in every dycore/physics column and fewer tridiagonal levels. Cuts kernel work and memory. | Very high over complex terrain, PBL height/cloud/tropopause, pressure gradients. | Operational-relaxed. | New product mode, remap/orography handling, multi-case skill, conservation. | Not in §3 because it is a new lower-fidelity product tier, not a WRF-faithful optimization. |
| Moisture species / aerosol-number reduction | non-lossless | Can cut microphysics/radiation state and kernels, especially Thompson aerosol/number paths. | High for cloud/precip skill. | Operational-relaxed. | Species-ablation matrix, precip/cloud metrics, water budget. | Not §3 unless it becomes implicit sedimentation; then evaluate only as non-lossless. |
| Spatial/domain simplification for operational product | non-lossless | Fewer 1 km nests, smaller inner domains, or coarser outer cadence cuts total domain-step count. This is often larger than a kernel tweak. | High product-scope risk: local wind/cloud detail. | Operational-relaxed/product-tier. | Side-by-side station/grid skill, object/neighborhood cloud/wind metrics, cost/performance. | Not §3.16 if framed as product scope, not false 6-10x per-cell scaling. |

## Ideas that collapse into ruled-out dead ends

| idea | why it collapses |
|---|---|
| Whole-step CUDA graph / command buffer as the next big win | §3.1: launches collapsed but wall was neutral in 4 A/B pairs because the clean profile is device-bound. |
| Command-buffer global flag | §3.2: measured -15% to -21% coupled. |
| Conditional-while capture / data-dependent early-out for high-iteration loops | §3.3: +17% wall. MYNN convergence early-out must not be re-proposed unless a new measured flattening path avoids the same failure. |
| Hoisting denominators/stage constants out of scans | §3.4: anti-optimization, +5.5%, materialized arrays cost more than fused recompute. |
| Scan-unroll knobs for Thomas/sedimentation/acoustic as a broad lever | §3.5: measured ~0 in-program. |
| cond50 Python unroll without niter cut | §3.6: loss. niter=16 is already shipped. |
| Implicit sedimentation as WRF-faithful | §3.7 rejected. It can only reappear as an explicit non-lossless fast-mode candidate. |
| Pallas/megakernel as a v0.22 core architecture project | §3.8 and v0.20: S7 structural fusion is done; full small-GPU megakernel mode is niche, high effort, and not the deployment-scale missing piece. |
| XLA tridiagonal_solve / cuSPARSE swap for acoustic/MYNN | §3.9 and §3.10: acoustic swap high numerical risk, MYNN cuSPARSE already optimal. |
| Global fp32 or fp32 precision-only nest speedup | §3.11 and §3.12: invalid or refuted. Current fp32 value is VRAM/capability, not single-card speed. |
| 2x+ tiny single-card canary nest speedup by tuning | §3.13: not reachable in measured profile. |
| De-fuse default for runtime | §3.14 and v0.21 docs: de-fuse is low-RAM compile/debug path; runtime regresses/OOM risk. |
| 6-10x large-grid per-cell saturation | §3.16: refuted; honest deployment number is about 2x band. |

## Top-5 recommendations

1. Run P4 dt/n_sound as the top runtime sprint. It is the only remaining lossless/WRF-faithful item with a plausible material wall-clock win because it cuts step/substep count.
2. Run v0.22 R0 fused compile memory-fitting sweep exactly as scoped. It does not speed warm kernels, but it is the only open path to lower fused cold-compile RAM without de-fuse runtime regression.
3. Scope a BouLac compile-path fix, not another BouLac algorithm sprint. The algorithm is already proven in isolation; the blocker is full-pipeline XLA compile/stall.
4. Build a cadence package: first WRF-faithful positive `bldt`/`cudt` semantics, then a separate non-lossless PBL/surface/Noah cadence ladder under operational-relaxed gates.
5. Define the operational-relaxed tier before implementing any non-lossless speed mode. Without that tier, the non-lossless list is not actionable and risks silently moving the product definition.

## Handoff

- Objective: independent v0.22 kernel-performance deep-dive, cross-checked against the ruled-out briefing.
- Files changed: `/tmp/v022_kernelperf_deepdive_gpt_REPORT.md` only. No repo files edited.
- Commands run: read-only `sed`, `rg`, `find`, `wc`, `jq`, `git status`, and `tmux list-panes`; no GPU/profiler/benchmark commands.
- Proof objects produced: this report. No new performance measurements because the task was CPU-only/read-only and the GPU was busy.
- Unresolved risks: source worktree is dirty with pre-existing changes; analysis did not validate current code by running model tests. Some small items need fresh GPU A/B to separate real wins from profiling artifacts.
- Next decision needed: choose whether v0.22 prioritizes runtime P4 first or compile-RAM R0 first; they are independent but contend for GPU gate time.
