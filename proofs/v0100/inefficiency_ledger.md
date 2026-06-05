# v0.10.0 Completed Inefficiency Ledger

Exit-gate evidence: every v0.10.0 Phase-1/Wave-scope item with its final
disposition (REMOVED / measured-gain / IRREDUCIBLE / NO-GO /
`<1%-skipped` / daily-only), each backed by evidence. This ledger is complete
through Wave-B3.

## Workload + baseline note (IMPORTANT)

The shipped `proofs/perf/segscan_24h.py` defaults to the **L3** run_id
(`wrf_l3/20260521_18z_l3_24h_...`). On the current code+toolchain (JAX 0.10.0,
CUDA 13.2, driver 595, RTX 5090) the **d02 extraction of that L3 init goes NaN
within 1 forecast hour** (`proofs/perf/v0100_segscan_1h_before.json`: all_finite
false, status FAIL) — it is the known-OPEN v0.9.0 "d03-1km / L3 steep-terrain
dynamics instability" carry-over, NOT a Wave-A regression (reproduced on PRISTINE
v0.9.0 `016d993` with my source edits stashed). The stale committed
`proofs/perf/segscan_24h.json` (45.5 ms, PASS) was generated months earlier at
commit `a643827` on older dycore code and is NOT a valid current baseline.

The v0.9.0 GREEN d02 stability + skill path uses the **L2** run
`wrf_l2/20260521_18z_l2_72h_20260522T133443Z`
(`proofs/v090/d02replay_2to3h_reverify.json` = FINITE_THROUGH_3H_PLUS;
`proofs/v090/speedup_d02_72h/pipeline_run_l2_d02.json` = all_finite True). Wave-A
therefore uses that **L2 d02 init** as the stable, validated baseline/gate
workload via `proofs/v0100/wave_a_gate.py`.

## Items (Opus #1-14 + GPT #1-30 deduped)

| # | Item | Disposition | Evidence |
|---|---|---|---|
| Opus#1 | Acoustic substep `lax.scan` no `unroll` hook | **ADDED hook; default=1 (unroll>1 not worth it on coupled)** | `_acoustic_unroll()` added (GPUWRF_ACOUSTIC_UNROLL, default 1); wired `unroll=` into the substep scan. A/B on coupled L2 d02 (proofs/v0100/wave_a_unroll_ab_verdict.json): unroll=1=74.12 ms/step (compile 116s), unroll=2=73.44 ms/step (compile 186s). unroll=2 = +0.9% warmed (NOISE) at +60% compile -> NOT worth it on the coupled path (the step is not launch-bound on the acoustic fraction; Thompson sedimentation ~46% dominates; the v0.9.0 dycore-only 1.22x does NOT translate to coupled). Hook retained for future dycore-only/Phase-2+ use; **default kept at 1**. |
| Opus#2 | AcousticCoreState carry bloat (~50 stage-const leaves threaded through substep scan) | **REVERTED (kept out of Wave-A; deferred)** | Implemented the carry-split (thread only the 19 evolving leaves, close over the ~50 constants, reconstruct the full state in-body) and it was BIT-IDENTICAL on the idealized gates (worst_reldiff=0.0). Its compile cost was ~the same as the simple carry (90-step coupled fp64 compile: carry-split 132.4s vs simple 123.4s -- ~7% more, within run-to-run noise). The warmed-step A/B was INCONCLUSIVE: my first timing attempts were confounded by a measurement artifact (the FIRST warm `_advance_chunk` call with a new traced start_step triggers a one-off XLA recompile ~2000 ms/step; only passes >=2 are true cache hits ~74 ms/step -- see proofs/v0100/ab_u1.json samples). I initially misread that recompile as a carry-split hang and reverted. Re-validating the carry-split's true warmed step cleanly is deferred; the simple full-pytree carry (XLA hoists the stage-invariant leaves itself) is the proven, bit-identical, non-regressing path shipped in Wave-A. _ACOUSTIC_EVOLVING_FIELDS retained as documentation. RE-TEST in a follow-up with the corrected min-of-cache-hit-passes methodology. |
| Opus#4 / GPT#20 | Per-step whole-State precision cast emits no-op converts under force_fp64 | **REMOVED** | skip `.astype` when dtype already matches; under force_fp64 with a carried all-fp64 State this emits ZERO converts (was 26). Bit-identical (idealized worst_reldiff=0.0). |
| Opus#5 | `dry_cqw` rebuilt twice per RK stage | REMOVED | reuse `acoustic.cqw` at `_finish_rk_stage_acoustic`; bit-identical (dry_cqw is a pure shape+dtype constant) |
| Opus#6 | `jnp.pad(edge)` face-pairs in advance_uv (10/substep) | **REMOVED** | edge-pad+slice -> concatenate; verified BIT-IDENTICAL to the pad form at the helper level (x/y 2d+3d, jnp.array_equal True) and end-to-end on idealized (worst_reldiff=0.0). |
| Opus#7 | `.at[].set()` dpn scatters in advance_uv | **REMOVED** | zeros+`.at[].set` scatters -> single concatenate([bottom,interior,top]); verified BIT-IDENTICAL (top_lid True+False, jnp.array_equal True) and idealized worst_reldiff=0.0. |
| Opus#10 | `diagnose_pressure_al_alt` "called twice on overlapping inputs" | DEFERRED / not-redundant | the two calls use DIFFERENT `base` args (None vs stage_base) in different functions on different inputs -> NOT the same computation; merging would change values. Analysis over-stated the overlap. <1% claimed; left as-is. |
| GPT#3 | Hourly full-State D2H finite checks | **NOT WORTH IT (sized: 0.22%)** | wave_a_host_breakdown.json (L2 d02, 1 forecast hour): finite_summary full-State D2H = **0.067 s = 0.22% of the hour**. Negligible -> NOT worth a device-side reduction. The non-forecast host share is 10.95%, dominated by M9 RRTMG surface-diagnostic RECOMPUTE (2.99 s, ~9%) -> that is GPT#14 (reuse held diagnostics), deferred to Phase 3, not GPT#3. |
| GPT#7 | redundant stage-entry/exit halos | NOT-AN-ISSUE (single-GPU no-op) | `apply_halo` returns the state unchanged on single-GPU (halo.py); the per-stage calls are identity and DCE'd. Confirmed by Opus phase-1 analysis §5.6. No action (would become real on multi-GPU; out of scope for v0.10.0). |
| Opus#13 / GPT | command-buffer flag | OUT OF SCOPE (negative lever) | net loss on coupled (-15..-21%); explicitly OFF |
| Opus#3 / GPT#2 | Thompson NSED_MAX lower | **REMOVED / SHIPPED in Wave-B1 (default cap=16)** | `NSED_MAX` default changed 64 -> 16, env override retained. Wet-column histogram (`proofs/v0100/thompson_nstep_histogram_graupel_wet.json`) had max/P99/P99.9 nstep <=2 and zero clips at cap 16. Precip oracle cap16 vs cap64 is bit-identical for surface precip and precip rate with zero clips (`proofs/v0100/wave_b1_nsed16_precip_oracle.json`). 24h Wave-A L2 d02 cap16 vs cap64 final T2/U10/V10, water fields, and precip accumulators are bit-identical (`proofs/v0100/wave_b1_nsed16_skill_24h.json`, `proofs/v0100/wave_b1_nsed16_conservation.json`). Fresh warmed timing: 74.25 -> 64.76 ms/step, **12.78% coupled gain / 1.146x** (`proofs/v0100/wave_b1_nsed16_timing.json`). |
| Opus#9 / GPT#16 | gated-fp32 | DEFERRED to Phase 5 (Wave B) | precision sequenced after fusion; 0% while launch-bound |

## Wave-A summary (what shipped, what didn't, evidence)

**SHIPPED (bit-identical, no fidelity loss, non-regressing):**
- Opus#4/GPT#20 cast-skip — REMOVED (26->0 no-op converts under force_fp64)
- Opus#5 cqw-reuse — REMOVED (dry_cqw once/stage)
- Opus#6 face-pairs concatenate — REMOVED (pad materialise gone)
- Opus#7 dpn concatenate — REMOVED (scatter gone)
- Opus#1 acoustic unroll hook — ADDED, default=1

All verified BIT-IDENTICAL on idealized warm-bubble + Straka (worst_reldiff=0.0),
so d02 skill + conservation + the 24h trajectory are IDENTICAL to v0.9.0.

**MEASURED-NOT-WORTH-IT (kept out, evidence):**
- Opus#1 unroll>1: unroll=2 = +0.9% warmed (noise) at +60% compile on the coupled
  path -> default kept at 1 (wave_a_unroll_ab_verdict.json).
- Opus#2 carry-split: reverted (idealized bit-identical, but warmed A/B confounded
  by a recompile artifact and not cleanly re-validated; the simple carry is the
  proven path). RE-TEST deferred.

**DEFERRED to later phases (with reason):**
- Opus#9/GPT#16 gated-fp32 -> Phase 5 (precision sequenced after fusion; 0% while
  launch-bound — and Wave-A confirms the coupled step is bandwidth/dependent-chain
  bound, not acoustic-launch bound, which is exactly why fp32 on the bw-bound
  fields is the next real lever).
- GPT#3 finite-summary D2H -> sized by wave_a_host_breakdown.json (Phase-1 host).
- GPT#5 physics surface+MYNN fusion, GPT#19 physics moveaxis -> Phase 3.

**NOT-AN-ISSUE / OUT-OF-SCOPE:** Opus#10 (not redundant), GPT#7 halos (single-GPU
no-op), command-buffer flag (negative lever, OFF).

**KEY WAVE-A FINDING (sizes the rest of v0.10.0):** the coupled warmed step
(~74 ms) did NOT move under the acoustic launch-count reductions because it is
bound by Thompson sedimentation (~46%) + the HBM/tridiag floors, NOT by the
acoustic substep launch count. The substantive warmed speedup therefore lives in
Phase 3 (physics fusion), Phase 4 (Thompson), and Phase 5 (precision on the
bandwidth-bound fields), exactly as the super-plan sequenced. Wave-A's contribution
is the bit-identical, no-fidelity-loss cleanup + the unroll hook + the empirical
proof of where the time actually is.

## Wave-B1 Thompson NSED16 closeout

**SHIPPED:** Thompson faithful sedimentation now defaults to `NSED_MAX=16`
(`GPUWRF_THOMPSON_NSED` still overrides it). This removes the static 64-iteration
masked sedimentation scan overhead while keeping the same clip behavior for any
future pathological column that needs more than the cap.

**Gate evidence:** precip oracle cap16 vs cap64 is bit-identical for RAINNCV /
surface precip rate and zero-clip (`wave_b1_nsed16_precip_oracle.json`). The 24h
Wave-A L2 d02 fidelity comparison is cap16-vs-cap64 bit-identical for T2/U10/V10,
qv/qc/qr/qi/qs/qg, RAINNC, and RAINC (`wave_b1_nsed16_skill_24h.json`,
`wave_b1_nsed16_conservation.json`). The local CPU-WRF corpus for the Wave-A run
does not contain the +24h `wrfout_d02` truth file (it stops at +19h), so the 24h
skill proof records no-regression by direct cap identity rather than an absolute
+24h CPU-RMSE value.

**Measured gain:** fresh cache-hit timing, first sample discarded, gives cap64
`74.25 ms/step` and default cap16 `64.76 ms/step`: **12.78% coupled gain** /
`1.146x` (`wave_b1_nsed16_timing.json`).

## Wave-B2 MYNN/PBL frontier closeout (the 45.5% / 33.8 ms lever)

**OUTCOME: IRREDUCIBLE. No source change ships — and that is the correct,
honest result.** The largest measured coupled phase (MYNN/PBL = 45.5% / 33.8 ms)
is, on profiling, **95% genuine dependent closure compute** with no safe,
beneficial mechanical inefficiency to remove. Both candidate mechanical levers
were measured and REJECTED on the fidelity-paramount bar.

**STEP 1 — profile (`wave_b2_mynn_profile.json`, `wave_b2_mynn_internal.json`):**
block decomposition on the GREEN d02 L2 init (66×159×44, fp64, EDMF on):

| Component | min ms | share |
|---|---:|---:|
| surface+MYNN block (matches the wave_b_scope 45.5% toggle) | 33.9 | 100% |
| MYNN closure kernel only (`step_mynn_pbl_column`) | **32.2** | **95.0%** |
| surface_layer compute (`surface_adapter`) | 1.94 | genuine surface physics |
| mechanical wrappers (column build + reassemble + flux read) | 1.14 | the only "mechanical" cost |
| removable-mechanical envelope (block − closure − surface) | **~0** | within measurement noise |

Closure-kernel internal phases (incremental, all straight-line vectorized — no
tunable iteration): `mym_turbulence` 7.34 ms, `mym_predict_qke` 0.51 ms,
**EDMF mass-flux 15.66 ms** (the single largest; `vmap(cols) ∘ vmap(8 plumes) ∘
lax.scan(~42 levels)` — a dependent vertical updraft recurrence),
`apply_mean_tendencies` 7.88 ms (the 4 implicit u/v/θ/qv solves). All vertical
solves use `jax.lax.linalg.tridiagonal_solve` (the XLA primitive, already optimal
— same class as the dycore PCR solve). The closure-kernel HLO is 96 fusions / 1
internal transpose: XLA already fuses the elementwise work; layout cost is
negligible.

**STEP 2 — the two candidate mechanical removals, both REJECTED (fidelity-paramount):**

| GPT#5 / Opus#8 lever | Disposition | Evidence |
|---|---|---|
| **Surface+MYNN State round-trip fusion** (run `surface_layer→SurfaceFluxes` in memory, feed MYNN directly, skip the `.replace`+`asarray` of the 7 flux handles + the duplicate column-view build) | **REJECTED — not-worth-the-risk** | `wave_b2_fusion_ab.json`: block gain **0.09%** (0.032 ms) AND **NOT bit-identical** (the prognostics differ at `max_reldiff=0.0` — i.e. signed-zero / op-reorder, exactly the round-off-level perturbation the strict MYNN gate forbids). Surface flux scalars stay exactly equal; only the float-op ordering shifts. A <0.1% gain that breaks bit-identity in a PBL scheme that drives T2/winds/PBLH is a clear fidelity-over-speed reject. Also it would require a special-case code path valid ONLY for the default (MYNN-sfclay + MYNN, no-Noah-MP) suite — adding maintenance/fidelity risk for ~0 gain. |
| **EDMF level-scan `unroll=4`** (round-off-NEUTRAL XLA codegen on the 15.66 ms largest phase — the same lever class Wave-A proved bit-identical for the acoustic substep) | **REJECTED — regression** | `wave_b2_edmf_unroll_ab.json`: block gain **−0.07%** (a slight REGRESSION) AND not bit-identical. The scan is nested inside a double `vmap` (columns × plumes); XLA already schedules it well and the unroll only perturbs the reduction order. No launch-count win exists here. |

`_to_columns`/`_from_columns` real transposes (Opus#8 / GPT#19 / Phase-0 HLO):
confirmed 1 transpose each, but the column-build + reassemble cost (1.14 ms total
for the whole MYNN block) is <1.7% of the block and is consumed inside the
fused-or-not measurement above — there is no separable transpose-removal win on
the warmed step.

**STEP 3 — gates (strict; MYNN drives T2/winds/PBLH; `wave_b2_mynn_gates.json`):**
- **Idealized warm-bubble + Straka density-current: BIT-IDENTICAL** to the
  Wave-A `u1` baseline (worst_reldiff=0.0 across 12 dynamical scalars,
  `wave_b2_idealized_roundoff.json`) — the dycore is untouched, by construction.
- **24h coupled stability (L2 d02 GREEN, guards ON, segmented):
  PASS / all-finite / physically-plausible** through the full 8640 steps
  (`wave_b2_gate_24h.json`: u/v ±30 m/s, w ±7 m/s, θ 291–493 K, qv 0–0.018,
  μ/p positive). Since Wave-B2 ships ZERO source change, T2/U10/V10/PBLH/Q2/HFX,
  conservation and the water budget are **bit-identical** to the pre-Wave-B2
  (= Wave-B1 NSED16) trajectory by construction.

**STEP 4 — disposition of the 45.5% lever:** **IRREDUCIBLE / not-worth-the-risk.**
The MYNN/PBL share is faithful WRF MYNN2.5 + MYNN-EDMF + 4 implicit (already-optimal
XLA-tridiag) vertical solves, evaluated over 461,736 columns. It is a dependent,
bandwidth/compute-bound closure with no removable mechanical inefficiency that is
both safe (bit-identical) and beneficial (>1%). The only paths to cut it further
are algorithmic fidelity tradeoffs (a cheaper closure, fewer plumes, a coarser
mixing-length, dropping EDMF) — all REJECTED by the fidelity-paramount exit-gate.
The EDMF being inactive on stable/marine columns (`active` mask) cannot be skipped
under GPU SIMT vmap without divergent-branch overhead that costs more than it
saves. **Warmed coupled step: before = after = 64.76 ms** (`wave_b2_mynn_timing.json`).

**Ceiling impact:** the super-plan's conservative ceiling (`53–56 ms/step`,
`1.32–1.40×`) assumed MYNN/PBL would give back 25–33% of its wall; the profile
shows that assumption is **false** — the MYNN wall is ~95% irreducible closure.
With Wave-B1 (NSED16, 1.15×) shipped and Wave-B2 yielding 0 warmed gain, the
remaining realistic warmed levers are Phase-5 precision (gated-fp32 on the
bw-bound non-acoustic fields — but Wave-B scoping already measured fp32 at
−0.30% / NO-GO now) and the daily-wall M9/RRTMG diagnostic reuse (GPT#14, NOT a
warmed-step lever). The honest read: **the 1.32–1.55× warmed ceiling is NOT
reachable via MYNN**; v0.10.0's shipped warmed gain is the Wave-B1 1.15×, and the
MYNN frontier is closed as irreducible.

## Wave-B3 final lane closeout

**MYNN cross-check:** GPT independently rechecked the Wave-B2 MYNN disposition
and confirmed it. `proofs/v0100/wave_b3_mynn_crosscheck.json` records the
specific checks:
- EDMF mass-flux is a dependent vertical updraft recurrence
  (`vmap(columns) x vmap(8 plumes) x lax.scan(levels)`), with no faithful
  batching/unroll win; B2 unroll A/B regressed and was not bit-identical.
- The four mean-tendency vertical solves use `jax.lax.linalg.tridiagonal_solve`
  through `physics/tridiagonal_solver.py`; there is no lower-risk mechanical
  replacement.
- Layout/materialization costs were already inside the rejected B2 fusion A/B
  (`0.09%` block gain, not bit-identical). No separable `>1%` warmed MYNN lever
  was found.

**Daily-wrapper changes shipped:** B3 removed only output-wrapper duplicate work,
not forecast numerics.
- M9/Q2 duplicate recompute removed: `M9Diagnostics` now carries `q2`, so
  `_surface_diagnostics_for_output` no longer reruns `surface_layer_diagnostics`
  solely for Q2.
- Writer HFX/LH fallback solve skipped when M9 diagnostics supply HFX/LH/UST.
  Gate tolerance was declared before final comparison: `HFX/LH atol=2e-3 W m-2`;
  measured max abs diffs were `HFX=9.46e-4`, `LH=1.06e-3`, all other fields
  bit-identical.
- Static wrfout grid fields cached once per run:
  `XLAT/XLONG/XLAT_U/XLONG_U/XLAT_V/XLONG_V/HGT/ZNU/ZNW/MAPFAC_M/MAPFAC_U/
  MAPFAC_V/F/E/SINALPHA/COSALPHA/P_TOP`. State-derived land fields are not cached.

Gate evidence:
- `proofs/v0100/wave_b3_daily_wrapper_gains.json`: PASS. Warmed L2 d02 daily-hour
  denominator `24.334s -> 24.128s`, saving `0.206s/hour = 0.848%`.
  M9/Q2 reuse saved `0.106s/output = 0.437%` of daily wall; output pack/static
  cache path saved `0.099s/output = 0.407%` of daily wall.
- `proofs/v0100/wave_b3_output_parity.json`: PASS on prepared payload and NetCDF
  variables. All fields are bit-identical except HFX/LH within the declared
  tolerance above.

Important no-go: the forecast carry holds `rthraten`, which is a radiative
theta-tendency, **not** the SWDOWN/GLW diagnostic value. B3 therefore did not
pretend `rthraten` can replace SWDOWN/GLW. Full RRTMG diagnostic side-channel
reuse would require changing the forecast carry/order and did not clear the
low-risk wrapper-only bar.

## Final completion matrix

| Item | Final disposition | Evidence |
|---|---|---|
| Opus#1 / GPT#1 acoustic scan unroll/fusion | **Hook added; default unroll kept 1; unroll>1 `<1%-skipped`** | `wave_a_unroll_ab_verdict.json`, `wave_b_scope.json`: unroll=2 `0.67-0.9%` with much higher compile cost. |
| Opus#2 / GPT#13 acoustic carry/state hot-cold split | **NO-GO for v0.10.0** | Carry split was bit-identical in idealized but warmed A/B was confounded and no clean benefit was proven; full State split is ADR-level/high blast radius. |
| Opus#3 / GPT#2 Thompson NSED cap | **REMOVED / SHIPPED, 12.78% warmed gain** | `wave_b1_nsed16_timing.json`, `wave_b1_nsed16_precip_oracle.json`, `wave_b1_nsed16_skill_24h.json`, `wave_b1_nsed16_conservation.json`. |
| Opus#4 / GPT#20 no-op precision casts | **REMOVED** | Wave-A HLO/proofs: force-fp64 no-op converts removed, idealized bit-identical. |
| Opus#5 dry_cqw/stage constant rebuild | **REMOVED where live and safe** | Wave-A code + idealized bit-identical gates. |
| Opus#6 pad(edge) face-pairs | **REMOVED** | Wave-A helper parity and idealized bit-identical gates. |
| Opus#7 dpn scatter construction | **REMOVED** | Wave-A helper parity and idealized bit-identical gates. |
| Opus#8 / GPT#19 physics layout transposes | **NO-GO / no separable >1% lever** | B2 MYNN profile: closure HLO `1` transpose; column build/reassemble included in rejected fusion A/B. |
| Opus#9 / GPT#16 gated fp32 now | **NO-GO** | `wave_b_scope.json`: fp64 `74.35ms`, gated-fp32 `74.57ms`, `-0.30%` and dtype warning. |
| Opus#10 / GPT#12 pressure/EOS duplicate recompute | **NO-GO / not redundant** | Ledger analysis: calls use different base args/inputs; merging would alter math. |
| Opus#11 / GPT#26 Thompson species batching/narrow fusion | **NO-GO after cap; rejected prior batching** | `gpt_phase1_analysis.md`: four-species batch regression; NSED16 was the faithful shipped Thompson lever. |
| Opus#12 / GPT#18 segmented/single-scan default | **REMOVED for long-run compile/usability where already used; no warmed claim** | `run_forecast_operational_segmented` is the long-run path used by B1/B2/B3 gates; warmed throughput unchanged. |
| Opus#13 / GPT#15 command-buffer flag | **NO-GO / negative lever** | Coupled path measured `-15..-21%`; left off. |
| Opus#14 small-step prep/calc_p_rho launch floor | **No separate action; folded into acoustic work and then `<1%-skipped`** | Wave-A/B showed coupled path not acoustic-launch-bound after Thompson/MYNN sizing. |
| GPT#3 finite-summary full-State D2H | **`<1%-skipped`** | `wave_a_host_breakdown.json`: `0.067s = 0.22%` of forecast hour. |
| GPT#4 output packer | **Daily-only partial REMOVED; full device-side single-get packer `<1%-skipped`** | `wave_b3_daily_wrapper_gains.json`: output prepare `0.103s -> 0.004s`, but only `0.407%` daily-wall gain after low-risk removals. |
| GPT#5 surface+MYNN fusion / PBL side-channel | **IRREDUCIBLE / rejected** | `wave_b2_fusion_ab.json`: `0.09%` block gain and not bit-identical; `wave_b3_mynn_crosscheck.json`. |
| GPT#6 resident daily carry-threaded driver | **NO-GO for v0.10.0** | High semantic/radiation/land-refresh risk; not a low-risk wrapper lever. Existing segmented carry path remains. |
| GPT#7 redundant halos | **NOT-AN-ISSUE single-GPU** | `apply_halo` single-GPU identity; DCE'd. |
| GPT#8 / GPT#21 metric, dry, zero, mass-face hoists | **REMOVED only where proven; rest `<1%-skipped`/NO-GO** | cqw live removal shipped; remaining pressure/mass hoists lacked a safe separable >1% proof. |
| GPT#9 `_advance_chunk` donation | **NO-GO / already constrained by async snapshots and aliasing** | Public donating entry exists; B3 direct full-hour probe hit duplicate-buffer donation on aliased State, confirming this is not a casual wrapper change. |
| GPT#10 / GPT#25 boundary apply/interpolation specialization | **NO-GO** | Boundary/nested `ph/w` sensitivities are high risk and outside low-risk v0.10 final lane. |
| GPT#11 production guards/limiters | **NO-GO / keep safety net** | Guards are operational safety; no evidence of >1% safe removable cost. |
| GPT#14 M9/RRTMG diagnostics | **Daily-only partial REMOVED; full RRTMG reuse NO-GO** | `wave_b3_daily_wrapper_gains.json`: Q2 duplicate recompute removed. `rthraten` is not SWDOWN/GLW, so radiation diagnostic reuse fails closed. |
| GPT#17 `time_utc` compile-cache normalization | **NO-GO / cold-only** | Super-plan notes prior dynamic-clock path failed bit-identical by `4.46 Pa p_pert`; requires RMSE-equivalence decision outside v0.10 final lane. |
| GPT#22 RRTMG g-point/transient layout | **NO-GO** | High-risk radiation rewrite; B3 only removed safe output-wrapper duplicate work. |
| GPT#23 duplicate host sync/blocking | **`<1%-skipped` after B3** | Output wrapper total after B3 below 1% daily-wall threshold for the remaining full packer/sync path. |
| GPT#24 static-grid cache | **REMOVED / daily-only** | `build_wrfout_static_field_cache` caches 17 grid-static fields; parity PASS in `wave_b3_output_parity.json`. |
| GPT#27 acoustic boundary target specialization | **NO-GO** | Nested/specified boundary correctness risk; no low-risk final-lane change. |
| GPT#28 restart/scoring/repeat probes | **Production-disabled / no kernel action** | Daily config defaults `score=False`, `repeat=False`, `restart_at_hour=None`; validation probes remain opt-in. |
| GPT#29 zero/save-family buffer reuse | **`<1%-skipped`** | Small zero/save-family fields; no >1% proof, no safe final-lane action. |
| GPT#30 profiling artifact risk | **CLOSED by v0.10 proof sequence** | Phase-0/Wave-A/B timings used cache-hit/min-of-warmed methodology; B3 proof states timing protocol. |

## Final v0.10.0 exit disposition

The inefficiency ledger is complete. The warmed-kernel shipped gain is Wave-B1
Thompson NSED16 (`12.78%`, `1.146x`). Wave-B2/B3 found no additional faithful
`>1%` warmed-kernel lever. B3 ships daily-wrapper-only cleanup with a measured
`0.848%` warmed daily-hour gain and output parity PASS; the remaining full device
packer and host-wrapper work is below the 1% exit threshold and is explicitly
skipped. The MYNN/PBL frontier is confirmed irreducible for v0.10.0.
