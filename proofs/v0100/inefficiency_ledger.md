# v0.10.0 Wave-A Inefficiency Ledger

Exit-gate evidence: every Wave-A item with its final disposition (REMOVED /
measured-gain / DEFERRED-to-PhaseN / not-worth-risk), each backed by evidence.
Wave-A scope = launch-count reduction + low-risk, PRECISION-INVARIANT (no fp32,
no Thompson NSED change, no command-buffer flag).

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
| GPT#3 | Hourly full-State D2H finite checks | <TBD> | daily-wrapper host breakdown sizes it; device-side scalar reduction if material |
| GPT#7 | redundant stage-entry/exit halos | <TBD> | single-GPU `apply_halo` is identity no-op (halo.py); DCE; verify HLO |
| Opus#13 / GPT | command-buffer flag | OUT OF SCOPE (negative lever) | net loss on coupled (-15..-21%); explicitly OFF |
| Opus#3 / GPT#2 | Thompson NSED_MAX lower | DEFERRED to Phase 4 | graupel wet-column evidence absent (phase0 histogram); not zero-clip-safe globally |
| Opus#9 / GPT#16 | gated-fp32 | DEFERRED to Phase 5 (Wave B) | precision sequenced after fusion; 0% while launch-bound |

(Filled in with measured gains after the AFTER profiles land.)
