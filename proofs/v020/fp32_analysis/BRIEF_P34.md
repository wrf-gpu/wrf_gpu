# DEEP ANALYSIS BRIEF — v0.20.0 points 3 & 4 (STRUCTURAL speedup + timestep/cadence)

You are ONE of TWO independent analysts (the other is the other model). You just wrote the
fp32 analysis (point 1). Now deep-analyze POINTS 3 & 4 of the v0.20.0 roadmap. PLANNING
ONLY: no GPU runs, no source edits. Independent report.

## WHY THIS MATTERS (read your + the peer's fp32 reports first)
Read proofs/v020/fp32_analysis/OPUS_FP32_PLAN.md AND GPT_FP32_PLAN.md. BOTH concluded the
nested all-7 max_dom=9 step is **launch-count / HBM / occupancy bound (STRUCTURAL), not
fp64-ALU bound** — so precision alone caps at ~1.1-1.4x on the nest. THEREFORE points 3 & 4
(structural + algorithmic) are the REAL near-term levers for nest wall-time. Build on the
fp32 reports' structural findings; do not re-derive them.

## POINT 3 — cut host-orchestration: on-device boundary-builds + sync-cadence + kernel fusion
The nested step issues ~5025 advances + ~4422 host force/boundary-builds per fc-h and
~12k tiny dependent kernels/step (launch-bound). Analyze:
- Moving `build_child_boundary_package` / the per-substep force/boundary construction fully
  INTO the device/fused region (cut the ~4422 host builds + the host<->device hops).
- `root_sync_cadence` / block_until_ready policy tuning (queue depth vs VRAM).
- The **fused acoustic + vertical-implicit-solve kernel** that GPT FP32-S2 flagged as "the
  next step — structural, not a dtype tweak" (fuse the tiny dependent kernels that dominate
  the launch count: loop_multiply_fusion / input_reduce_fusion / the PCR tridiagonal solve).
- CUDA-graph / command-buffer capture of the per-substep cascade (note: prior graph on/off
  A/B was ~no-op for the OLD leaf program — explain when it WOULD help now).
Source anchors: src/gpuwrf/coupling/boundary_construction.py, runtime/domain_tree.py
(run_domain_tree_callbacks / the fused cascade), runtime/operational_mode.py (_advance_chunk),
proofs/v018/maxdom9_speedup/profile_opus.md (nsys: launches/syncs).

## POINT 4 — larger timestep + physics-call cadence (algorithmic, stability-gated)
- Larger acoustic/RK or `time_step`/`time_step_sound` where stable; CFL headroom on the
  nested ratios. Fewer steps = faster, but stability/fidelity risk.
- Physics-call cadence (radt, cu/pbl/microphysics call frequency) vs WRF defaults — is
  anything called more often than needed?
- Gate strictly on the SAME acceptance policy as point 1: no blow-up (except cumulative/
  QVAPOR) + wind/temp/cloud 24-120h skill preserved on the nested benchmark.

## DELIVER (your report -> OPUS_P34_PLAN.md or GPT_P34_PLAN.md under proofs/v020/fp32_analysis/)
For EACH of point 3 and point 4: (a) strategic analysis + expected speedup (honest, tied to
the launch/HBM/occupancy model), (b) step-by-step incremental plan, (c) validation/probes/
tools (how to measure launch-count/HBM/occupancy deltas + nested tolerance/skill), (d) road-
blocks + ideas, (e) how nested all-7 max_dom=9 stays fully functional. Rank point-3 sub-levers
by reward/effort. NO GPU; NO edits. Report to pane 0:1 when done.
