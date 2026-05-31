# GPT-5.5 sidecar — case3 V10 momentum-budget attribution review

You are a skeptical reviewer for a JAX GPU port of WRF v4 (Canary d02, 66x159, 93% water).
Branch worker/opus/v10-momentum, base 59915f2. Read-only review; do NOT edit code.
Working dir: /home/enric/src/wrf_gpu2/.claude/worktrees/agent-abc754c924875d0fb

## The defect (proven by prior sprints)
case3 (init 2026-05-21 18z, L3) 24h forecast: V10 below persistence over water.
- V10 skill (=1 - GPU_RMSE/pers_RMSE): ALL -0.099, water -0.132. (skill>0 beats persistence)
- u* (friction velocity) MATCHES WRF (0.255 vs 0.261) -> surface DRAG faithful.
- lowest-level prognostic wind over water: GPU v0=-4.85 vs WRF k0 v=-7.11 (too WEAK);
  GPU u0=+1.12 vs WRF -0.69 (WRONG SIGN). The k0 wind VECTOR is wrong.

## NEW finding I just produced (the key pivot from the case2 root-cause)
Boundary-frame skill decomposition of the EXISTING case3 24h fields:
- V10 water skill -0.132; EXCLUDING the 5-cell boundary frame it gets WORSE: -0.204.
- The boundary frame itself scores WELL: V10 frame skill +0.39, U10 frame +0.24.
- Deep-interior box (rows 20-46, cols 30-120) water V10 skill -0.141.
=> Unlike case2 (where the boundary normal-momentum plume was the cause, since fixed
   with apply_normal_bdy_work strength=20), case3's V10 deficiency is in the DEEP
   INTERIOR over water, NOT the boundary. The boundary protection is working here.

So the case3 residual is one of: (1) dycore interior momentum transport/PGF, or
(2) MYNN-PBL momentum vertical mixing (over-draining the lowest level).

## My planned attribution (please critique for soundness / what would falsify each conclusion)
I will run an instrumented 24h case3 GPU forecast and dump, over water:
1. The lowest-level u/v momentum tendency split into: dycore (RK+acoustic) delta vs
   MYNN-PBL increment (du_mass/dv_mass from _state_from_mynn_output) per step.
2. The full vertical u/v profile at k0..k5 over water at 24h vs WRF (from wrfout U/V on
   mass levels) — to see if the column is uniformly weak (dycore) or only the lowest
   level is weak relative to aloft (MYNN over-mixing momentum down/out).
3. A MYNN-momentum-OFF counterfactual (zero the PBL u/v increment only; keep theta/qv/
   surface fluxes) re-run 24h, re-score V10 water skill. If V10 improves toward WRF and
   the column k0 wind strengthens -> MYNN over-mixing is the (partial) cause (a SAFE fix
   target in mynn_pbl.py). If V10 unchanged/worse -> the deficiency is the dycore interior
   flow (sacrosanct; defer).

## Questions
1. Is the MYNN-momentum-OFF counterfactual a valid isolation? MYNN momentum mixing
   transfers momentum vertically; turning it off removes BOTH the surface friction sink
   AND the downward mixing of faster aloft wind. Over water with weak mixing, which
   dominates the k0 wind? How do I avoid mis-attributing?
2. The vector is wrong in DIRECTION (u0 sign flip), not just magnitude. Vertical mixing
   is direction-preserving-ish (mixes toward the column mean). A direction error smells
   like horizontal advection / PGF / Coriolis (dycore), not vertical PBL mixing. Does
   that argue the cause is dycore, making the whole exercise a DEFER? What evidence would
   distinguish a PBL-fixable magnitude error from a dycore direction error?
3. What is the cleanest WRF-faithful, low-regression MYNN momentum lever IF over-mixing
   is confirmed (e.g. the master length scale el, the dfm diffusivity, Pr number)? Or is
   any such change too risky vs the -0.099 residual?

Write your verdict to proofs/wind/v10_momentum_sidecar_verdict.md. Be concise and decisive.
Key files (read-only): src/gpuwrf/physics/mynn_pbl.py (momentum in _apply_mean_tendencies,
_mym_turbulence dfm), src/gpuwrf/coupling/physics_couplers.py (_state_from_mynn_output,
add_a2c increments), src/gpuwrf/dynamics/acoustic_wrf.py + dynamics/core/* (dycore, READ ONLY),
proofs/wind/case3_wind_residual_findings.md, proofs/wind/WIND_SKILL_ROOT_CAUSE.md.
