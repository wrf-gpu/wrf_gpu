# GPT-5.5 sidecar: critique the case3 wind-residual decomposition + proposed surface-diagnostic lever

You are a skeptical reviewer. Read-only. Argue the OPPOSING position where you can.
Be concise and decision-oriented.

## Context (a JAX GPU port of WRF v4, Canary d02, reusing CPU-WRF corpus as truth)

A merged dycore/boundary fix made near-surface winds beat persistence on case2
(all leads). RESIDUAL: case3 (init 2026-05-21 18z, L3 1km d02) 24h V10 loses to
persistence (skill -0.099), U10 ties (-0.001). My mission: prove case3 is a real
deficiency vs a regime/metric limit, and if real apply the MINIMAL surface-path
fix (file ownership: surface_layer.py / mynn_pbl.py only; dycore/boundary/
thompson are OFF-LIMITS).

## STEP 1 evidence (regime vs deficiency) — DECISIVE

case3 has MULTIPLE independent CPU-WRF forecasts of the same init on the same
d02 grid (L2 9->3km parent, L3 3->1km parent). Their mutual V10/U10 RMSE
(=irreducible CPU-WRF uncertainty) at the same valid times:
  - case3 V10: CPU-WRF L2-vs-L3 self-spread = 0.002-0.020 m/s while persistence
    error = 2.2-2.8 m/s  -> ratio ~0.001-0.009
  - case3 U10: same, ratio ~0.001-0.007
  - case2 V10 (proven-skillful reference): self-spread/persistence ratio ~0.005-0.06
case3 init V10 mean = -5.4 m/s, std 2.1 (a STRONG southerly regime, NOT calm);
field evolves (change-from-init RMSE 1.6->3.1 m/s over 24h).
=> CONCLUSION: case3 V10 is strongly predictable; CPU-WRF tracks it to ~0.02 m/s
   regardless of parent grid. The GPU losing to persistence is a REAL deficiency,
   not a regime/metric limit.

QUESTION 1: Is this regime-vs-deficiency logic sound? Any reason the L2-vs-L3
self-spread being tiny would NOT imply predictability (e.g. shared lateral BC
making d02 near-deterministic — does that change the "deficiency" conclusion)?

## STEP 2 evidence (WHERE the deficiency is) — GPU localization @ 24h, over water (93% of domain)

| quantity                    | CPU-WRF (truth) | GPU   | gap        |
|-----------------------------|-----------------|-------|------------|
| lowest model-level wspd (k=0)| 7.90            | 6.11  | GPU -1.8   |
| lowest-level v0              | -7.11           | -4.85 | GPU +2.3   |
| 10 m wspd diagnostic         | 7.37            | 5.35  | GPU -2.0   |
| diagnostic ratio (10m / k0)  | 0.933           | 0.871 | GPU -0.06  |
neutral-log ratio gz10oz0/gz1oz0 = 0.896. za~25.7m, znt=0.00285, stable marine
(regime~1.66, zol~+0.25, br~+0.025). WRF MYNN for za>13m uses U10=U1D*PSIX10/PSIX
(stability-corrected, module_sf_mynn.F:1127-1131). Our sfclayrev uses the same
PSIX10/PSIX form.

So TWO gaps:
  (A) PRIMARY: GPU lowest-level prognostic wind is itself ~1.8-2.3 m/s too WEAK
      over water (6.11 vs 7.90). This is upstream of the surface layer — in the
      dycore/PBL momentum. OFF my file ownership.
  (B) SECONDARY: GPU diagnostic ratio 0.871 vs WRF 0.933 (WRF is ABOVE neutral
      0.896; ours BELOW). A stable stability-correction can only push BELOW
      neutral, so WRF's 0.933 implies WRF's effective z/L is much closer to
      neutral than ours -> our sfclayrev computes a MORE stable z/L and
      over-suppresses the 10m wind by ~0.06 ratio (~0.5 m/s). IN my ownership.

QUESTION 2: Do you agree gap (A) dominates and is NOT in surface_layer.py?
Is the diagnostic-ratio fix (B) worth doing given it recovers only ~0.5 m/s of a
~2 m/s deficit and case3 V10 RMSE would stay >persistence (deficit dominated by
A)? Or is the honest answer "case3 is a real deficiency in the prognostic wind
(dycore/PBL), NOT closeable from the surface layer; the surface diagnostic
contributes a minor secondary suppression"?

QUESTION 3 (risk): case2 and case3 SHARE the surface code. Post-fix case2 water
V10 bias is only -0.18 (well-balanced). If I lift the stable-marine diagnostic
ratio toward WRF (raise the 10m wind), it HELPS case3 (too weak) but could HURT
case2 (already balanced / slightly too fast on U10 +0.62). Is a diagnostic change
that trades case2 for case3 acceptable, or should I leave the diagnostic alone
and document case3 as a prognostic-wind (dycore/PBL) deficiency outside surface
scope? Argue both sides; give a recommendation.

Output: ANSWERS to Q1-Q3 + a one-line VERDICT: is there a sound, low-regression
surface_layer.py lever, or is the honest close "case3 residual = prognostic-wind
deficiency outside surface ownership"?
