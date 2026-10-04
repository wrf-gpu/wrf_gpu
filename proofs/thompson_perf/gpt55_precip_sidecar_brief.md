# GPT-5.5 sidecar: review precip-oracle validation + implicit-sed ADOPT/REJECT

## Context
JAX GPU port of WRF v4 Thompson microphysics. The shipped DEFAULT is a
FAITHFUL-EXPLICIT sedimentation (mirrors WRF's sub-stepped upwind, fixed
NSED_SUBSTEPS=64). The ONLY remaining >=10x speed lever is replacing it with a
single-sweep IMPLICIT backward-Euler sedimentation (~2.4x on the kernel) — a
NUMERICAL SCHEME CHANGE. We just built a PRECIPITATING WRF Thompson oracle (a
standalone single-column harness driving the REAL WRF mp_gt_driver on 8
near-saturated columns with active rain/snow/graupel/ice, dumped via the same
WRFGPU2 oracle instrumentation). dt=18s, one step.

## Results vs the precipitating WRF oracle (one 18s step, 8 columns x 44 lev)

WRF surface precip (rainncv mm/col): cols 1-5 ~5e-11 (≈0), cols 6-8: 0.0182, 0.0210, 0.0240. Total 0.0632 mm.

JAX FAITHFUL-EXPLICIT:
  per-field vs WRF on active cells: qr mean_rel 2.7% (max 35%), qs mean_rel 8%,
  qg mean_rel 36% (max 100% on a few low-mass cells), qv mean_rel 0.5%.
  Vertical qr/qv profiles match WRF to ~1-7% layer by layer (verified).
  surface precip: 0.0135..0.0431 mm/col, total 0.221 mm  (3.5x WRF's 0.063).
  water closure (vapor+condensate change + precip): max_rel 1.5e-6 (excellent).

JAX IMPLICIT backward-Euler (sed swapped in at WRF's exact ordering position):
  nsub=1: total surface precip 0.381 mm (1.7x faithful, 6x WRF)
  nsub=2: 0.319 mm
  nsub=4: 0.278 mm   -> converges DOWN toward faithful 0.221 as nsub rises
  qr mean_rel vs WRF: 5.5% (nsub1) vs faithful 2.7%; qg/qi/qs similar to faithful.
  water closure: max_rel ~1e-6 (mass-conserving, as expected for upwind BE).

## Root cause of the faithful-vs-WRF surface-precip gap (already found)
WRF (module_mp_thompson.F:3636, 3791, 3817):
  - nstep is ADAPTIVE per column = INT(DT/(dz/vt)+1) (small for weak/thin
    layers), JAX uses a FIXED NSED=64.
  - WRF only accumulates surface precip when surface rain density
    rr(kts) > R1*1000 = 1e-9 kg/m3; JAX extracts the bottom-face flux every
    substep with NO threshold.
  So for weakly-precipitating cols (1-5) WRF holds rain in the lowest layer
  (qr@k0 WRF 4.25e-4 > JAX 3.09e-4) and reports ~0 precip; JAX bleeds it out.
  The column qr/qv PROFILES still match WRF to a few %; only the surface-flux
  attribution differs.

## Questions for you
1. Is the faithful-explicit JAX kernel's surface-precip over-attribution (no
   rr>1e-9 threshold, fixed NSED vs adaptive nstep) a CORRECTNESS BUG that
   should block the 0.1.0 "nightly Canary precipitates" functional gate, or an
   acceptable small surface-BC difference given the profiles + water-closure are
   faithful? What honest tolerance band would you set for #32?
2. Implicit-sed is ~1.7x more surface precip than faithful in one step
   (converging down with nsub). For an OPERATIONAL forecast (precip accumulation
   + T2/U10/V10 skill vs CPU-WRF), is one-sweep BE's extra diffusion
   disqualifying, or is nsub=2-4 BE defensible if coupled skill holds? Give a
   crisp ADOPT/REJECT lean with the single most important caveat.
3. Anything in the methodology (single-column oracle, monkeypatched sed at WRF
   ordering, masked-active per-field metric) that would make you distrust the
   ADOPT/REJECT call?

Answer concisely (<400 words). This is a milestone-gating decision.
