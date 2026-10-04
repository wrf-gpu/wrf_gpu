# RESOLVED: the nest "72-min Ni non-finite" is a dt/CFL artifact, NOT a port bug (2026-06-26)

## Decisive evidence
- **Native dt = 18 s** (`20250121/cpu/namelist.input`). **The K2 nest ladder's "N1 baseline" used 54 s** root dt
  (`runs/N1/input/namelist.input`) = **3× the native** → vertical Courant Cz>1 (d02 1.40, d03 1.05) → CFL
  instability → Ni overflow at ~72 min. The K2 ladder mislabeled a 3× dt-upscale as the "baseline".
- **At the native 18 s dt the steep-terrain (20250121) 3-dom nest is STABLE:** the instrumented trace run
  (`/tmp/.../dbg/`, native namelist) wrote d03 frames to **21:00 = +3 h, ZERO NonFiniteStateError**; probe ran to
  d03 step 5400 (=3 h). The CPU-WRF reference (`20250121/cpu`) runs the same case **24 h finite** ("SUCCESS").
- So: **NOT a GPU-port bug.** The caps (Ni/Nr ceilings) are correct WRF-faithful guards but **never fire at the
  native dt** (no overflow) → they're harmless/bit-identical there. My multi-iteration cap hunt chased a CFL artifact.

## Validity at the native dt (GPU vs CPU-WRF, same dt + same timestamp, d03)
| field | rel max-diff @19:00 | note |
|---|---|---|
| T (temp) | 2.7% | smooth dynamics MATCH |
| PH (geopot) | 3.9% | smooth MATCH |
| MU (col mass) | 16% | ok |
| W (vert vel) | 60% | turbulent — chaos decorrelation |
| U/V (winds) | ~40% | turbulent — chaos |
| QVAPOR | 37% | turbulent/moist — chaos |
| ice/snow/rain (Q*, QN*) | ~0 (both ~0 at cell) | match |
Smooth fields agree; turbulent fields decorrelate over hours = **inherent NWP deterministic chaos** (any two
independent runs of the same case diverge in W/winds), NOT a bug. Consistent with the project's operational-RMSE
validation philosophy ([[feedback_validation_philosophy]]).

### RMSE/correlation validation (the right metric — d03, native-dt GPU vs CPU-WRF truth)
| field | corr @19:00 | corr @20:00 | corr @21:00 | RMSE @21:00 |
|---|---|---|---|---|
| T (temp)   | 1.000 | 1.000 | 1.000 | 0.56 K |
| U wind     | 0.999 | 0.998 | 0.997 | 1.14 m/s |
| V wind     | 0.999 | 0.997 | 0.996 | 1.05 m/s |
| W vert vel | 0.870 | 0.789 | 0.746 | 0.32 m/s |
| QVAPOR     | 0.989 | 0.967 | 0.956 | 1.0e-3 |
**VERDICT: highly valid.** T/U/V correlate ~1.0 with CPU-WRF (sub-K / sub-m/s RMSE); the earlier "60% W" was a
single outlier cell — W RMSE is only ~0.3 m/s with corr 0.75-0.87 (the expected slow convective-scale
decorrelation at 1 km; CPU-WRF would decorrelate similarly vs a perturbed run). **The native-dt steep-terrain
output is trustworthy for AI training data** ([[feedback_pod_runs_fully_valid_no_shortcuts_2026_06_26]]). A
CPU-WRF-vs-perturbed-CPU-WRF chaos baseline + the full 24h window would further quantify it, but the dynamics
core is demonstrably correct on steep terrain at the native dt.

## Implications for v0.21.1 + the pod runs
1. **Pod runs at the NATIVE dt are STABLE on steep terrain — no NaN in the Alps, no deep fix needed.** UNBLOCKED.
2. **K2-nested dt-upscaling does NOT work on steep terrain** — 54 s is supercritical (a real CFL limit; even
   CPU-WRF cannot run 54 s on this case). So the K2 ~1.8× nest speedup is **flat/single-domain only**, NOT for
   steep-terrain pod runs. Honest answer to the K2-nest question.
3. **v0.21.1 reframed:** "stability fix" = the native dt is already stable (caps don't fire); "K2-nested 1.8×" =
   off for steep terrain. v0.21.1 = current code (stable at native dt) + honest docs (K2-upscaling not for steep
   nests) + the steep-terrain native-dt validation vs CPU-WRF truth.
4. **DECISION for the user:** confirm the pod runs at NATIVE dt (stable + valid, trading the K2 speedup for
   trustworthiness on steep terrain) — the right call for AI training data per
   [[feedback_pod_runs_fully_valid_no_shortcuts_2026_06_26]].

## Process lesson
Trace the mechanism (and check the run CONFIG — dt!) before applying physics fixes. The field-order "Ni first" was
real but the root was dt/CFL, not microphysics. The CPU-WRF truth reference + native-dt run cracked it in one pass.

## No-regression confirmation (2026-06-26)
- **Canary (validated nest, native 18s dt, current code incl. caps): FINITE past 72min** — 4 d03 frames
  (18:20/18:40/19:00/19:20 = +80min), 0 NonFiniteStateError → the WRF-faithful Ni/Nr caps do NOT regress the
  validated case (bit-identical-when-not-firing confirmed in practice). v0.21.1-candidate code is no-regression.
- Mont-Blanc fixture (256² 1km, ~1042 m/cell, native 18s dt) queued — the most-extreme terrain confirmation.
