# Recomposition stability isolation ladder

Branch: `worker/opus/recomp`. Real case: Gen2 d02
`/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z`
(mass grid 44 x 66 x 159, dt=1 s, n_acoustic=4, fp64). All runs guards-OFF
unless stated; a field is "unphysical" when it leaves the production sanitiser
envelope (theta in [150,550] K, |w|<=50, |u|<=150 m/s), so "first unphysical
step" == "first step the production guard would engage".

## Headline

The coupled real-case instability that B4 observed (sanitiser pinning
theta/w/u within ~30 steps) is **rooted in the DRY DYNAMICAL CORE, not in the
physics coupling**. The merged physics lanes are not the trigger.

## Ladder results (proofs/recomp/*.json)

| Config | dt | damping | result | first unphysical | first NaN |
|---|---|---|---|---|---|
| dycore only | 2.0 | none | UNSTABLE | step 1 (w=73) | step 5 |
| dycore only | 1.0 | none | UNSTABLE | step 2 | step 9 |
| dycore only | 0.25 | none | survives 30 steps but w->103 (>50) | step 5 | none@30 |
| dycore only | 1.0 | smdiv 0.1 | UNSTABLE | step 2 | step 10 |
| dycore only | 1.0 | rayleigh 0.2 | UNSTABLE | step 2 | step 13 |
| dycore only | 1.0 | smdiv0.1+rayleigh1.0 | UNSTABLE | step 5 | step 18 |
| dycore only | 1.0 | smdiv0.1+rayleigh5.0 top0.66 | stable 40, **UNSTABLE by 200** | step 43 | step 50 |
| +boundary | 1.0 | smdiv0.1+rayleigh5.0 | u/v -> 344 | step 17 | none@40 |
| +surface | 1.0 | smdiv0.1+rayleigh5.0 | same as +boundary | step 17 | none@40 |
| +mynn | 1.0 | smdiv0.1+rayleigh5.0 | u/v -> 227 (mynn tempers) | step 30 | none@40 |
| +radiation(every step) | 1.0 | smdiv0.1+rayleigh5.0 | theta NaN | step 7 | step 9 |

Production sanitized path (`run_replay_scan`, guard ON, 30 steps):
**GUARD IS LOAD-BEARING** — 20k-130k cells clipped EVERY step; |w| pinned at
exactly 50.00 from step 2; theta hits the 150 K floor by step 30. Total clips
3,069,606 over 30 steps. The forecast "stays finite" only because the clamp
holds w at its ceiling continuously. (proofs/recomp/production_sanitized_probe.json)

## Isolated triggers (in order of dominance)

1. **DRY DYCORE — upper-level w mode + horizontal u mode (PRIMARY).**
   `proofs/recomp/w_blowup_locate.json`: w noise originates at the top of the
   column (z=40/44, ~88% height) at step 1 and cascades downward
   (z 40->38->36->35->33->32) over 6 steps, concentrated at fixed (y,x) points
   (y=30,x=62; y=49,x=22). Strong upper-level Rayleigh w-damping (coef 5.0,
   ramp from 66% height) suppresses the w mode for ~40 steps, but then a slower
   **horizontal-u growth** takes over (u: 133->138->144->157->188->219->267->
   406 m/s, explodes step 49). So damping only DELAYS; the dycore is
   structurally unstable on this real init. This matches the documented open
   dycore residual (`project_dycore_rewrite_status_2026_05_29`: "large-step
   pg_buoy_w buoyancy not saturating, prime suspect rhs_ph/ph_tend STUB").
   This is `src/gpuwrf/dynamics/**` — out of physics-recomposition scope.

2. **Lateral boundaries accelerate the u mode (B4).** With the dycore w-mode
   damped, adding `apply_lateral_boundaries` pulls the u/v blow-up forward from
   step 43 to step 17 (u,v both -> 150+ by step 17, growing together). The
   relaxation/spec forcing on the decoupled real side-history appears to pump
   horizontal momentum. Needs B4 review once the dycore is stable.

3. **Radiation theta sign/magnitude vs cadence (B3).** Applied EVERY 1-second
   step in the ladder, RRTMG heating drives theta negative by step 7 (-19320 K)
   then NaN. The B3 kernel is validated in isolation (SWDOWN peak 1066 W/m2,
   proofs/b3/diurnal_sanity.json), so this is a CADENCE/application artifact:
   `rrtmg_adapter` does `T += dt * heating_rate` with the single-step dt, but
   production only calls it every `radiation_cadence_steps` (60). Heating must
   be applied at the right cadence/scaling. Re-test at production cadence once
   the dycore is stable; flag the adapter's dt-vs-cadence contract for B3.

## Conclusion / handoff

The physics recomposition (B2 surface+MYNN, B3 RRTMG, B4 boundaries) merged
cleanly and the lanes pass their tests. The coupled real-case is NOT stable,
but the dominant trigger is the dry dycore's structural w+u instability on the
real initial state, which is dynamics-core territory (closed/frozen, owned by
the dycore rewrite track). Per the operating model this needs the dycore-fix
mind (GPT-5.5 / dycore track), not a physics-lane masking change. Triggers 2
and 3 are real but secondary and only become visible once the dycore is fixed.

Reproduce:
```
PYTHONPATH=src python proofs/recomp/stability_ladder.py --steps 40 --dt 1.0 \
  --rungs dycore,boundary,surface,mynn,radiation
PYTHONPATH=src python proofs/recomp/stability_ladder.py --steps 40 --dt 1.0 \
  --smdiv 0.1 --rayleigh 5.0 --rayleigh-top-frac 0.66 \
  --rungs dycore,boundary,surface,mynn,radiation
PYTHONPATH=src python proofs/recomp/production_sanitized_probe.py --steps 30
PYTHONPATH=src python proofs/recomp/w_blowup_locate.py --steps 6
PYTHONPATH=src python proofs/recomp/damping_test.py --steps 40
```
