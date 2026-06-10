# V0.14 Canary h24 Residual Adjudication — Proof Summary

Verdict: `FABLE_FIX_REQUIRED_AFTER_RUN_FROZEN_LAND_TSK_NESTED_PIPELINE_NO_LSM`

CPU-only analysis of the live Canary d02 72h moist-cqw run at h24 (run alive at
h29 during analysis; GPU untouched). Full numbers in
`canary_h24_residual_adjudication.json`; narrative + sprint definition in
`.agent/reviews/2026-06-10-v014-fable-canary-h24-residual.md`.

## The one new root cause: GPU land skin temperature is frozen

GPU land-mean `TSK` is **constant to 0.01 K for all 29 hours** on both domains
(d02: 294.88 K, d01: 300.95 K) while CPU truth swings 283 -> 310 K diurnally.
The atmosphere feels it — land HFX bias +183 W/m2 at night, -308 W/m2 at noon —
so this is physics, not writer-only.

Code owner: `src/gpuwrf/integration/nested_pipeline.py::_make_namelist` builds
`OperationalNamelist.from_grid(...)` without `use_noahmp` / `sf_surface_physics`
/ `noahmp_static`; `use_noahmp` defaults `False`, so the land tile stays on the
prescribed bulk path and `state.t_skin` never advances. CPU truth runs Noah-MP.
The validated Noah-MP coupler exists and is wired in `daily_pipeline`, but was
never threaded into the standalone nested pipeline that all canary/TOST runs use
since the v0.13 KI-5 rerouting (including the v0.12.0 final 24h nested gate).

## What the formal h24 `FAIL` actually consists of

- `MUB/PB` static: max 250 Pa **100% inside the 5-cell nest frame**; interior
  max 0.0078 Pa (fp32 wrfout quantum). Known static lane.
- `QVAPOR`: marginally over the 1e-3 RMSE limit from h12 (1.29e-3 at h24);
  growth decelerating, bias shrinking after h24. Watch item.
- `T2`: fails only leads 10-12 (1.54-1.59 vs 1.5), night stable-PBL hours =
  direct frozen-TSK signature; recovers daytime.
- **No dynamics field fails its manifest at any lead.** PSFC worst lead h18
  RMSE 105.3 < 120 limit.

## Dynamics health (the moist-cqw/PSFC/LBC fixes hold)

U/V/W/T RMSE saturate (U ~0.8 m/s flat from h9; W ~0.044; T peaks 0.71 K then
declines). PSFC per-lead RMSE is diurnal and recovering
(52 -> 11 -> 105@h18 -> 35@h24 -> 40@h29), not monotonic: no renewed
LBC-cadence drift (old failure was 51-73 Pa/h monotonic). The comparator
"slope/h" aliases the diurnal cycle and should not be read as drift here.

The domain-wide noon mass dip originates on parent d01 (ocean -98 Pa, land
-65 Pa at noon) and recovers by evening; contributors are frozen land TSK on
both domains plus the bounded RRTMG clear-sky and ocean moisture-flux lanes.
Re-adjudicate after the land fix; no dycore sprint on this signal.

## Decision

`FABLE_FIX_REQUIRED_AFTER_RUN` — let the run finish (stability/dynamics
evidence remains valid), do not promote it as the release green gate, do not
launch Switzerland GPU (land-dominated alpine domain) before the
nested-pipeline land-surface activation fix, then rerun the Canary 72h gate.
