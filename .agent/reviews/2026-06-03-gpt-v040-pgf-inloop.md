# GPT v0.4.0 PGF In-Loop Isolation Handoff - 2026-06-03

## Objective

Run a paired in-loop momentum-term isolation for the real d01 case
`20260429_18z_l2_72h_20260524T204451Z` from branch
`worker/gpt/v040-lbc-budget-trace`, using the same low-level state for JAX and
WRF-source terms. The purpose was to split the prior hourly PGF/momentum-balance
lead into PGF vs advection vs Coriolis vs metric/map coupling before any fix.

## Decision

The divergent in-loop operator is **not large-step horizontal PGF**.

At the identical RK1 / first acoustic-substep input state, PGF matches the
WRF-source oracle at roundoff:

- `large_step_pgf_u_total`: RMSE `4.12e-15`, max abs `4.55e-13`
- `large_step_pgf_v_total`: RMSE `4.18e-15`, max abs `4.55e-13`
- terrain-following geopotential correction is present and non-trivial:
  `u_terrain_no_moisture` RMSE `26.72`, max abs `4351.81`;
  `v_terrain_no_moisture` RMSE `28.78`, max abs `5667.42`
- cqu/cqv full-moisture coupling is active: mean `~0.997738`
- map-factor PGF ratios are carried: `msfux/msfuy = 1`, `msfvy/msfvx = 1` on this d01
- c1h/c2h hybrid-coordinate coefficients are loaded and used

Coriolis also matches the WRF-source oracle exactly at this savepoint.

The isolated same-units momentum tendency residual is **momentum advection**,
largest in `momentum_advection_v_vertical_flux`: low-level RMSE `34.63`, full
RMSE `93.24`, full max abs `3918.06` at `(k=28, j=8, i=92)`. The adjacent
V-advection terms are similarly large: `v_y_flux` low-level RMSE `34.12`,
`v_total` low-level RMSE `33.40`, `v_x_flux` low-level RMSE `23.68`.

The upstream metric/BC diagnostic is `transport_coupling_rv_calc_mu_uv`: JAX
currently uses the periodic `couple_velocities_periodic` path on real d01, while
WRF specified d01 uses edge `calc_mu_uv`, `couple_momentum`, `calc_ww_cp`, and
degraded/upstream normal-boundary fluxes in `advect_u`/`advect_v`.

## WRF Source Anchors

- PGF u: `dyn_em/module_big_step_utilities_em.F:2453-2488`
- PGF v: `dyn_em/module_big_step_utilities_em.F:2373-2404`
- moisture coupling `cqu/cqv`: `dyn_em/module_big_step_utilities_em.F:787-850`
- `calc_mu_uv`: `dyn_em/module_big_step_utilities_em.F:26-180`
- `couple_momentum`: `dyn_em/module_big_step_utilities_em.F:329-399`
- `calc_ww_cp`: `dyn_em/module_big_step_utilities_em.F:640-782`
- `advect_u`: `dyn_em/module_advect_em.F:493-1240`, vertical `:1336-1526`
- `advect_v`: `dyn_em/module_advect_em.F:1911-2818`, vertical `:2820-3024`
- Coriolis driver/body: `dyn_em/module_em.F:1402-1428`,
  `dyn_em/module_big_step_utilities_em.F:3924-4134`
- default momentum advection orders: `Registry/Registry.EM_COMMON:2874-2875`
  (`h_mom_adv_order=5`, `v_mom_adv_order=3`)

## Files Changed

- `proofs/v040/pgf_inloop_isolation.py`
- `proofs/v040/pgf_inloop_isolation.json`
- `.agent/reviews/2026-06-03-gpt-v040-pgf-inloop.md`

No production model code changed.

## Commands Run

- `pwd && git status --short && git log -1 --oneline --decorate`
- Source inspections with `rg`, `nl`, `sed` against JAX and WRF source files.
- Metadata inspections of target `wrfinput_d01` / `wrfout_d01_2026-04-29_18:00:00`.
- `python -m py_compile proofs/v040/pgf_inloop_isolation.py`
- `python proofs/v040/pgf_inloop_isolation.py`

No GPU job was launched; this was CPU-only formula/savepoint isolation, so no
`nvidia-smi` reservation was needed.

## Proof Objects Produced

- `proofs/v040/pgf_inloop_isolation.json`

No `forecast_gate_postfix3_report.json` was produced because no production fix
was applied.

## Fix Status

Fix is **proposed**, not applied.

Required fix: add a grid/BC-conditional real-boundary flux-advection path
implementing WRF `calc_mu_uv`, `calc_ww_cp`, and `advect_u`/`advect_v`
specified-boundary h=5/v=3 degradation. Preserve the current periodic helper for
idealized periodic cases so Straka/warm-bubble periodic gates remain unchanged.

This is not a surgical PGF edit and should not be patched as a one-off mask or
tolerance change.

## Checks

- Proof script compiles.
- JAX decomposed advection total equals current public `advect_u_flux` /
  `advect_v_flux` exactly in the proof sanity rows.
- PGF component checks confirm terrain term, moisture coupling, map ratios, and
  hybrid coefficients are present in the JAX formula.

Idealized/replay/2-date forecast gates were not rerun because production code was
not changed.

## Unresolved Risks

- WRF side is a source-equivalent NumPy oracle anchored to the verified WRF
  source lines, not a recompiled/instrumented WRF executable dump.
- The production fix is larger than this isolation sprint because it must add the
  real specified-boundary flux-advection path without perturbing periodic
  idealized gates.
- v0.4.0 cannot close yet; the 2-date PSFC/U10 bias is expected to remain until
  the real-boundary advection path is implemented and forecast-gated.

## Next Decision Needed

Approve an implementation sprint for the real-boundary momentum advection path,
with acceptance gates:

1. WRF-source savepoint parity for `calc_mu_uv`/`calc_ww_cp` and
   `advect_u`/`advect_v` on the 20260429 real d01 state.
2. Idealized periodic no-regression proving the current periodic helper remains
   selected and bit-stable.
3. 2-date `20260429` + `20260521` forecast gate to confirm PSFC/U10 collapse.
