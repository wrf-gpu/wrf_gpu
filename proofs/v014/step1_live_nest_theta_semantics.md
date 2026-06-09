# V0.14 Step-1 Live-Nest Theta Semantics

Verdict: `STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- Required ancestor `7ae33eda` present: `True`.
- Real run `USE_THETA_M`: `{'wrfinput_attr': 1, 'wrfout_attr': 1, 'namelist_output': 'USE_THETA_M=1          ,'}`.
- Raw/current live dry `T_STATE` vs WRF pre-call: max_abs `5.490173101425171` / `5.490173101425171`.
- `adjust_tempqv` directly on raw dry `T` with `use_theta_m=1`: max_abs `5.490177290476879`.
- WRF dry-to-moist theta conversion only: max_abs `0.753296811070129`, rmse `0.015916793982291767`.
- WRF `theta_m` conversion plus `adjust_tempqv`: max_abs `0.00541785382188209`, rmse `5.068868142015466e-05`, p99 `4.546931764011239e-05`.
- Same candidate with fp32 arithmetic: max_abs `0.00543212890625`, rmse `5.171032344972347e-05`.
- Wrong order, dry adjust then moist conversion: max_abs `0.27921682023503536`.
- Report-only `QVAPOR` candidate vs `wrfout_d02` H0: max_abs `3.838436518426372e-06`, rmse `2.8529167414336877e-08`.
- Continuity vs WRF pre-call: `P_STATE` max_abs `69.96875`, `PB` `0.05357326504599769`, `MUB` `0.05002361937658861`, `PHB` `0.10811684231157415`.

## Interpretation

- `USE_THETA_M=1` means operational in-memory WRF `grid%t_2` is moist perturbation theta. For this run, `State.theta` should represent WRF `grid%t_2 + 300 K` if it is intended to mirror solve-time WRF state.
- The live-nest theta residual is not closed by `adjust_tempqv` alone on raw dry NetCDF `T`. The needed semantic sequence is dry `T` to moist theta, then WRF `adjust_tempqv`.
- That sequence reduces max_abs from `5.490173101425171` to `0.00541785382188209`, but it remains above the prior `1e-3 K` material threshold, so no production patch was made.
- The named accepted WRF pre-call truth does not include `QVAPOR`; the QVAPOR comparison in this proof is against `wrfout_d02` H0 and is report-only, not an accepted pre-call proof.

## WRF Source Evidence

- `share/mediation_integrate.F:726-762`: saves `mub`, blends `ht/mub/phb`, then calls `adjust_tempqv` with `nest%t_2`, `nest%p`, and `QVAPOR`.
- `dyn_em/nest_init_utils.F:812-890`: `adjust_tempqv` computes old/new pressure, preserves RH, updates `th`, then updates `qv`.
- `dyn_em/module_initialize_real.F:4918-4928`: when `use_theta_m=1`, WRF converts dry `grid%t_2` to moist theta in memory.

Detailed tables are in `proofs/v014/step1_live_nest_theta_semantics.json`.
