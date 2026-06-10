# Worker Report: V0.14 GPT RRTMG/RTHRATEN Closure

Summary: GPT-5.5 xhigh found and fixed the dominant RRTMG dry-temperature input bug. Metric-backed `_rrtmg_column_inputs` now mirrors WRF `phy_prep`: it decouples stored moist theta `theta_m` to dry theta before converting to RRTMG `T3D=t`. Grid-less analytic callers keep the prior fallback path.

Files changed by worker:

- `src/gpuwrf/coupling/physics_couplers.py`
- `tests/test_v014_dry_source_leaf_wiring.py`
- `tests/test_m5_rrtmg_gate.py`
- `proofs/v014/rrtmg_rthraten_closure.{py,json,md}`
- `proofs/v014/rrtmg_step1_forcing_parity.{py,json,md}`
- `proofs/v014/noahmp_step1_closure.{py,json,md}`
- `proofs/v014/mynn_rthblten_step1_closure.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-gpt-rrtmg-rthraten-closure.md`

Primary result:

- `T3D=t` max_abs improves `5.521345498302992 K -> 0.08944393302414255 K`.
- GLW RMSE improves `17.520282676793663 -> 0.35152062180132065 W/m2`.
- Mass-coupled RTHRATEN RMSE improves `2.4884141898276413 -> 0.3645729657536835`; max_abs improves `19.425283200182427 -> 2.798351397503893`.

Unresolved:

- Strict Step-1 remains red and formally bounded, not release-green: post-fix `T_TENDF` max_abs `55.92970981221765`, RMSE `0.499664626865853`, p99 `0.9529013024125962`.
- Remaining strict max is MYNN level-2.5/QKE floor; RRTMG remains field-significant but no longer the pre-fix gross blocker.
