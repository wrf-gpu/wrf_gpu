# Memory Patch: V0.14 GPT RRTMG/RTHRATEN Closure

Reviewer Status: APPLY TO ROADMAP/MEMORY.

Durable memory entry:

- RRTMG/RTHRATEN dry-temperature input bug closed on 2026-06-10.
- Exact WRF boundary: `RRTMG_LWRAD:T3D=t`.
- Production owner: `src/gpuwrf/coupling/physics_couplers.py::_rrtmg_column_inputs`.
- Fix: metric-backed RRTMG input decouples `theta_m` to dry theta before temperature conversion; grid-less fallback remains historical.
- Proofs:
  - `proofs/v014/rrtmg_rthraten_closure.{py,json,md}`
  - `proofs/v014/rrtmg_step1_forcing_parity.{py,json,md}`
  - `proofs/v014/mynn_rthblten_step1_closure.{py,json,md}`
  - `proofs/v014/noahmp_step1_closure.{py,json,md}`

Numbers to remember:

- `T3D=t` max_abs `5.5213 K -> 0.08944 K`.
- GLW RMSE `17.5203 -> 0.3515 W/m2`.
- Mass-coupled RTHRATEN RMSE `2.4884 -> 0.3646`; max_abs `19.4253 -> 2.7984`.
- Strict Step-1 remains red/bounded at max_abs `55.9297`, RMSE `0.4997`, p99 `0.9529`.

Next manager memory:

- Do not call strict MYNN+RRTMG bitwise tolerance green. It is formally bounded and gate-unreachable under the old `1e-3/1e-5` mass-coupled threshold.
- TOST start signal requires short operational field/rollout falsifier plus a recorded tolerance-policy decision.
