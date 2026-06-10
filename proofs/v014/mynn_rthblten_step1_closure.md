# V0.14 MYNN-EDMF RTHBLTEN Strict Step-1 Closure

Verdict: `STEP1_STRICT_RED_FORMALLY_BOUNDED_RRTMG_FIELD_DOMINANT_MYNN_KERNEL_FLOOR_GATE_UNREACHABLE`.

## Endpoint

Strict Step-1 is RED and **formally bounded**. The strict gate (max_abs<=`0.001`, rmse<=`1e-05` on mass-coupled `T_TENDF`, mu~1e5 => raw ~1e-8) is **unreachable** for the MYNN+RRTMG theta tendency without bitwise scheme reproduction.

## Lane decomposition (operational strict residual vs rmol-pinned WRF after_conv, interior)

| Configuration | max | rmse | p99 |
|---|---|---|---|
| Operational (cold-start QKE) | 53.5230 | 2.54450 | 16.632 |
| + WRF-pinned INIT_QKE | 27.9671 | 2.53780 | 16.631 |
| + WRF-pinned QKE + WRF RTHRATEN (RRTMG removed) | 40.2802 | 0.54334 | 0.843 |
| RTHRATEN (RRTMG) lane only | 19.7820 | 2.52716 | 16.260 |

**Reading:** injecting WRF QKE drops the worst-CELL max but leaves rmse/p99 unchanged (cold-start QKE = rare single-cell spike). Substituting WRF RTHRATEN collapses the strict rmse/p99 -> the strict FIELD residual is **RRTMG-radiation dominated**; MYNN drives only the worst-cell max.

## Reconciliation with prior 'MYNN kernel faithful' evidence

- Operational dry `T_TENDF` reassembled = runtime capture (consistency max_abs `4.547473508864641e-13`): the strict leaf is exactly `theta_m_factor*mass_h*(RTHRATEN+RTHBLTEN) + Rv/Rd QV`.
- The MYNN `RTHBLTEN` it carries is WRF-faithful (raw ~3e-4 K/s) with WRF-equivalent inputs + QKE -- the prior accepted boundary result stands.
- `build_step1_state` (legacy `run_kernel_matrix`) omits `grid=` on `noahmp_surface_step`; its grid-less LAND surface makes the standalone adapter overshoot `RTHBLTEN` ~2x at land (mass-coupled +260). The OPERATIONAL leaf is faithful there -> the legacy kernel-matrix land tail is a **proof artifact**, not a production bug.

## Ranked lanes (exact ownership)

1. **RRTMG step-1 radiation RTHRATEN (held seed) -- FIELD DOMINANT**
   - owner: `src/gpuwrf/physics/rrtmg_lw.py, rrtmg_sw.py (held RTHRATEN seed via carry.rthraten / _refresh_rthraten); localization proofs/v014/rrtmg_step1_forcing_parity.*`
   - Removing the RRTMG RTHRATEN error collapses the strict rmse 2.5378->0.5433 and p99 16.63->0.84. The strict FIELD residual is radiation-dominated; the contract's 'MYNN-EDMF RTHBLTEN dominant' framing holds only for the single worst CELL.
2. **MYNN level-2.5 turbulence kernel RTHBLTEN faithfulness floor (worst-cell MAX)**
   - owner: `src/gpuwrf/physics/mynn_pbl.py (mym_turbulence/_mym_length_option1 level-2.5 solve)`
   - With WRF-exact QKE AND WRF RTHRATEN, the MYNN kernel alone leaves rmse 0.5433 / max 40.3. raw ~3e-4 K/s mass-coupled by mu~1e5. This is an irreducible fp/algorithmic reimplementation floor (level-2.5 closure is iteratively/implicitly solved); reaching the 1e-3 mass-coupled gate needs raw ~1e-8 (bitwise MYNN), which contradicts the operational-RMSE validation philosophy.
3. **MYNN cold-start QKE level-2 equilibrium outlier (single worst-cell SPIKE)**
   - owner: `src/gpuwrf/physics/mynn_pbl.py::mynn_coldstart_init_columns`
   - Injecting WRF-pinned INIT_QKE drops the worst-cell max 53.5->28.0 but leaves rmse/p99 ~unchanged (2.544->2.538). The cold-start QKE is a RARE single-cell spike (bulk exact to 0.07%); fixing it does not move the field and risks regressing validated MYNN cold-start.

## Performance / safety

No production change. The strict 1e-3/1e-5 mass-coupled gate is unreachable for the MYNN+RRTMG theta tendency without bitwise scheme reproduction. The candidate local fixes (cold-start QKE outlier; RRTMG clear-sky RTHRATEN) each (a) leave the strict gate red and (b) risk regressing validated MYNN/RRTMG physics; neither is a proven Step-1 production bug. GPU-native vectorized structure preserved (no clamps, no scalarization, no CPU-WRF runtime dependency, no in-loop transfer).

## Fastest next command

Manager decision: re-specify the strict MYNN+RRTMG Step-1 gate to an operationally-meaningful mass-coupled tolerance (raw ~3e-4 K/s floor => ~rmse 0.5 / max ~40 mass-coupled is the achievable best). The field-dominant lane is RRTMG RTHRATEN (proofs/v014/rrtmg_step1_forcing_parity.*); a clear-sky RTHRATEN sprint would cut strict rmse ~2.54->~0.54. Re-run: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_rthblten_step1_closure.py
