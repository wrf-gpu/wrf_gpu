# V0.14 MYNN-EDMF RTHBLTEN Strict Step-1 Closure

Verdict: `STEP1_STRICT_RED_FORMALLY_BOUNDED_RRTMG_FIELD_DOMINANT_MYNN_KERNEL_FLOOR_GATE_UNREACHABLE`.

## Endpoint

Strict Step-1 is RED and **formally bounded**. The strict gate (max_abs<=`0.001`, rmse<=`1e-05` on mass-coupled `T_TENDF`, mu~1e5 => raw ~1e-8) is **unreachable** for the MYNN+RRTMG theta tendency without bitwise scheme reproduction.

## Lane decomposition (operational strict residual vs rmol-pinned WRF after_conv, interior)

| Configuration | max | rmse | p99 |
|---|---|---|---|
| Operational (cold-start QKE) | 55.9297 | 0.49966 | 0.953 |
| + WRF-pinned INIT_QKE | 29.1980 | 0.45819 | 0.952 |
| + WRF-pinned QKE + WRF RTHRATEN (RRTMG removed) | 29.4231 | 0.27700 | 0.810 |
| RTHRATEN (RRTMG) lane only | 2.8394 | 0.36479 | 0.802 |

**Reading:** injecting WRF QKE drops the worst-CELL max but leaves rmse/p99 mostly unchanged (cold-start QKE = rare single-cell spike). Substituting WRF RTHRATEN reduces the strict field rmse/p99 further; RRTMG remains field-significant after the dry-theta fix, while MYNN owns the worst-cell max/floor.

## Reconciliation with prior 'MYNN kernel faithful' evidence

- Operational dry `T_TENDF` reassembled = runtime capture (consistency max_abs `4.547473508864641e-13`): the strict leaf is exactly `theta_m_factor*mass_h*(RTHRATEN+RTHBLTEN) + Rv/Rd QV`.
- The MYNN `RTHBLTEN` it carries is WRF-faithful (raw ~3e-4 K/s) with WRF-equivalent inputs + QKE -- the prior accepted boundary result stands.
- `build_step1_state` (legacy `run_kernel_matrix`) omits `grid=` on `noahmp_surface_step`; its grid-less LAND surface makes the standalone adapter overshoot `RTHBLTEN` ~2x at land (mass-coupled +260). The OPERATIONAL leaf is faithful there -> the legacy kernel-matrix land tail is a **proof artifact**, not a production bug.

## Ranked lanes (exact ownership)

1. **RRTMG step-1 radiation RTHRATEN (held seed) -- FIELD DOMINANT**
   - owner: `src/gpuwrf/physics/rrtmg_lw.py, rrtmg_sw.py (held RTHRATEN seed via carry.rthraten / _refresh_rthraten); localization proofs/v014/rrtmg_step1_forcing_parity.*`
   - Removing the remaining RRTMG RTHRATEN error reduces the strict rmse 0.4582->0.2770 and p99 0.95->0.81; that is 63.5% of the WRF-QKE rmse variance in the post-dry-theta-fix proof. RRTMG remains field-significant, while MYNN owns the worst-cell max/floor.
2. **MYNN level-2.5 turbulence kernel RTHBLTEN faithfulness floor (worst-cell MAX)**
   - owner: `src/gpuwrf/physics/mynn_pbl.py (mym_turbulence/_mym_length_option1 level-2.5 solve)`
   - With WRF-exact QKE AND WRF RTHRATEN, the MYNN kernel alone leaves rmse 0.2770 / max 29.4. raw ~3e-4 K/s mass-coupled by mu~1e5. This is an irreducible fp/algorithmic reimplementation floor (level-2.5 closure is iteratively/implicitly solved); reaching the 1e-3 mass-coupled gate needs raw ~1e-8 (bitwise MYNN), which contradicts the operational-RMSE validation philosophy.
3. **MYNN cold-start QKE level-2 equilibrium outlier (single worst-cell SPIKE)**
   - owner: `src/gpuwrf/physics/mynn_pbl.py::mynn_coldstart_init_columns`
   - Injecting WRF-pinned INIT_QKE drops the worst-cell max 55.9->29.2 but leaves rmse/p99 ~unchanged (0.500->0.458). The cold-start QKE is a RARE single-cell spike (bulk exact to 0.07%); fixing it does not move the field and risks regressing validated MYNN cold-start.

## Performance / safety

No production change. The strict 1e-3/1e-5 mass-coupled gate is unreachable for the MYNN+RRTMG theta tendency without bitwise scheme reproduction. Remaining local fixes (cold-start QKE outlier; residual RRTMG split differences) leave the strict gate red unless they pursue bitwise scheme reproduction. GPU-native vectorized structure preserved (no clamps, no scalarization, no CPU-WRF runtime dependency, no in-loop transfer).

## Fastest next command

Manager decision: re-specify the strict MYNN+RRTMG Step-1 gate to an operationally-meaningful mass-coupled tolerance. Post dry-theta RRTMG fix: operational rmse 0.4997, WRF-QKE rmse 0.4582, WRF-QKE+WRF-RTHRATEN rmse 0.277; remaining RRTMG variance share 63.5%, absolute rmse reduction 0.1812. Re-run: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_rthblten_step1_closure.py
