# Manager Closeout

Merge Decision: ACCEPT_AND_COMMIT.

The sprint achieved the contract's fallback endpoint. It did not close strict
Step-1, but it closed the first suspected production bug with a WRF-anchored
proof and narrowed the remaining blocker beyond the original two-lane split.

Accepted result:

- Production fix in `noahmp_surface_hook._build_column_view(state, grid=...)`
  supplies WRF `phy_prep` dry `t_air`, true `psfc`, hydrostatic `p`, and density
  to the NoahMP/sfclay path.
- `operational_mode._physics_step_forcing` passes `grid=namelist.grid` so the
  operational path uses the WRF-faithful view.
- Water-path surface flux proof is strong: water HFX RMSE
  `11.87 -> 1.37 -> 0.0118 W/m2`, water `ust` near exact.
- Strict Step-1 improves from max_abs `1489.51`, RMSE `12.15` to max_abs
  `53.52`, RMSE `2.54`, but remains red and release-blocking.

Next manager action: open an endpoint-sized MYNN-EDMF `RTHBLTEN` sprint. That
sprint should own the relevant MYNN kernel files, keep performance structure
intact, and produce either strict Step-1 green or a WRF-anchored formal bound.
RRTMG is secondary and should follow after MYNN unless new proof reverses the
ranking.

TOST, Switzerland-GPU, and FP32 R1/R2 stay paused.
