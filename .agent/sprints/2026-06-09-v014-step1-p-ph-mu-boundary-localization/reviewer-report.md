# Reviewer Report: V0.14 Step-1 P/PH/MU Boundary Localization

## Findings

- The sprint met its contract with a focused CPU-only source/substage
  comparator and did not run GPU validation or long forecasts.
- The old pre-theta-fix `T_STATE` localization is stale for the current branch:
  the current rerun localizes the first P-family state residual to `P_STATE`
  after WRF `first_rk_step_part1`.
- The residual is before WRF dry boundary application and before JAX lateral
  boundary application, so boundary application is not the first checked cause.
- RK1 `small_step_prep` and `calc_p_rho(step=0)` work arrays are exact for the
  checked work fields, so the current first checked mismatch is upstream of
  those work arrays.
- No source fix is justified yet because the existing truth does not split the
  internal `first_rk_step_part1` state writes or post-acoustic/pre-refresh
  pressure refresh.

## Correctness Risks

The proof does not prove boundary-package equality for raw `p/ph/mu` leaves.
It also does not prove whether the final `P` residual is from acoustic finish,
pressure refresh, or an upstream state write propagated through later stages.

## Performance Risks

None introduced. No production source was changed, and the proof does not add
host/device transfer to any timestep loop.

## Required Fixes

Open the next localization sprint to emit one WRF scratch surface inside
`first_rk_step_part1` around `phy_prep`/`calc_p_rho_phi` state writes for
`P/MU/W`, or explicitly split post-acoustic/pre-refresh pressure if the manager
chooses to chase the downstream final `P` residual first.

Decision:

Approve merge of proof and closeout artifacts. Keep TOST, Switzerland, FP32
source work, and memory follow-ups paused behind grid parity.
