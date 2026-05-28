# JAX Dycore Deep Review & Findings

## 1. Root-cause hypothesis

The JAX dycore's operational forecast failure and the limiter's saturation ($8640/8640$ steps with an `Infinity` residual) trace to three critical, interacting bugs:

1. **Complete Deletion of Advection in Operational RK3 stages**:
   During the Milestone 6 horizontal pressure gradient fix (commit `db9249c`), the advection step was completely replaced with the horizontal pressure gradient calculation (`_horizontal_pressure_gradient_tendencies`) inside `_rk_scan_step` of [operational_mode.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/runtime/operational_mode.py). As a result, advection of momentum ($U, V, W$), theta ($\theta$), moisture ($Q_v, Q_c, \dots$), and pressure ($p$) is entirely disabled in operational mode.
   
2. **Loss of Column Dry Mass Perturbation (`mu_save`) in the Acoustic Loop**:
   In [acoustic.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/acoustic.py#L261), `acoustic_substep_core` replaces the grid mass perturbation `state.mu` with the small-step delta `mu_delta = advanced["muts"] - state.mut` instead of the new total physical perturbation `advanced["mu"]`. Consequently, when the next substep computes `mu_save = state.mu - mu_work_old`, the subtraction of the delta from itself yields `0`. This completely wipes out the background physical perturbation `mu_save` for all substeps after the first, leaving a mass-depleted system that violates conservation and leads to rapid pressure/density collapse.

3. **Incorrect Theta Decoupling State Reference**:
   In `_decouple_theta_after_advance` inside [acoustic.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/acoustic.py#L185), the numerator reconstructs the decoupled theta using the running substep theta (`state.theta`) instead of the stage-start saved theta (`state.theta_1`), directly violating the WRF split-explicit algorithm.

### Why the 100-step test passes:
The 100-step validation parity test (`test_dycore_100_steps.py`) utilizes the comparison script [m6b6_coupled_step_compare.py](file:///home/enric/src/wrf_gpu2/scripts/m6b6_coupled_step_compare.py), which contains a **testing tautology (self-compare)**. The comparator writes the JAX-produced outputs to disk during `emit_tier` and then reads them back to compare JAX to JAX. Since the test never verifies JAX against a true Fortran-generated oracle, the complete absence of advection and the `mu_save` depletion went completely unnoticed in CI. When running the coupled operational mode with real topography and boundaries, this unphysical dycore immediately diverges and explodes.

---

## 2. Why GPT-5.5 missed it

The two GPT workers failed due to the following reasoning errors:
1. **Unquestioning Trust in the Test Harness**: The workers assumed that because the 100-step parity test passed, the underlying dynamics equations were correct. They failed to audit the test comparator (`m6b6_coupled_step_compare.py`) and discover the self-compare tautology.
2. **Myopic Single-Line Fix Mentality**: They attempted local, isolated line tweaks on the theta decoupling formula or sign flips on `mu_tendency` without tracing the state variables (such as `mu_save` and `theta_1`) through multiple acoustic loops, overlooking the fact that `mu_save` was zeroing out.
3. **Overlooking commented-out/missing logic**: They did not verify if the large-timestep RK stages were actually computing advection. They missed that the core advection step had been completely excised from the operational loop.

---

## 3. Recommended next sprint

We recommend a coordinated multi-line refactor of the acoustic core and operational RK stages:

### A. Core Code Fixes:
1. **Restore Advection**: Re-introduce `compute_advection_tendencies` in the RK stages of `_rk_scan_step` in [operational_mode.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/runtime/operational_mode.py), combining advection tendencies and pressure gradient tendencies before advancing the stage.
2. **Fix `mu` Replacement in the Acoustic Loop**: In [acoustic.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/acoustic.py#L261), update the returned state in `acoustic_substep_core` to store the total physical perturbation:
   ```diff
   -mu=mu_delta,
   +mu=advanced["mu"],
   ```
3. **Fix Theta Decoupling**: In `_decouple_theta_after_advance` in [acoustic.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/acoustic.py#L188), reference the stage-start theta:
   ```diff
   -numerator = theta_mass + state.theta * (state.c1h[:, None, None] * state.mut[None, :, :] + state.c2h[:, None, None])
   +numerator = theta_mass + state.theta_1 * (state.c1h[:, None, None] * state.mut[None, :, :] + state.c2h[:, None, None])
   ```

### B. Acceptance Criteria (AC):
- **AC1 (Advection Active)**: The 1h diagnostic harness runs and verifies that `dycore_rk3` registers active delta contributions for advection.
- **AC2 (Mass Conservation)**: The positive-definite limiter activity drops significantly and the limiter mass residual remains bounded at machine epsilon.
- **AC3 (Finiteness)**: The 24h operational forecast runs to completion without hitting NaNs or Infinities.
- **AC4 (Oracle Parity)**: The validation test suite is repaired to compare JAX against actual pre-computed Fortran savepoints, achieving bitwise parity within tolerance.

---

## 4. Worst-case fallback

If the recommended coordinated fix fails to stabilize the coupled forecast runs, the current fractional-step operator splitting (dynamics followed sequentially by physics/boundaries) is structurally insufficient. The fallback is to restructure the operational driver to match WRF's integrated RK3 cadence:
- Compute physics tendencies at RK stage 1 and carry them as constant source terms.
- Compute advection and buoyancy dynamically at each RK stage.
- Pass the combined dynamic + physics tendencies into the acoustic loop.

---

AGY_REVIEW_COMPLETE
