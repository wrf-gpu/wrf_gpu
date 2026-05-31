# Independent Cross-Check Findings: wrf_gpu2 Verdict Validation

This document presents a read-only independent cross-check of the JAX GPU port of WRF v4 (Canary d02) at the manager-2026-05-23 state.

---

## A. Verdict Methodology Evaluation

We evaluated the validation harness in [verdict_3case.py](file:///home/enric/src/wrf_gpu2/proofs/m19/verdict_3case.py), [verdict_result.json](file:///home/enric/src/wrf_gpu2/proofs/m19/verdict_result.json), and [cases_manifest.json](file:///home/enric/src/wrf_gpu2/proofs/m19/cases_manifest.json). The 3-case verdict has returned a `PASS` rating. However, the methodology contains critical gotchas that make this pass misleading:

### 1. Gridded Whole-Domain RMSE Masking Topographic Errors
* **Topography vs. Ocean Ratio:** In the Canary Islands $159 \times 66$ nest domain d02 ([cases_manifest.json:L8](file:///home/enric/src/wrf_gpu2/proofs/m19/cases_manifest.json#L8)), approximately **92% of the grid points represent open ocean** (flat terrain, constant SST, no topography).
* **Topographic Masking:** Topography-induced vertical velocity boundary conditions and land-surface processes affect less than 8% of the grid cells. Because the whole-domain gridded RMSE calculation ([verdict_3case.py:L97-113](file:///home/enric/src/wrf_gpu2/proofs/m19/verdict_3case.py#L97-L113)) averages all grid cells equally, massive localized errors (e.g., the $10^5$ scaling error in the terrain boundary condition for $w$ detailed below) over the steep Canary volcanic terrain are completely diluted and hidden by the quiet ocean background.

### 2. Generous Thresholds and Persistence Vulnerability
* **High Ceilings:** The ceilings for a pass are set to generous values ([verdict_3case.py:L72-77](file:///home/enric/src/wrf_gpu2/proofs/m19/verdict_3case.py#L72-L77)):
  * **T2:** 5.0 K (24h) to 7.0 K (72h)
  * **U10/V10:** 6.0 m/s (24h) to 9.0 m/s (72h)
* **Persistence Scores:** Because the Canary Islands are located in a highly steady subtropical trade-wind regime, the meteorological state exhibits extreme persistence day-to-day. Over a 24-72h lead time, a dummy "persistence model" that does not integrate time at all (retaining the $t=0$ initial state) would yield a T2 RMSE of $\sim 1.0-1.5$ K and wind RMSE of $\sim 2.0-3.0$ m/s, easily passing the validation harness. Consequently, the validation harness is incapable of rejecting a broken forecast that fails to capture the true dynamics.

### 3. Lateral Boundary Nudging Constraints
* **Predictability Illusion:** The RMSE not degrading with lead time (e.g., Case 2 V10 RMSE dropping from 3.27 m/s at 24h to 2.77 m/s at 72h) is a geometric artifact. The d02 domain is extremely narrow in the y-direction (66 grid cells = 198 km). Under typical trade wind speeds, the entire domain's air is replaced by lateral boundary conditions every $\approx 5.5$ hours. Since the specified outer boundary relaxation zone ([operational_mode.py:L1523-1541](file:///home/enric/src/wrf_gpu2/src/gpuwrf/runtime/operational_mode.py#L1523-1541)) constantly injects the CPU-WRF history files (the "truth"), the model is strongly guided and prevented from diverging chaotically. This boundary nudging masks the fact that the interior dynamics are physically incorrect.

---

## B. Evaluation of the bbfa269 Coupler Fix

We evaluated the coupler implementations in [physics_couplers.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/coupling/physics_couplers.py) and [acoustic.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/acoustic.py), and cross-checked them against pristine CPU-WRF code.

### 1. The Terrain $w$ Boundary Condition Scaling Gotcha
* **The Bug:** In [acoustic.py:L583-584](file:///home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/acoustic.py#L583-L584), the coupler passes the uncoupled physical winds `uv_state.u_1` and `uv_state.v_1` to [advance_w_wrf](file:///home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/advance_w.py#L131):
  ```python
  u=uv_state.u_1,
  v=uv_state.v_1,
  ```
  This is physically incorrect. In CPU-WRF [solve_em.F:L1500-1501](file:///home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F#L1500-1501), the arrays passed to `advance_w` are `grid%u_2` and `grid%v_2`:
  ```fortran
  CALL advance_w( grid%w_2, rw_tend, grid%ww, w_save, &
                  grid%u_2, grid%v_2,                 &
  ```
  Inside the acoustic small-step loop, `grid%u_2` and `grid%v_2` are **coupled perturbation work variables** (multiplied by the column dry mass $\mu \approx 10^5 \text{ Pa}$ and divided by map scale factors). 
* **The Scaling Impact:** By feeding decoupled winds, [advance_w_wrf](file:///home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/advance_w.py#L300-L303) calculates the terrain vertical velocity `w_surface` in physical m/s ($O(1)$) instead of coupled units ($O(10^5)$). At the end of the acoustic stage, [small_step_finish_wrf](file:///home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/small_step_finish.py#L53) decouples `w` using the mass-coupled formula:
  ```python
  w = (prep.msfty[None, :, :] * w_work + prep.w_save * mass_w_current) / _safe_denominator(mass_w_stage)
  ```
  Because `w_work[0]` (the surface face) is uncoupled, it is divided by `mass_w_stage` ($\approx 10^5 \text{ Pa}$), scaling the terrain-forced vertical velocity down by $10^5$. This effectively silences mountain-wave generation in the JAX model.

### 2. MYNN Staggering & Boundary Mismatches
We identified boundary-treatment differences between JAX and WRF in the PBL incremental momentum coupling:
* **`_add_a2c_u_increment` Staggering mismatch:**
  * **JAX:** In [physics_couplers.py:L779](file:///home/enric/src/wrf_gpu2/src/gpuwrf/coupling/physics_couplers.py#L779), JAX updates the entire y-axis row slice `:`:
    ```python
    du_face = du_face.at[:, :, 1:-1].set(interior)
    ```
  * **WRF:** In [module_physics_addtendc.F:L2567-2569](file:///home/enric/src/wrf_pristine/WRF/phys/module_physics_addtendc.F#L2567-L2569), WRF restricts the y-axis loop bounds for specified/nested boundaries:
    ```fortran
    IF ( config_flags%specified .or. config_flags%nested) j_start = MAX(jds+1,jts)
    IF ( config_flags%specified .or. config_flags%nested) j_end   = MIN(jde-2,jte)
    ```
    This means WRF does *not* apply PBL tendencies on the northern/southern boundary rows ($j=1$ and $j=ny$), whereas JAX does.
* **`_add_a2c_v_increment` Staggering mismatch:**
  * **JAX:** In [physics_couplers.py:L794](file:///home/enric/src/wrf_gpu2/src/gpuwrf/coupling/physics_couplers.py#L794), JAX updates the entire x-axis column slice `:`:
    ```python
    dv_face = dv_face.at[:, 1:-1, :].set(interior)
    ```
  * **WRF:** In [module_physics_addtendc.F:L2617-2619](file:///home/enric/src/wrf_pristine/WRF/phys/module_physics_addtendc.F#L2617-L2619), WRF restricts the x-axis loop bounds:
    ```fortran
    IF ( config_flags%specified .or. config_flags%nested) i_start = MAX(ids+1,its)
    IF ( config_flags%specified .or. config_flags%nested) i_end   = MIN(ide-2,ite)
    ```
    This means WRF does *not* apply PBL tendencies on the eastern/western boundary columns ($i=1$ and $i=nx$), whereas JAX does.

---

AGY_CROSSCHECK_COMPLETE

**VERDICT:** NO-FLY. The fly call is UNSOUND because the `bbfa269` terrain-w coupling fix scales down mountain-wave generation by $10^5$, which is masked by the ocean-dominated whole-domain RMSE harness.
