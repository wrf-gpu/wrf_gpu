The following is an independent, orthogonal architectural review of the proposed **c2 dycore sprint** plan. 

---

# Gemini Architecture Review: Is the c2 Dycore Sprint the Right Next Move?

## Executive Summary
While the consensus from the manager, GPT scout, and Opus meta-review is to immediately initiate the 3–5 day **c2-A1 sprint** (followed by a 3-week implementation), this review takes a more cautious, mathematically focused position. 

The stability failures in the `c1` dycore (bubble blowout at 350s, negative pressure anomalies of -239 kPa, and high sanitization rates) are **numerical and mathematical in origin, not software architectural**. Rewriting the code structure to mirror Pace (FV3) or ICON4Py will resolve software engineering modularity, but it will **not** automatically solve the core physical-numerical coupling bugs. If you proceed with `c2` immediately, you risk spending 3 weeks building a beautifully structured, highly modular JAX codebase that *still* blows up at 350s because the underlying coordinate transformations, metric terms, and boundary couplings are mathematically incorrect.

**Recommendation:** **(B) Postpone the full c2 sprint by 2 days to execute a targeted "Numerical Stability Spike" on the c1 codebase.** Use this spike to isolate the mathematical requirements (particularly base-state decomposition and metric terms over steep terrain) before locking down the `c2` architectural skeleton.

---

### 1. Is c2 the Right Next Move?
The conclusion that *"c1 needs multiple WRF stabilization mechanisms all together, and sequential fixes won't converge"* is partially correct, but it conflates **numerical completeness** with **software architecture**. 

The `c1` core did not fail because the code was monolithic; it failed because it lacks the mathematical terms required to balance the equations of motion in a non-hydrostatic, compressible, regional framework. In NWP, the advection scheme, pressure gradient force (PGF), buoyancy, and metric terms form a tightly coupled system. If you omit the metric terms for sloping coordinate surfaces or map scale factors, the system is mathematically unbalanced, leading to linear error growth that eventually triggers non-linear numerical instability (blowup).

Porting the *architecture* of Pace or ICON4Py will give you clean modules (e.g., `AcousticDynamics`, `RayleighDamping`), which is excellent for long-term maintainability. However, **modularity does not yield stability**. If the mathematical formulations inside those clean modules are wrong, the code will still fail. 

The c2 sprint is the right move for software engineering, but it is **too early** to commit to a 3-week implementation until the exact mathematical formulations of the WRF-compatible coordinate system are proven in a simplified environment.

---

### 2. Pace/ICON4Py vs. Alternatives
The manager’s choice of Pace/FV3 as the primary architectural reference has a major **grid-type and coordinate-system mismatch**:

| Feature | WRF (Target) | Pace / FV3 (Reference) | ICON / ICON4Py (Reference) |
| :--- | :--- | :--- | :--- |
| **Grid Type** | Arakawa C-Grid (Rectangular) | Arakawa D-Grid (Cubed-Sphere) | Arakawa C-Grid (Triangular/Icosahedral) |
| **Vertical Coordinate** | Hybrid-eta ($\eta$) mass-coordinate | Lagrangian control volume + Remapping | Non-hydrostatic terrain-following |
| **Horizontal Operators** | Finite Difference (2nd/5th/6th order) | Finite Volume (flux-form Lin-Rood) | Mimetic Finite Volume/Difference |

#### The Mismatch Risks:
* **Arakawa C-Grid vs. D-Grid:** Pace stores velocities on cell edges parallel to the edge (D-grid), whereas WRF stores them normal to cell faces (C-grid). The horizontal reconstruction, interpolation, and acoustic step logic are fundamentally different. Trying to map Pace’s architectural patterns onto a C-grid will lead to significant implementation impedance mismatch.
* **Vertical Remapping vs. Terrain-Following:** FV3 uses a Lagrangian vertical coordinate that deforms with the flow and is periodically remapped to a reference grid. WRF uses an Eulerian, terrain-following, hybrid-pressure coordinate. The acoustic step in WRF solves a vertical tridiagonal system for geopotential perturbation ($\phi'$) and vertical velocity ($w$). Pace's vertical solver architecture will not map to WRF's vertical solver.

#### Better References:
1. **Dinosaur / NeuralGCM (JAX):** Excellent reference for *software design patterns in JAX* (functional state passing, pytree structures, and using `jax.lax.scan` for time integration). Use this for JAX-style, but **not** for physics or grid numerical schemes.
2. **COSMO-DYCORE (GPU / GridTools):** COSMO is a regional, C-grid, non-hydrostatic model with a terrain-following coordinate, very similar to WRF. Its GPU port (using C++ GridTools) is a much closer algorithmic relative than Pace.
3. **WRF `dyn_em` Source Code:** The mathematical equations must come **strictly** from the WRF source. Trying to adapt Pace's or ICON's equations to match WRF boundary files (`wrfinput`/`wrfbdy`) is a recipe for permanent incompatibility.

---

### 3. Hidden Architectural Risks of JAX in Regional NWP
We are attempting a **world-first**: a regional, WRF-compatible, C-grid, hybrid-eta dycore in pure JAX. This comes with three critical architectural risks:

#### A. GT4Py (DSL) vs. JAX (XLA) Compilation Dynamics
Pace and ICON4Py rely on GT4Py, which compiles stencil operations directly to specialized CUDA kernels with custom memory layouts and domain-specific optimizations (like automated halo exchanges).
* **XLA Stencil Fusion:** JAX compiles via XLA, which was designed for dense linear algebra (GEMM). While XLA is excellent at fusing pointwise operations, it historically struggles with complex 3D stencil patterns (like 5th-order advection + metric terms). If stencils are not written carefully, XLA will generate excessive global memory reads/writes, destroying GPU memory bandwidth.
* **Compilation Time Pressure:** A complete regional dycore contains hundreds of stencil operations. If compile-time constants are not managed correctly, JAX compile times can exceed 30 minutes, crippling developer iteration speed.

#### B. Lateral Boundary Conditions (LBCs) in JAX
Unlike global models (NeuralGCM/Dinosaur), a regional model requires updating the outer 5-point boundary zone (halo) with time-dependent LBCs (from `wrfbdy`) at every Runge-Kutta step. 
* In JAX, slicing and modifying array boundaries (e.g., `x = x.at[boundary].set(lbc_val)`) returns a new array. If not optimized by XLA into in-place updates, this will lead to massive GPU memory fragmentation and overhead.

#### C. Tridiagonal Solver Cost
The implicit acoustic solver in the Klemp-Skamarock split-explicit scheme requires solving a tridiagonal system in every vertical column. In JAX, this must be done using `jax.lax.associative_scan` or a custom Thomas algorithm. If written naively, this vertical coupling prevents horizontal vectorization and becomes a massive GPU bottleneck.

---

### 4. The Hybrid-Eta & Orography Question
Adding `c1h`/`c2h`/`c3h`/`c4h` arrays to `c2` is necessary, but **grossly insufficient** on its own. 

The Canary Islands present some of the steepest terrain in regional modeling (Mount Teide rises to 3,718m over a horizontal distance of ~15km). If you run a 3km model over this terrain, you must address two non-negotiable mathematical requirements:

1. **Metric Terms for Sloping Surfaces:** 
   In a terrain-following coordinate system, horizontal derivatives (e.g., $\frac{\partial p}{\partial x}$) cannot be computed along coordinate surfaces without correction terms:
   $$\left(\frac{\partial p}{\partial x}\right)_z = \left(\frac{\partial p}{\partial x}\right)_\eta - \frac{g}{\alpha} \frac{\partial z}{\partial x} \frac{\partial p}{\partial \eta}$$
   If these slope correction terms ($\frac{\partial z}{\partial x}$) are omitted or poorly discretized, the Pressure Gradient Force (PGF) will calculate spurious horizontal winds of 50–100 m/s near the volcanic slopes, causing immediate numerical blowup.
2. **Base-State vs. Perturbation Decomposition:**
   To prevent catastrophic truncation errors in the PGF over steep slopes, WRF decomposes geopotential ($\phi$) and pressure ($p$) into a hydrostatic base-state ($\bar{\phi}, \bar{p}$) and a perturbation ($\phi', p'$). The model solves for the perturbation variables. If `c1` solved for total variables, the truncation error alone explains the negative pressure anomalies (-239 kPa) and linear RMSE growth. **This decomposition must be native to the `c2` state representation from day one.**

---

### 5. Will the Constitutional 4× Speedup Target Survive?
Yes, the 4× speedup target is highly secure. Even with the added complexity of map factors, hybrid-eta metrics, 6 damping modules, and `lax.scan` loops, the RTX 5090 has a memory bandwidth of ~2.0 TB/s.

NWP dycores are heavily memory-bandwidth bound. Adding stabilization terms (like 6th-order hyperdiffusion or Rayleigh sponges) adds computational density (FLOPs/byte) but does not significantly increase the number of large array reads/writes, *provided* XLA fuses these operations into a single kernel pass. 

* The `c1-A7` speedup of **44.33×** will likely compile down to **20×–30×** once the full physics-coupling, LBC updates, and metric terms are implemented. This remains well above the 4× constitutional target.

---

### 6. Honest Recommendation

My recommendation is **(B) Postpone the full c2 sprint — execute a 2-day "Numerical Stability Spike" first.**

#### The Plan:
1. **Freeze the `c2` planning for 48 hours.**
2. **Run two diagnostic tests on the existing `c1` branch:**
   * **Test 1 (Flat vs. Mountain):** Run a 2D vertical slice test case over flat terrain, and then over a steep idealized ridge (e.g., Schär mountain wave test). If the flat case is stable for 3,000s but the mountain case blows up at 300s, you have isolated the bug to **missing coordinate metric terms** and PGF truncation errors, confirming that base-state decomposition is your #1 priority.
   * **Test 2 (Brute-Force Damping Hack):** Hardcode a simple divergence damping (`smdiv`) and a basic vertical Rayleigh sponge in `c1`. If these crude additions stabilize the warm bubble past 350s, it proves your core solver mathematics are sound and only require damping. If it still blows up, your pressure solver or acoustic time-splitting has a fundamental formulation error.
3. **Use the findings from this 2-day spike to write the `c2` Architecture Decision Record (ADR).** You will then know exactly what state variables must be decomposed ($\bar{p}$ vs $p'$) and what metric terms must be carried in `GridData` before writing a single line of the new framework.

This approach prevents the team from blindly porting Pace's structural layout, only to realize weeks later that they built the wrong mathematical foundation for a terrain-following, WRF-compatible regional model.
