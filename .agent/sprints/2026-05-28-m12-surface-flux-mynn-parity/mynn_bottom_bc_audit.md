# M12 MYNN Bottom-BC Sign Audit

## Verdict

`PASS_WITH_DENSITY_SCALE_FIX`.

The GPU adapter uses the WRF sign convention for surface scalar and momentum fluxes:

- `theta_flux`: kinematic sensible heat flux, positive upward into the atmosphere.
- `qv_flux`: kinematic water-vapor flux, positive upward into the atmosphere.
- `tau_u`, `tau_v`: signed kinematic momentum flux components, positive upward into the atmosphere; for wind over fixed ground these are normally opposite-signed to the lowest model-level wind.

No scalar sign inversion was found. The explicit momentum bottom tendency was changed from `dt/dz * tau` to `dt/dz * rhosfc/rho0 * tau`, matching WRF's rho-weighted bottom drag scaling.

## WRF Reference

- `phys/MYNN-EDMF/module_bl_mynnedmf.F90:4240-4241` documents `flt` and `flq` as surface fluxes for heat and water.
- `phys/MYNN-EDMF/module_bl_mynnedmf.F90:4581-4589` adds bottom heat flux as `+ dtz*rhosfc*flt*rhoinv`.
- `phys/MYNN-EDMF/module_bl_mynnedmf.F90:4781-4796` limits negative water-vapor flux, then adds `+ dtz*rhosfc*qvflux*rhoinv`.
- `phys/MYNN-EDMF/module_bl_mynnedmf.F90:4434-4437` and `4510` put bottom momentum drag into the rho-weighted U/V implicit systems as `rhosfc*ust**2/wspd`.

The project ADR-008 interface notes agree: `theta_flux`, `qv_flux`, and `fltv` are positive upward; `tau_u/tau_v` are positive upward components and opposite the lowest-level wind for fixed-ground drag.

## GPU Code Audit

- `src/gpuwrf/coupling/physics_couplers.py:146-154` places the surface fluxes in bottom-column slot 0 only.
- `src/gpuwrf/coupling/physics_couplers.py:322-331` applies scalar increments with `+ dt/dz*rhosfc/rho0`, matching WRF's positive-upward RHS sign.
- `src/gpuwrf/coupling/physics_couplers.py:332-335` now applies signed momentum flux components with the same `rhosfc/rho0` scaling. Because `surface_layer.py` defines drag flux as `tau_u = -ustar**2*u/|u|` and `tau_v = -ustar**2*v/|v|`, adding `tau` damps positive wind and accelerates negative wind toward zero.

## Remaining Limitation

The dry MYNN column kernel still computes its internal neutral-bulk surface terms before the adapter's explicit real-surface bottom correction. This audit only covers the adapter sign convention and the explicit bottom-BC application permitted by the M12 writable-file scope.
