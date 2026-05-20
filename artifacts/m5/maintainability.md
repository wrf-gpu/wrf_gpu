# Thompson M5-S1 Maintainability

Fixture generation used Path B. Path A would require compiling WRF's full `module_mp_thompson.F.pre` dependency stack (`module_wrf_error`, `module_mp_radar`, lookup-table initialization, WRF preprocessor flags), which is not self-contained in this repo's M5 worker scope. The Path-B formulas are source-mapped to the WRF snapshot and intentionally limited to sedimentation-free source/sink processes.

Included: driver boundary species prep and rho formula (lines 1070-1274), saturation over water/ice (lines 5444-5490), cloud condensation Newton adjustment (lines 3456-3556), rain evaporation shape (lines 3561-3633), freezing/melting phase changes (lines 4000-4152), and tendency/update constraints (lines 3024-3273).

Skipped because sedimentation is out of M5-S1: terminal velocity, substepping, and flux-divergence sections beginning at lines 3655-3972. Also skipped: aerosol scavenging/activation tables and radar/effective-radius diagnostics, because they are optional diagnostics outside the frozen M5-S1 output state.

The JAX kernel is one public `@jax.jit` with `dt` and `debug` static. The stripped sibling physically omits debug hooks; diff sha256 is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
