# Thompson M5-S1 Maintainability

Fixture generation used Path B-strict after Path A investigation found no reusable `module_mp_thompson` object/module under `../wrf_gpu` or `/mnt/data/wrf_gpu2`; only `wrf.exe` binaries and the `.F.pre` source snapshot were present. A direct wrapper compile would need `module_wrf_error`, `module_mp_radar`, model constants, preprocessor flags, and lookup-table init outside this worker's file ownership.

Included: driver boundary species prep and rho formula (lines 1070-1274), saturation over water/ice (lines 5444-5495), cloud condensation Newton adjustment (lines 3456-3556), Berry-Reinhardt autoconversion (lines 2242-2258), rain-cloud-water collection shape (lines 2260-2268, with bounded collection-efficiency proxy because `t_Efrw` is a generated table), Srivastava-Coen rain evaporation (lines 3561-3636), cloud-ice/snow/graupel deposition and sublimation (lines 2709-2770), rain freezing + snow/graupel melting (lines 2658-2669 and 2845-2889), and final mass/number constraints (lines 4033-4142).

Skipped because sedimentation is out of M5-S1: terminal velocity, substepping, and flux-divergence sections beginning at lines 3655-3972. Also skipped: aerosol activation/scavenging, WRF-generated lookup tables that are not fixture inputs (`t_Efrw`, freezing tables), radar/effective-radius diagnostics, and graupel volume/hail state.

The fixture oracle uses a WRF-style NumPy tendency ledger (`qvten`, `qcten`, `tten`) and process-rate names; it does not call the JAX kernel or share its helper sequence. The JAX kernel is one public `@jax.jit` with `dt` and `debug` static. The stripped sibling physically omits debug hooks; diff sha256 is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
