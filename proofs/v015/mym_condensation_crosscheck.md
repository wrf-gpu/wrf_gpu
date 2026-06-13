# v0.15 mym_condensation CASE(2) transcription cross-check

Method: an INDEPENDENT scalar NumPy reference following the Fortran
(`phys/module_bl_mynnedmf.F` CASE(2), incl. both rh_hack branches, the
sequential rh updates, all z-dependent sigma floors, the tropopause cut and the
top-level zeroing) was hand-written and evaluated level-by-level on a 30-level
randomized column (mixed saturation 0.5–1.05 RH, random qc/qi/qs presence,
random qsq) and compared against the vectorized JAX kernel
(`gpuwrf.physics.mynn_sgs_cloud.mym_condensation_cloudpdf2`).

Result: **max relative mismatch 0.0** on qc_bl/qi_bl/cldfra_bl at every level
(19 cloudy levels, cf up to 0.959, k_tropo=28 agreed).

Runner: tests/test_v015_mynn_sgs_cloud.py covers structural contracts
(dry=cloud-free, saturated BL cloudy, ice partition below tice, shallow-cu
overwrite gating, qsq floor/production, rollback flag).

## qsq prognostic solve cross-check

The closure-2.6 `_mym_predict_qsq` tridiagonal (XLA primitive) was compared
against an independent scalar Thomas-algorithm reference implementing the WRF
rows (a/b/c/d incl. the pdq(kts)=pdq(kts+1) surface row, kmdz=rho*dfq
interfaces, dissipation bp=2*qkw/(b2*0.5*(el+el+1)), a(kte)=-1 zero-gradient
top, max(x,1e-17) floor) on 3 randomized 24-level columns:
**max rel diff 4.4e-16** (machine precision); top zero-gradient exact.
