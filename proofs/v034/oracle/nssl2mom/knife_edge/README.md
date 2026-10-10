# Case-15 knife-edge in pristine WRF (lane o1-nssl)

`driver_knife.F90` = `nssl2mom_oracle_driver_v2.F90` + one hook: QC at Fortran level 13 of case 15 is scaled by
(1 + O1_EPS) after the column is built. Linked against the UNMODIFIED fp64 module object
(`-fdefault-real-8`, same flags as build_and_run.sh). `cn_out_k13_vs_eps.txt`: final activated CCN (CN, #/kg) at
that level. WRF itself switches between 1.762593549821861e7 (eps=0, the stored oracle) and 1.550161978460388e7
for |eps| ~ 2e-16 .. 1e-13 (one rounding unit of the 5.4e-5 kg/kg input): nssl_2mom_gs line 24553 zeroes the
droplet number only if the post-freezing cloud mass is exactly <= 0, so the sign of a round-off residual decides
whether droplets survive into NUCOND. The JAX port lands on the second branch (1.5501619784604e7, rel 8e-15 to
WRF's value); tests/v034/nssl2mom/test_column_e2e.py accepts exactly these two WRF values at that cell.
