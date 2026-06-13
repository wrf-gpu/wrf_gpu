# v0.15 Thompson riming constants — direct WRF-Fortran cross-check

All constants for the cold-phase riming block validated against the WRF
module_mp_thompson.F construction (V4.7.1):

- Snow Ds bins: Ds(1)=3.0637e-4, Ds(100)=1.9584e-2 m -- the log-bin geometry
  (D0s=300um -> 2cm, Ds(n)=sqrt(xDx(n)*xDx(n+1))) matches to <1e-12.
- t1_qs_qc = PI*0.25*av_s = 31.4159 (line 794).
- cge(9) = bv_g+3+mu_g = 3.6410; t1_qg_qc = PI*0.25*av_g*Gamma(cge9) = 438.087
  (line 766/2432; mu_g=0 single-density mp8).
- graupel Stokes vtg moment ratio Gamma(bv_g+4)/Gamma(4) = 2.3636 (line 2421).
- Snow-bin lookup index: a 1 mm snow particle maps to the SAME bin as the WRF
  Fortran ``1 + INT(nbs*DLOG(xDs/Ds(1))/DLOG(Ds(nbs)/Ds(1)))`` (0-based 28).

Behavioral check (supercooled marine/Alpine column, dt=18 s): riming ON moves
supercooled qc into qs/qg (qg 0 -> 1.5e-5, qc sink), conserving column water;
riming OFF (GPUWRF_THOMPSON_RIMING=0) keeps qg=0 (deposition-only). Single-step
precip-oracle parity vs WRF RAINNCV stays inside the frozen +-3% band with
riming ON (tests/test_thompson_precip_oracle.py).
