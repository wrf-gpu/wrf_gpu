! =====================================================================
! v0.13 single-column WSM7 (WRF mp_physics=24) oracle driver.
!
! Drives the UNMODIFIED WRF phys/module_mp_wsm7.F scheme:
!   wsm7init       (graupel/hail mode: WRF default; runs radar_init)
!   wsm72D         (the full 7-class single-moment hail microphysics; called
!                   directly on a single column with qci(1:2), qrs(1:4))
!   effectRad_wsm7 (effective radii for radiation)
! on a prescribed single column (idim=1) and dumps the FULL input state
! (t,qv,qc,qr,qi,qs,qg,qh,den,p,delz,pii) and the FULL output state
! (t,qv,qc,qr,qi,qs,qg,qh after the call) plus surface accumulators
! (rain,rainncv,snow,snowncv,graupel,graupelncv,hail,hailncv,sr) and
! effective radii (re_cloud,re_ice,re_snow).
!
! This is the GOLD oracle for the JAX port: it is the real Fortran WSM7
! scheme (module_mp_wsm7.F), not a re-implementation, so the JAX port cannot
! "self-compare". The init is the real wsm7init so every saved module
! constant (gammas, slope maxima, fall-speed prefactors including hail) is
! exactly WRF.
!
! Usage: ./wsm7_oracle <case_id>     (1..6 regimes, see build_column)
! Output: flat key=value text dump on stdout (parsed by Python into JSON).
!
! Default REAL = REAL*4 (single precision); the JAX port runs fp64. Parity is
! therefore to a PREDECLARED physical tolerance, never bitwise (see
! run_wsm7_parity.py). A -DDOUBLE_PRECISION build (-fdefault-real-8) provides
! the fp64 reference used for the categorical effective-radius diagnostics.
! =====================================================================
PROGRAM wsm7_oracle
  USE module_mp_wsm7, ONLY : wsm7init, wsm72D, effectRad_wsm7
  IMPLICIT NONE

  ! ---- WRF model constants (share/module_model_constants.F), bound exactly
  !      as in module_microphysics_driver.F CALL wsm7(...) ----
  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0     ! cpd
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: CPV    = 4.0*R_V
  REAL, PARAMETER :: CLIQ   = 4190.0
  REAL, PARAMETER :: CICE   = 2106.0
  REAL, PARAMETER :: PSAT   = 610.78
  REAL, PARAMETER :: XLV    = 2.5E6           ! xlv0
  REAL, PARAMETER :: XLS    = 2.85E6
  REAL, PARAMETER :: XLF    = 3.50E5          ! xlf0
  REAL, PARAMETER :: SVPT0  = 273.15          ! t0c
  REAL, PARAMETER :: EP_1   = R_V/R_D - 1.0
  REAL, PARAMETER :: EP_2   = R_D/R_V
  REAL, PARAMETER :: EPSILON= 1.E-15          ! qmin
  REAL, PARAMETER :: RHOAIR0= 1.28            ! den0
  REAL, PARAMETER :: RHOWATER=1000.0          ! denr
  REAL, PARAMETER :: DENS_SNOW=100.0          ! dens (wsm7init arg)
  REAL, PARAMETER :: P1000  = 1.0E5
  REAL, PARAMETER :: ROVCP  = R_D/CP

  ! effective-radius background/max values (module_microphysics_driver.F defaults)
  REAL, PARAMETER :: RE_QC_BG = 2.49E-6
  REAL, PARAMETER :: RE_QI_BG = 4.99E-6
  REAL, PARAMETER :: RE_QS_BG = 9.99E-6

  INTEGER, PARAMETER :: KX  = 40
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX
  INTEGER, PARAMETER :: lat=1

  ! single column (im=1) arrays
  REAL, DIMENSION(its:ite,kts:kte)   :: t, q
  REAL, DIMENSION(its:ite,kts:kte,2) :: qci
  REAL, DIMENSION(its:ite,kts:kte,4) :: qrs
  REAL, DIMENSION(its:ite,kts:kte)   :: den, p, delz, pii
  REAL, DIMENSION(kts:kte)           :: re_qc, re_qi, re_qs
  REAL, DIMENSION(its:ite)           :: rain, rainncv, sr
  REAL, DIMENSION(its:ite,jts:jte)   :: snow, snowncv, graupel, graupelncv, hail, hailncv

  ! 1d copies for effectRad
  REAL, DIMENSION(kts:kte) :: t1d, qc1d, qi1d, qs1d, den1d

  ! saved copies of the inputs for dumping
  REAL, DIMENSION(kts:kte) :: t0, q0, qc0, qr0, qi0, qs0, qg0, qh0, pii0

  REAL :: DT
  INTEGER :: k, case_id
  CHARACTER(LEN=32) :: arg

  ! ---- parse case id ----
  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 90.0   ! representative convection-permitting microphysics dt (s)

  ! ---- initialize WSM7 module constants (runs radar_init) ----
  CALL wsm7init(RHOAIR0, RHOWATER, DENS_SNOW, CLIQ, CPV, .FALSE.)

  ! ---- build the chosen column ----
  CALL build_column(case_id, t, q, qci, qrs, den, p, delz, pii)

  ! save inputs (the EXACT pre-call values the scheme receives).
  DO k = kts, kte
    t0(k)  = t(1,k)
    q0(k)  = q(1,k)
    qc0(k) = qci(1,k,1)
    qi0(k) = qci(1,k,2)
    qr0(k) = qrs(1,k,1)
    qs0(k) = qrs(1,k,2)
    qg0(k) = qrs(1,k,3)
    qh0(k) = qrs(1,k,4)
    pii0(k)= pii(1,k)
  END DO

  ! accumulators: rain/snow/graupel/hail are running totals (start at 0);
  ! the ncv + sr are zeroed inside wsm72D.
  rain=0.; rainncv=0.; sr=0.
  snow=0.; snowncv=0.; graupel=0.; graupelncv=0.; hail=0.; hailncv=0.

  ! ---- call the real WSM7 microphysics (wsm72D directly) ----
  CALL wsm72D(t, q, qci, qrs, den, p, delz                        &
             ,DT, G, CP, CPV, R_D, R_V, SVPT0                     &
             ,EP_1, EP_2, EPSILON                                 &
             ,XLS, XLV, XLF, RHOAIR0, RHOWATER                    &
             ,CLIQ, CICE, PSAT                                    &
             ,lat                                                 &
             ,rain, rainncv                                       &
             ,sr                                                  &
             ,its,ite, jts,jte, kts,kte                           &
             ,its,ite, jts,jte, kts,kte                           &
             ,its,ite, jts,jte, kts,kte                           &
             ,snow, snowncv                                       &
             ,graupel, graupelncv                                 &
             ,hail, hailncv                                       )

  ! ---- effective radii (radiation-consistent), uses POST-microphysics state ----
  re_qc = RE_QC_BG; re_qi = RE_QI_BG; re_qs = RE_QS_BG
  DO k = kts, kte
    t1d(k)  = t(1,k)
    qc1d(k) = qci(1,k,1)
    qi1d(k) = qci(1,k,2)
    qs1d(k) = qrs(1,k,2)
    den1d(k)= den(1,k)
  END DO
  CALL effectRad_wsm7(t1d, qc1d, qi1d, qs1d, den1d, EPSILON, SVPT0, &
                      re_qc, re_qi, re_qs, kts, kte, 1, 1)
  DO k = kts, kte
    re_qc(k) = MAX(RE_QC_BG, MIN(re_qc(k),  50.E-6))
    re_qi(k) = MAX(RE_QI_BG, MIN(re_qi(k), 125.E-6))
    re_qs(k) = MAX(RE_QS_BG, MIN(re_qs(k), 999.E-6))
  END DO

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  ! inputs
  CALL dump_col('T_IN',   t0)
  CALL dump_col('QV_IN',  q0)
  CALL dump_col('QC_IN',  qc0)
  CALL dump_col('QR_IN',  qr0)
  CALL dump_col('QI_IN',  qi0)
  CALL dump_col('QS_IN',  qs0)
  CALL dump_col('QG_IN',  qg0)
  CALL dump_col('QH_IN',  qh0)
  CALL dump_col('PII',    pii0)
  CALL dump_col2('DEN',   den)
  CALL dump_col2('P',     p)
  CALL dump_col2('DELZ',  delz)
  ! outputs (post-WSM7)
  CALL dump_col2('T_OUT',  t)
  CALL dump_col2('QV_OUT', q)
  CALL dump_qci('QC_OUT', qci, 1)
  CALL dump_qci('QI_OUT', qci, 2)
  CALL dump_col3('QR_OUT', qrs, 1)
  CALL dump_col3('QS_OUT', qrs, 2)
  CALL dump_col3('QG_OUT', qrs, 3)
  CALL dump_col3('QH_OUT', qrs, 4)
  CALL dump_col1('RE_CLOUD', re_qc)
  CALL dump_col1('RE_ICE',   re_qi)
  CALL dump_col1('RE_SNOW',  re_qs)
  WRITE(*,'(A,ES23.15)') 'RAIN=',       rain(1)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=',    rainncv(1)
  WRITE(*,'(A,ES23.15)') 'SNOW=',       snow(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNCV=',    snowncv(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPEL=',    graupel(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNCV=', graupelncv(1,1)
  WRITE(*,'(A,ES23.15)') 'HAIL=',       hail(1,1)
  WRITE(*,'(A,ES23.15)') 'HAILNCV=',    hailncv(1,1)
  WRITE(*,'(A,ES23.15)') 'SR=',         sr(1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col1(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col1

  SUBROUTINE dump_col2(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(its:ite,kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk)
    END DO
  END SUBROUTINE dump_col2

  SUBROUTINE dump_col3(name, arr, idx)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(its:ite,kts:kte,4), INTENT(IN) :: arr
    INTEGER, INTENT(IN) :: idx
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,idx)
    END DO
  END SUBROUTINE dump_col3

  SUBROUTINE dump_qci(name, arr, idx)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(its:ite,kts:kte,2), INTENT(IN) :: arr
    INTEGER, INTENT(IN) :: idx
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,idx)
    END DO
  END SUBROUTINE dump_qci

  ! ------------------------------------------------------------------
  ! Predeclared single columns. case_id:
  !  1 = warm moist boundary layer, supersaturated low levels
  !  2 = deep mixed-phase: warm below, ice/snow/graupel/hail aloft, melting
  !  3 = cold ice/snow column (all subfreezing), supersaturated wrt ice
  !  4 = hail/graupel-dominant convective core: large qr/qi/qg/qh + updraft
  !  5 = subsaturated mid-level with rain/snow/hail falling -> evap/sublim/melt
  !  6 = clean/near-zero hydrometeor column with slight supersaturation
  ! Hydrostatic pressure from a temperature/qv integration; bottom-up index
  ! (k=1 lowest model layer) matching WSM7 ordering.
  ! ------------------------------------------------------------------
  SUBROUTINE build_column(cid, tt, qq, qcii, qrss, dd, pp, dz, exner)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(its:ite,kts:kte), INTENT(OUT) :: tt, qq, dd, pp, dz, exner
    REAL, DIMENSION(its:ite,kts:kte,2), INTENT(OUT) :: qcii
    REAL, DIMENSION(its:ite,kts:kte,4), INTENT(OUT) :: qrss
    REAL :: psfc, tsfc, theta_sfc, ztop
    REAL :: th_k, t_k, p_k, tv_k, z_k, es, qsw, rh_k
    REAL :: lapse, rh_ml, rh_trop, zml
    REAL, DIMENSION(KX) :: zz
    INTEGER :: kk

    ztop = 16000.0
    DO kk = 1, KX
      zz(kk) = ztop * ( (REAL(kk)-0.5)/REAL(KX) )**1.15
    END DO

    SELECT CASE (cid)
    CASE (1)   ! warm moist BL, supersaturated low levels
      psfc=1000.0E2; tsfc=298.0; zml=1500.0; lapse=5.0E-3; rh_ml=1.02; rh_trop=0.40
    CASE (2)   ! deep mixed-phase with melting layer
      psfc=1000.0E2; tsfc=287.0; zml=600.0;  lapse=6.0E-3; rh_ml=0.98; rh_trop=0.60
    CASE (3)   ! cold ice/snow column, ice-supersaturated
      psfc=850.0E2;  tsfc=258.0; zml=400.0;  lapse=5.5E-3; rh_ml=1.05; rh_trop=0.70
    CASE (4)   ! hail/graupel-dominant convective core
      psfc=1000.0E2; tsfc=296.0; zml=1000.0; lapse=6.5E-3; rh_ml=1.00; rh_trop=0.65
    CASE (5)   ! subsaturated mid-level, falling rain/snow/hail
      psfc=950.0E2;  tsfc=283.0; zml=300.0;  lapse=6.0E-3; rh_ml=0.55; rh_trop=0.30
    CASE (6)   ! clean column, slight supersaturation
      psfc=1000.0E2; tsfc=295.0; zml=2000.0; lapse=5.0E-3; rh_ml=1.01; rh_trop=0.50
    CASE DEFAULT
      psfc=1000.0E2; tsfc=295.0; zml=1000.0; lapse=5.5E-3; rh_ml=0.95; rh_trop=0.50
    END SELECT

    theta_sfc = tsfc * (P1000/psfc)**ROVCP
    p_k = psfc
    DO kk = 1, KX
      z_k = zz(kk)
      IF (z_k <= zml) THEN
        th_k = theta_sfc
      ELSE
        th_k = theta_sfc + lapse*(z_k - zml)
      END IF
      IF (z_k <= zml) THEN
        rh_k = rh_ml
      ELSE
        rh_k = rh_trop + (rh_ml-rh_trop)*EXP(-(z_k-zml)/3000.0)
      END IF
      IF (kk == 1) THEN
        t_k  = th_k*(psfc/P1000)**ROVCP
        tv_k = t_k
        p_k  = psfc * EXP(-G*zz(1)/(R_D*tv_k))
      ELSE
        t_k  = th_k*(p_k/P1000)**ROVCP
        tv_k = t_k*(1.0+0.608*qq(1,kk-1))
        p_k  = p_k * EXP(-G*(zz(kk)-zz(kk-1))/(R_D*tv_k))
      END IF
      t_k = th_k*(p_k/P1000)**ROVCP
      es  = 610.78*EXP(17.27*(t_k-273.15)/(t_k-35.86))
      qsw = 0.622*es/(p_k-es)
      qq(1,kk)  = MAX(rh_k*qsw, 1.0E-8)
      tt(1,kk)  = t_k
      pp(1,kk)  = p_k
      tv_k = t_k*(1.0+0.608*qq(1,kk))
      dd(1,kk)  = p_k/(R_D*tv_k)
      exner(1,kk) = (p_k/P1000)**ROVCP
      IF (kk == 1) THEN
        dz(1,kk) = 2.0*zz(1)
      ELSE
        dz(1,kk) = zz(kk)-zz(kk-1)
      END IF
    END DO

    ! Seed hydrometeors per regime: qci(1)=cloud water, qci(2)=cloud ice;
    ! qrs(1)=rain, qrs(2)=snow, qrs(3)=graupel, qrs(4)=hail.
    qcii=0.; qrss=0.
    DO kk = 1, KX
      z_k = zz(kk)
      SELECT CASE (cid)
      CASE (1)   ! warm: cloud + a little rain
        IF (z_k < 3000.0)  qcii(1,kk,1) = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0)  qrss(1,kk,1) = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
      CASE (2)   ! mixed-phase: cloud/rain low, ice/snow/graupel/hail aloft
        IF (z_k < 4000.0)  qcii(1,kk,1) = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k < 5000.0)  qrss(1,kk,1) = 6.0E-4*EXP(-((z_k-1200.0)/1500.0)**2)
        IF (z_k > 4000.0)  qcii(1,kk,2) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 3500.0)  qrss(1,kk,2) = 8.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        IF (z_k > 3500.0)  qrss(1,kk,3) = 5.0E-4*EXP(-((z_k-5500.0)/2500.0)**2)
        IF (z_k > 4000.0)  qrss(1,kk,4) = 3.0E-4*EXP(-((z_k-6000.0)/2500.0)**2)
      CASE (3)   ! cold: ice + snow aloft, some graupel/hail
        qcii(1,kk,2) = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qrss(1,kk,2) = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
        IF (z_k > 7000.0) qrss(1,kk,3) = 1.0E-4*EXP(-((z_k-9000.0)/2500.0)**2)
        IF (z_k > 7000.0) qrss(1,kk,4) = 8.0E-5*EXP(-((z_k-9000.0)/2500.0)**2)
      CASE (4)   ! hail/graupel convective core: big qr,qi,qs,qg,qh + cloud
        IF (z_k < 5000.0)  qcii(1,kk,1) = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        IF (z_k < 6000.0)  qrss(1,kk,1) = 1.5E-3*EXP(-((z_k-2500.0)/2000.0)**2)
        IF (z_k > 4000.0)  qcii(1,kk,2) = 4.0E-4*EXP(-((z_k-7500.0)/2500.0)**2)
        qrss(1,kk,2) = 1.0E-3*EXP(-((z_k-6000.0)/3000.0)**2)
        qrss(1,kk,3) = 2.0E-3*EXP(-((z_k-6000.0)/2500.0)**2)
        qrss(1,kk,4) = 1.5E-3*EXP(-((z_k-6500.0)/2500.0)**2)
      CASE (5)   ! subsaturated: rain + snow + hail falling into dry air
        IF (z_k < 6000.0)  qrss(1,kk,1) = 7.0E-4*EXP(-((z_k-3000.0)/2000.0)**2)
        qrss(1,kk,2) = 6.0E-4*EXP(-((z_k-5000.0)/2500.0)**2)
        IF (z_k > 5000.0) qrss(1,kk,3) = 2.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 5000.0) qrss(1,kk,4) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
      CASE (6)   ! clean: trace cloud only -> condensation path
        IF (z_k < 3000.0)  qcii(1,kk,1) = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
      END SELECT
    END DO
  END SUBROUTINE build_column

END PROGRAM wsm7_oracle
