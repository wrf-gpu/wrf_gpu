! =====================================================================
! v0.6.0 single-column WSM6 (WRF mp_physics=6) oracle driver.
!
! Drives the UNMODIFIED WRF physics_mmm WSM6 scheme
!   mp_wsm6_init  (hail_opt=0, graupel mode = WRF default)
!   mp_wsm6_run   (the full 6-class single-moment microphysics)
!   mp_wsm6_effectRad_run  (effective radii for radiation)
! on a prescribed single column (idim=1) and dumps the FULL input state
! (t,qv,qc,qr,qi,qs,qg,den,p,delz,pii) and the FULL output state
! (t,qv,qc,qr,qi,qs,qg after the call) plus surface accumulators
! (rain,rainncv,snow,snowncv,graupel,graupelncv,sr) and effective radii
! (re_cloud,re_ice,re_snow).
!
! This is the GOLD oracle for the JAX port: it is the real Fortran WSM6
! scheme (mp_wsm6.F90), not a re-implementation, so the JAX port cannot
! "self-compare". The init is the real mp_wsm6_init so every saved module
! constant (gammas, slope maxima, fall-speed prefactors) is exactly WRF.
!
! Usage: ./wsm6_oracle <case_id>
!   case_id selects one of the predeclared columns (see build_column).
! Output: flat key=value text dump on stdout (parsed by Python into JSON).
!
! kind_phys = selected_real_kind(6) => the MMM physics runs in REAL*4
! (single precision); the JAX port runs fp64. Parity is therefore to a
! PREDECLARED physical tolerance, never bitwise (see run_wsm6_parity.py).
! =====================================================================
PROGRAM wsm6_oracle
  USE ccpp_kind_types, ONLY : kind_phys
  USE mp_wsm6,         ONLY : mp_wsm6_init, mp_wsm6_run
  USE mp_wsm6_effectrad, ONLY : mp_wsm6_effectRad_run
  IMPLICIT NONE

  ! ---- WRF model constants (share/module_model_constants.F), bound exactly
  !      as in module_microphysics_driver.F CALL wsm6(...) ----
  REAL(kind=kind_phys), PARAMETER :: G      = 9.81
  REAL(kind=kind_phys), PARAMETER :: R_D    = 287.0
  REAL(kind=kind_phys), PARAMETER :: CP     = 7.0*R_D/2.0     ! cpd
  REAL(kind=kind_phys), PARAMETER :: R_V    = 461.6
  REAL(kind=kind_phys), PARAMETER :: CPV    = 4.0*R_V
  REAL(kind=kind_phys), PARAMETER :: CLIQ   = 4190.0
  REAL(kind=kind_phys), PARAMETER :: CICE   = 2106.0
  REAL(kind=kind_phys), PARAMETER :: PSAT   = 610.78
  REAL(kind=kind_phys), PARAMETER :: XLV    = 2.5E6           ! xlv0
  REAL(kind=kind_phys), PARAMETER :: XLS    = 2.85E6
  REAL(kind=kind_phys), PARAMETER :: XLF    = 3.50E5          ! xlf0
  REAL(kind=kind_phys), PARAMETER :: SVPT0  = 273.15          ! t0c
  REAL(kind=kind_phys), PARAMETER :: EP_1   = R_V/R_D - 1.0
  REAL(kind=kind_phys), PARAMETER :: EP_2   = R_D/R_V
  REAL(kind=kind_phys), PARAMETER :: EPSILON= 1.E-15          ! qmin
  REAL(kind=kind_phys), PARAMETER :: RHOAIR0= 1.28            ! den0
  REAL(kind=kind_phys), PARAMETER :: RHOWATER=1000.0          ! denr
  REAL(kind=kind_phys), PARAMETER :: DENS_SNOW=100.0          ! dens (mp_wsm6_init arg)
  REAL(kind=kind_phys), PARAMETER :: P1000  = 1.0E5
  REAL(kind=kind_phys), PARAMETER :: ROVCP  = R_D/CP

  ! effective-radius background/max values (module_microphysics_driver.F defaults)
  REAL(kind=kind_phys), PARAMETER :: RE_QC_BG = 2.49E-6
  REAL(kind=kind_phys), PARAMETER :: RE_QI_BG = 4.99E-6
  REAL(kind=kind_phys), PARAMETER :: RE_QS_BG = 9.99E-6
  REAL(kind=kind_phys), PARAMETER :: RE_QC_MAX= 50.E-6
  REAL(kind=kind_phys), PARAMETER :: RE_QI_MAX= 125.E-6
  REAL(kind=kind_phys), PARAMETER :: RE_QS_MAX= 999.E-6

  INTEGER, PARAMETER :: KX  = 40
  INTEGER, PARAMETER :: its=1, ite=1, kts=1, kte=KX

  ! single column (im=1) arrays, shape (its:ite, kts:kte)
  REAL(kind=kind_phys), DIMENSION(its:ite,kts:kte) :: t, q, qc, qr, qi, qs, qg
  REAL(kind=kind_phys), DIMENSION(its:ite,kts:kte) :: den, p, delz, pii
  REAL(kind=kind_phys), DIMENSION(its:ite,kts:kte) :: re_qc, re_qi, re_qs
  REAL(kind=kind_phys), DIMENSION(its:ite) :: rain, rainncv, sr
  REAL(kind=kind_phys), DIMENSION(its:ite) :: snow, snowncv, graupel, graupelncv

  ! saved copies of the inputs for dumping
  REAL(kind=kind_phys), DIMENSION(kts:kte) :: t0, q0, qc0, qr0, qi0, qs0, qg0
  REAL(kind=kind_phys), DIMENSION(kts:kte) :: pii0

  REAL(kind=kind_phys) :: DT
  CHARACTER(LEN=512) :: errmsg
  INTEGER :: errflg
  INTEGER :: k, case_id
  CHARACTER(LEN=32) :: arg
  LOGICAL :: do_microp_re

  ! ---- parse case id ----
  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 90.0   ! representative convection-permitting microphysics dt (s)

  ! ---- initialize WSM6 module constants (hail_opt=0 => graupel mode) ----
  CALL mp_wsm6_init(RHOAIR0, RHOWATER, DENS_SNOW, CLIQ, CPV, 0, errmsg, errflg)
  IF (errflg /= 0) THEN
    WRITE(*,'(A)') 'INIT_FATAL: '//TRIM(errmsg)
    STOP 2
  END IF

  ! ---- build the chosen column (fills t,q,qc,qr,qi,qs,qg,den,p,delz,pii) ----
  CALL build_column(case_id, t, q, qc, qr, qi, qs, qg, den, p, delz, pii)

  ! save inputs (after WSM6's own internal max(.,0) padding does NOT alter inputs
  ! we feed; we dump the EXACT pre-call values the scheme receives).
  DO k = kts, kte
    t0(k)  = t(1,k);  q0(k)  = q(1,k);  qc0(k) = qc(1,k); qr0(k) = qr(1,k)
    qi0(k) = qi(1,k); qs0(k) = qs(1,k); qg0(k) = qg(1,k); pii0(k)= pii(1,k)
  END DO

  ! accumulators initialized to zero (rainncv/snowncv/graupelncv/sr are zeroed
  ! inside mp_wsm6_run; rain/snow/graupel are running totals, start at 0).
  rain=0.; rainncv=0.; sr=0.
  snow=0.; snowncv=0.; graupel=0.; graupelncv=0.

  ! ---- call the real WSM6 microphysics ----
  CALL mp_wsm6_run(t=t, q=q, qc=qc, qi=qi, qr=qr, qs=qs, qg=qg,            &
                   den=den, p=p, delz=delz, delt=DT, g=G, cpd=CP, cpv=CPV, &
                   rd=R_D, rv=R_V, t0c=SVPT0, ep1=EP_1, ep2=EP_2,          &
                   qmin=EPSILON, xls=XLS, xlv0=XLV, xlf0=XLF,              &
                   den0=RHOAIR0, denr=RHOWATER, cliq=CLIQ, cice=CICE,      &
                   psat=PSAT, rain=rain, rainncv=rainncv, sr=sr,           &
                   snow=snow, snowncv=snowncv, graupel=graupel,            &
                   graupelncv=graupelncv, its=its, ite=ite, kts=kts,       &
                   kte=kte, errmsg=errmsg, errflg=errflg)
  IF (errflg /= 0) THEN
    WRITE(*,'(A)') 'RUN_FATAL: '//TRIM(errmsg)
    STOP 3
  END IF

  ! ---- effective radii (radiation-consistent), uses POST-microphysics t,qc,qi,qs ----
  do_microp_re = .true.
  re_qc = RE_QC_BG; re_qi = RE_QI_BG; re_qs = RE_QS_BG
  CALL mp_wsm6_effectRad_run(do_microp_re, t, qc, qi, qs, den, EPSILON, SVPT0, &
                   RE_QC_BG, RE_QI_BG, RE_QS_BG, RE_QC_MAX, RE_QI_MAX, RE_QS_MAX, &
                   re_qc, re_qi, re_qs, its, ite, kts, kte, errmsg, errflg)

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
  CALL dump_col('PII',    pii0)
  CALL dump_col2('DEN',   den)
  CALL dump_col2('P',     p)
  CALL dump_col2('DELZ',  delz)
  ! outputs (post-WSM6)
  CALL dump_col2('T_OUT',  t)
  CALL dump_col2('QV_OUT', q)
  CALL dump_col2('QC_OUT', qc)
  CALL dump_col2('QR_OUT', qr)
  CALL dump_col2('QI_OUT', qi)
  CALL dump_col2('QS_OUT', qs)
  CALL dump_col2('QG_OUT', qg)
  CALL dump_col2('RE_CLOUD', re_qc)
  CALL dump_col2('RE_ICE',   re_qi)
  CALL dump_col2('RE_SNOW',  re_qs)
  WRITE(*,'(A,ES23.15)') 'RAIN=',       rain(1)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=',    rainncv(1)
  WRITE(*,'(A,ES23.15)') 'SNOW=',       snow(1)
  WRITE(*,'(A,ES23.15)') 'SNOWNCV=',    snowncv(1)
  WRITE(*,'(A,ES23.15)') 'GRAUPEL=',    graupel(1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNCV=', graupelncv(1)
  WRITE(*,'(A,ES23.15)') 'SR=',         sr(1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(kind=kind_phys), DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col2(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(kind=kind_phys), DIMENSION(its:ite,kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk)
    END DO
  END SUBROUTINE dump_col2

  ! ------------------------------------------------------------------
  ! Predeclared single columns. case_id:
  !  1 = warm moist boundary layer, supersaturated low levels
  !      -> cloud->rain autoconversion + accretion + warm-rain sedimentation
  !  2 = deep mixed-phase: warm moist below, ice/snow/graupel aloft, melting
  !      layer near surface -> melting (psmlt/pgmlt), accretion, sedimentation
  !  3 = cold ice/snow column (all subfreezing), supersaturated wrt ice
  !      -> ice nucleation/deposition, snow/graupel deposition, ice->snow aut
  !  4 = graupel-dominant convective core: large qr/qi/qg with strong updraft
  !      thermodynamics -> riming, piacr/pgacr, freezing of rain to graupel
  !  5 = subsaturated mid-level with rain/snow falling -> evaporation/sublim
  !      (prevp<0, psevp, pgevp), partial melting
  !  6 = clean/near-zero hydrometeor column with slight supersaturation
  !      -> pcond condensation only (warm), tests saturation adjustment path
  ! Hydrostatic pressure from a temperature/qv integration; bottom-up index
  ! (k=1 lowest model layer) matching WSM6 ordering.
  ! ------------------------------------------------------------------
  SUBROUTINE build_column(cid, tt, qq, qcc, qrr, qii, qss, qgg, dd, pp, dz, exner)
    INTEGER, INTENT(IN) :: cid
    REAL(kind=kind_phys), DIMENSION(its:ite,kts:kte), INTENT(OUT) :: &
         tt, qq, qcc, qrr, qii, qss, qgg, dd, pp, dz, exner
    REAL(kind=kind_phys) :: psfc, tsfc, theta_sfc, ztop
    REAL(kind=kind_phys) :: th_k, t_k, p_k, tv_k, z_k, es, qsw, rh_k
    REAL(kind=kind_phys) :: lapse, rh_ml, rh_trop, zml
    REAL(kind=kind_phys), DIMENSION(KX) :: zz
    INTEGER :: kk

    ztop = 16000.0
    DO kk = 1, KX
      zz(kk) = ztop * ( (REAL(kk,kind_phys)-0.5)/REAL(KX,kind_phys) )**1.15
    END DO

    SELECT CASE (cid)
    CASE (1)   ! warm moist BL, supersaturated low levels
      psfc=1000.0E2; tsfc=298.0; zml=1500.0; lapse=5.0E-3; rh_ml=1.02; rh_trop=0.40
    CASE (2)   ! deep mixed-phase with melting layer
      psfc=1000.0E2; tsfc=287.0; zml=600.0;  lapse=6.0E-3; rh_ml=0.98; rh_trop=0.60
    CASE (3)   ! cold ice/snow column, ice-supersaturated
      psfc=850.0E2;  tsfc=258.0; zml=400.0;  lapse=5.5E-3; rh_ml=1.05; rh_trop=0.70
    CASE (4)   ! graupel-dominant convective core
      psfc=1000.0E2; tsfc=296.0; zml=1000.0; lapse=6.5E-3; rh_ml=1.00; rh_trop=0.65
    CASE (5)   ! subsaturated mid-level, falling rain/snow
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
      ! saturation vapor pressure (Tetens, liquid) for the RH target
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

    ! Seed hydrometeors per regime. Profiles are physically plausible bumps so
    ! that the relevant WSM6 process branches are exercised.
    qcc=0.; qrr=0.; qii=0.; qss=0.; qgg=0.
    DO kk = 1, KX
      z_k = zz(kk)
      SELECT CASE (cid)
      CASE (1)   ! warm: cloud + a little rain in the low cloud layer
        IF (z_k < 3000.0)  qcc(1,kk) = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0)  qrr(1,kk) = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
      CASE (2)   ! mixed-phase: cloud low, rain mid, ice/snow/graupel aloft
        IF (z_k < 4000.0)  qcc(1,kk) = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k < 5000.0)  qrr(1,kk) = 6.0E-4*EXP(-((z_k-1200.0)/1500.0)**2)
        IF (z_k > 4000.0)  qii(1,kk) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 3500.0)  qss(1,kk) = 8.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        IF (z_k > 3500.0)  qgg(1,kk) = 5.0E-4*EXP(-((z_k-5500.0)/2500.0)**2)
      CASE (3)   ! cold: ice + snow aloft, no warm rain
        qii(1,kk) = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qss(1,kk) = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
        IF (z_k > 7000.0) qgg(1,kk) = 1.0E-4*EXP(-((z_k-9000.0)/2500.0)**2)
      CASE (4)   ! graupel-dominant convective core: big qr,qi,qs,qg + cloud
        IF (z_k < 5000.0)  qcc(1,kk) = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        IF (z_k < 6000.0)  qrr(1,kk) = 1.5E-3*EXP(-((z_k-2500.0)/2000.0)**2)
        IF (z_k > 4000.0)  qii(1,kk) = 4.0E-4*EXP(-((z_k-7500.0)/2500.0)**2)
        qss(1,kk) = 1.0E-3*EXP(-((z_k-6000.0)/3000.0)**2)
        qgg(1,kk) = 2.0E-3*EXP(-((z_k-6000.0)/2500.0)**2)
      CASE (5)   ! subsaturated: rain + snow falling into dry air
        IF (z_k < 6000.0)  qrr(1,kk) = 7.0E-4*EXP(-((z_k-3000.0)/2000.0)**2)
        qss(1,kk) = 6.0E-4*EXP(-((z_k-5000.0)/2500.0)**2)
        IF (z_k > 5000.0) qgg(1,kk) = 2.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
      CASE (6)   ! clean: trace hydrometeors only -> condensation path
        IF (z_k < 3000.0)  qcc(1,kk) = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
      END SELECT
    END DO
  END SUBROUTINE build_column

END PROGRAM wsm6_oracle
