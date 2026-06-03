! =====================================================================
! v0.6.0 single-column Purdue-Lin (WRF mp_physics=2) oracle driver.
!
! Drives the UNMODIFIED WRF phys/module_mp_lin.F scheme:
!   lin_et_al(...)   -- the public entry (calls clphy1d + satadj internally)
! on a prescribed single column (its=ite=1, jts=jte=1) and dumps the FULL
! input state (th,qv,qc,qr,qi,qs,qg,rho,pii,p,z,dz8w) and the FULL output
! state (th,qv,qc,qr,qi,qs,qg after the call) plus the surface accumulators
! (RAINNCV, SNOWNCV, GRAUPELNCV, SR).
!
! This is the GOLD oracle for the JAX port: it is the real Fortran Lin
! scheme (module_mp_lin.F), not a re-implementation, so the JAX port cannot
! "self-compare". Graupel is active (F_QG=.true. => gindex=1), exactly as
! the WRF microphysics_driver invokes LINSCHEME for the operational suite.
!
! The model runs in default WRF REAL (single precision); the JAX port runs
! fp64. Parity is therefore to a PREDECLARED physical tolerance, never
! bitwise (see run_lin_parity.py).
!
! Usage: ./lin_oracle <case_id>
!   case_id selects one of the predeclared columns (see build_column).
! Output: flat key=value text dump on stdout (parsed by Python into JSON).
! =====================================================================
PROGRAM lin_oracle
  USE module_mp_lin, ONLY : lin_et_al
  IMPLICIT NONE

  ! ---- WRF model constants (share/module_model_constants.F), bound exactly
  !      as in module_microphysics_driver.F CALL lin_et_al(...) ----
  REAL, PARAMETER :: G        = 9.81
  REAL, PARAMETER :: R_D      = 287.0
  REAL, PARAMETER :: CP       = 7.0*R_D/2.0      ! cp
  REAL, PARAMETER :: R_V      = 461.6
  REAL, PARAMETER :: XLS      = 2.85E6
  REAL, PARAMETER :: XLV      = 2.5E6
  REAL, PARAMETER :: XLF      = 3.50E5
  REAL, PARAMETER :: RHOWATER = 1000.0
  REAL, PARAMETER :: RHOSNOW  = 100.0
  REAL, PARAMETER :: EP_2     = R_D/R_V
  REAL, PARAMETER :: SVP1     = 0.6112
  REAL, PARAMETER :: SVP2     = 17.67
  REAL, PARAMETER :: SVP3     = 29.65
  REAL, PARAMETER :: SVPT0    = 273.15
  REAL, PARAMETER :: P1000    = 1.0E5
  REAL, PARAMETER :: ROVCP    = R_D/CP

  INTEGER, PARAMETER :: KX  = 40
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  ! 3-D single column arrays (i,k,j)
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: th, qv, qc, qr, qi, qs, qg
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: rho, pii, p, z, dz8w
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: refl_10cm
  REAL, DIMENSION(ims:ime,jms:jme)         :: ht
  REAL, DIMENSION(ims:ime,jms:jme)         :: RAINNC, RAINNCV, SR
  REAL, DIMENSION(ims:ime,jms:jme)         :: SNOWNC, SNOWNCV
  REAL, DIMENSION(ims:ime,jms:jme)         :: GRAUPELNC, GRAUPELNCV

  ! saved copies of the inputs for dumping
  REAL, DIMENSION(kts:kte) :: th0, qv0, qc0, qr0, qi0, qs0, qg0
  REAL, DIMENSION(kts:kte) :: rho0, pii0, p0, z0, dz0

  REAL :: DT
  LOGICAL :: f_qg, f_qndrop
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

  f_qg = .true.       ! graupel active (operational LINSCHEME)
  f_qndrop = .false.  ! no prognostic droplet number

  ! ---- build the chosen column ----
  CALL build_column(case_id)

  ! save inputs (exact pre-call values fed to lin_et_al)
  DO k = kts, kte
    th0(k)  = th(1,k,1);  qv0(k)  = qv(1,k,1); qc0(k) = qc(1,k,1)
    qr0(k)  = qr(1,k,1);  qi0(k)  = qi(1,k,1); qs0(k) = qs(1,k,1)
    qg0(k)  = qg(1,k,1);  rho0(k) = rho(1,k,1); pii0(k)= pii(1,k,1)
    p0(k)   = p(1,k,1);   z0(k)   = z(1,k,1);  dz0(k) = dz8w(1,k,1)
  END DO

  ! accumulators initialized to zero
  RAINNC=0.; RAINNCV=0.; SR=0.
  SNOWNC=0.; SNOWNCV=0.; GRAUPELNC=0.; GRAUPELNCV=0.
  refl_10cm=0.

  ! ---- call the real Lin microphysics (no radar diag path) ----
  CALL lin_et_al(th=th, qv=qv, ql=qc, qr=qr, qi=qi, qs=qs,        &
                 rho=rho, pii=pii, p=p, dt_in=DT,                 &
                 z=z, ht=ht, dz8w=dz8w,                           &
                 grav=G, cp=CP, Rair=R_D, rvapor=R_V,             &
                 XLS=XLS, XLV=XLV, XLF=XLF,                       &
                 rhowater=RHOWATER, rhosnow=RHOSNOW,              &
                 EP2=EP_2, SVP1=SVP1, SVP2=SVP2, SVP3=SVP3,       &
                 SVPT0=SVPT0,                                     &
                 RAINNC=RAINNC, RAINNCV=RAINNCV,                  &
                 SNOWNC=SNOWNC, SNOWNCV=SNOWNCV,                  &
                 GRAUPELNC=GRAUPELNC, GRAUPELNCV=GRAUPELNCV,      &
                 SR=SR, refl_10cm=refl_10cm,                      &
                 ids=ids,ide=ide, jds=jds,jde=jde, kds=kds,kde=kde, &
                 ims=ims,ime=ime, jms=jms,jme=jme, kms=kms,kme=kme, &
                 its=its,ite=ite, jts=jts,jte=jte, kts=kts,kte=kte, &
                 F_QG=f_qg, F_QNDROP=f_qndrop, qg=qg)

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  ! inputs
  CALL dump_col('TH_IN',  th0)
  CALL dump_col('QV_IN',  qv0)
  CALL dump_col('QC_IN',  qc0)
  CALL dump_col('QR_IN',  qr0)
  CALL dump_col('QI_IN',  qi0)
  CALL dump_col('QS_IN',  qs0)
  CALL dump_col('QG_IN',  qg0)
  CALL dump_col('PII',    pii0)
  CALL dump_col('RHO',    rho0)
  CALL dump_col('P',      p0)
  CALL dump_col('Z',      z0)
  CALL dump_col('DZ8W',   dz0)
  ! outputs (post-Lin)
  CALL dump_col3('TH_OUT', th)
  CALL dump_col3('QV_OUT', qv)
  CALL dump_col3('QC_OUT', qc)
  CALL dump_col3('QR_OUT', qr)
  CALL dump_col3('QI_OUT', qi)
  CALL dump_col3('QS_OUT', qs)
  CALL dump_col3('QG_OUT', qg)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=',    RAINNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNCV=',    SNOWNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNCV=', GRAUPELNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'SR=',         SR(1,1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col3(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col3

  ! ------------------------------------------------------------------
  ! Predeclared single columns. case_id (mirrors the WSM6 oracle regimes):
  !  1 = warm moist boundary layer, supersaturated low levels
  !  2 = deep mixed-phase: warm below, ice/snow/graupel aloft, melting layer
  !  3 = cold ice/snow column (subfreezing), ice-supersaturated
  !  4 = graupel-dominant convective core: large qr/qi/qg
  !  5 = subsaturated mid-level with rain/snow falling -> evap/sublim
  !  6 = clean/near-zero hydrometeor column with slight supersaturation
  ! Hydrostatic pressure from a temperature/qv integration; bottom-up index
  ! (k=1 lowest model layer).
  ! ------------------------------------------------------------------
  SUBROUTINE build_column(cid)
    INTEGER, INTENT(IN) :: cid
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
    ht = 0.0
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
        tv_k = t_k*(1.0+0.608*qv(1,kk-1,1))
        p_k  = p_k * EXP(-G*(zz(kk)-zz(kk-1))/(R_D*tv_k))
      END IF
      t_k = th_k*(p_k/P1000)**ROVCP
      ! saturation vapor pressure (Tetens, liquid) for the RH target
      es  = 610.78*EXP(17.27*(t_k-273.15)/(t_k-35.86))
      qsw = 0.622*es/(p_k-es)
      qv(1,kk,1)  = MAX(rh_k*qsw, 1.0E-8)
      th(1,kk,1)  = th_k
      p(1,kk,1)   = p_k
      tv_k = t_k*(1.0+0.608*qv(1,kk,1))
      rho(1,kk,1) = p_k/(R_D*tv_k)
      pii(1,kk,1) = (p_k/P1000)**ROVCP
      z(1,kk,1)   = z_k
      IF (kk == 1) THEN
        dz8w(1,kk,1) = 2.0*zz(1)
      ELSE
        dz8w(1,kk,1) = zz(kk)-zz(kk-1)
      END IF
    END DO

    ! Seed hydrometeors per regime (same physically-plausible bumps as WSM6
    ! oracle so the relevant Lin process branches are exercised).
    qc=0.; qr=0.; qi=0.; qs=0.; qg=0.
    DO kk = 1, KX
      z_k = zz(kk)
      SELECT CASE (cid)
      CASE (1)
        IF (z_k < 3000.0)  qc(1,kk,1) = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0)  qr(1,kk,1) = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
      CASE (2)
        IF (z_k < 4000.0)  qc(1,kk,1) = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k < 5000.0)  qr(1,kk,1) = 6.0E-4*EXP(-((z_k-1200.0)/1500.0)**2)
        IF (z_k > 4000.0)  qi(1,kk,1) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 3500.0)  qs(1,kk,1) = 8.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        IF (z_k > 3500.0)  qg(1,kk,1) = 5.0E-4*EXP(-((z_k-5500.0)/2500.0)**2)
      CASE (3)
        qi(1,kk,1) = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qs(1,kk,1) = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
        IF (z_k > 7000.0) qg(1,kk,1) = 1.0E-4*EXP(-((z_k-9000.0)/2500.0)**2)
      CASE (4)
        IF (z_k < 5000.0)  qc(1,kk,1) = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        IF (z_k < 6000.0)  qr(1,kk,1) = 1.5E-3*EXP(-((z_k-2500.0)/2000.0)**2)
        IF (z_k > 4000.0)  qi(1,kk,1) = 4.0E-4*EXP(-((z_k-7500.0)/2500.0)**2)
        qs(1,kk,1) = 1.0E-3*EXP(-((z_k-6000.0)/3000.0)**2)
        qg(1,kk,1) = 2.0E-3*EXP(-((z_k-6000.0)/2500.0)**2)
      CASE (5)
        IF (z_k < 6000.0)  qr(1,kk,1) = 7.0E-4*EXP(-((z_k-3000.0)/2000.0)**2)
        qs(1,kk,1) = 6.0E-4*EXP(-((z_k-5000.0)/2500.0)**2)
        IF (z_k > 5000.0) qg(1,kk,1) = 2.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
      CASE (6)
        IF (z_k < 3000.0)  qc(1,kk,1) = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
      END SELECT
    END DO
  END SUBROUTINE build_column

END PROGRAM lin_oracle
