! =====================================================================
! v0.17 single-column Ferrier "new Eta" (WRF mp_physics=95, etampnew /
! EGCP01) microphysics oracle driver.
!
! Drives the UNMODIFIED WRF phys/module_mp_etanew.F scheme:
!   ETANEWinit  (reads the real ETAMPNEW_DATA.expanded_rain lookup tables
!                + builds the saturation-vapour tables via GPVS, and the
!                ice-growth table via MY_GROWTH_RATES, exactly as WRF does
!                at model start) -> fills MP_RESTART_STATE / TBPVS_STATE.
!   ETAMP_NEW   (the WRF-facing Ferrier driver: converts qv to specific
!                humidity, reconstitutes CWM = qc+qr+qi+qs from the
!                separate species + F_ICE/F_RAIN, calls EGCP01DRV ->
!                EGCP01COLUMN, and decomposes CWM back to qc,qr,qs,qi via
!                the updated F_ICE/F_RAIN/F_RIMEF fractions).
! on a prescribed single column (idim=1) and dumps the FULL input state
! (t,qv,qc,qr,qi,qs,f_ice,f_rain,f_rimef,den,p,delz,pii) and the FULL output
! state (t,qv,qc,qr,qi,qs,f_ice,f_rain,f_rimef after the call) plus the
! surface accumulator RAINNC/RAINNCV and the snow ratio SR.
!
! This is the GOLD oracle for the JAX port: it is the real Fortran Ferrier
! scheme (module_mp_etanew.F) with the real lookup tables, not a
! re-implementation, so the JAX port cannot "self-compare".
!
! NOTE on the moist menu: WRF's mp=95 (etampnew) registry package carries
! moist species qv,qc,qr,qs (NOT a separate qi leaf -- cloud ice is folded
! into the lumped condensate via F_ICE). ETAMP_NEW takes qv,qc,qr,qs,qt and
! reconstructs ice internally as qi = CWM*F_ICE. We expose qi separately in
! the dump (the diagnosed ice content CWM*F_ICE) purely so the JAX port can
! be checked on the recovered ice field.
!
! Usage: ./ferrier_oracle <case_id>     (1..6 regimes, see build_column)
! Output: flat key=value text dump on stdout (parsed by Python into JSON).
!
! Default REAL = REAL*4 (single precision, canonical WRF). The JAX port runs
! fp64; parity is to a PREDECLARED physical tolerance, never bitwise. A
! -DDOUBLE_PRECISION build (-fdefault-real-8) provides an fp64 reference (the
! lookup-table file is also read in fp64 from ETAMPNEW_DATA.expanded_rain_DBL).
! =====================================================================
PROGRAM ferrier_oracle
  USE module_mp_etanew, ONLY : ETAMP_NEW, ETANEWinit
  IMPLICIT NONE

  ! ---- WRF model constants (share/module_model_constants.F), the same
  !      values module_microphysics_driver.F binds for ETAMP_NEW. Ferrier
  !      pulls its physical constants from PRIVATE module parameters in
  !      module_mp_etanew.F; the driver call needs DX/DY only (for the
  !      RHgrd / DTPH derivation in init, which we mirror here).
  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: SVPT0  = 273.15
  REAL, PARAMETER :: P1000  = 1.0E5
  REAL, PARAMETER :: ROVCP  = R_D/CP

  INTEGER, PARAMETER :: KX  = 40
  INTEGER, PARAMETER :: NXT = 7501           ! must match module NX
  INTEGER, PARAMETER :: MY_T2 = 35
  INTEGER, PARAMETER :: ids=1,ide=2, jds=1,jde=2, kds=1,kde=KX+1
  INTEGER, PARAMETER :: ims=1,ime=1, jms=1,jme=1, kms=1,kme=KX
  INTEGER, PARAMETER :: its=1,ite=1, jts=1,jte=1, kts=1,kte=KX

  ! single column (im=1) arrays, WRF (i,k,j) memory order
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: dz8w,rho_phy,p_phy,pi_phy,th_phy
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: qv,qt,qc,qr,qs
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: f_ice_phy,f_rain_phy,f_rimef_phy
  REAL, DIMENSION(ims:ime,jms:jme)         :: rainnc,rainncv,sr
  INTEGER, DIMENSION(ims:ime,jms:jme)      :: lowlyr

  ! mp_restart / saturation-table state (filled by ETANEWinit)
  REAL, DIMENSION(MY_T2+8) :: mp_restart_state
  REAL, DIMENSION(NXT)     :: tbpvs_state, tbpvs0_state

  ! saved input copies for dumping (k = bottom-up model index)
  REAL, DIMENSION(kts:kte) :: t0,qv0,qc0,qr0,qi0,qs0,fice0,frain0,frimef0
  REAL, DIMENSION(kts:kte) :: pii0,den0,p0,dz0

  REAL :: DT, DX, DY
  INTEGER :: k, case_id
  CHARACTER(LEN=32) :: arg

  ! ---- parse case id ----
  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 90.0
  DX = 4000.0
  DY = 4000.0

  ! ---- initialize Ferrier module constants + lookup tables ----
  !      (reads ETAMPNEW_DATA.expanded_rain from CWD)
  lowlyr = 1
  mp_restart_state = 0.0
  tbpvs_state = 0.0
  tbpvs0_state = 0.0
  CALL ETANEWinit(0.0, DT, DX, DY, lowlyr, .FALSE.,                      &
       f_ice_phy, f_rain_phy, f_rimef_phy,                              &
       mp_restart_state, tbpvs_state, tbpvs0_state,                     &
       .TRUE.,                                                          &
       ids,ide, jds,jde, kds,kde,                                       &
       ims,ime, jms,jme, kms,kme,                                       &
       its,ite, jts,jte, kts,kte)

  ! ---- build the chosen column (sets qv,qc,qr,qs + den/p/pii/dz + the
  !      F_ICE/F_RAIN/F_RIMEF fractions consistent with the seeded ice) ----
  CALL build_column(case_id)

  ! WRF carries CWM (qt) = qc + qr + qi + qs. ETAMP_NEW expects qt to hold
  ! the TOTAL condensate; qc/qr/qs hold the (cloud water)/(rain)/(snow) parts
  ! and ice is folded into qt via F_ICE. Set qt = qc+qr+qi+qs and let F_ICE/
  ! F_RAIN decompose it, exactly as solve_em builds CWM before the call.
  DO k = kts, kte
    qt(1,k,1) = qc(1,k,1) + qr(1,k,1) + qs(1,k,1) + qi0(k)
  END DO

  ! save inputs (pre-call values)
  DO k = kts, kte
    t0(k)    = th_phy(1,k,1)*pi_phy(1,k,1)
    qv0(k)   = qv(1,k,1)
    qc0(k)   = qc(1,k,1)
    qr0(k)   = qr(1,k,1)
    qi0(k)   = qi0(k)   ! seeded ice content, set in build_column (host-assoc)
    qs0(k)   = qs(1,k,1)
    fice0(k) = f_ice_phy(1,k,1)
    frain0(k)= f_rain_phy(1,k,1)
    frimef0(k)=f_rimef_phy(1,k,1)
    pii0(k)  = pi_phy(1,k,1)
    den0(k)  = rho_phy(1,k,1)
    p0(k)    = p_phy(1,k,1)
    dz0(k)   = dz8w(1,k,1)
  END DO

  rainnc=0.; rainncv=0.; sr=0.

  ! ---- call the real Ferrier microphysics ----
  CALL ETAMP_NEW(1, DT, DX, DY,                                         &
       dz8w, rho_phy, p_phy, pi_phy, th_phy, qv, qt,                    &
       lowlyr, sr,                                                      &
       f_ice_phy, f_rain_phy, f_rimef_phy,                             &
       qc, qr, qs,                                                      &
       mp_restart_state, tbpvs_state, tbpvs0_state,                    &
       rainnc, rainncv,                                                &
       ids,ide, jds,jde, kds,kde,                                       &
       ims,ime, jms,jme, kms,kme,                                       &
       its,ite, jts,jte, kts,kte)

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  ! inputs
  CALL dump_col('T_IN',      t0)
  CALL dump_col('QV_IN',     qv0)
  CALL dump_col('QC_IN',     qc0)
  CALL dump_col('QR_IN',     qr0)
  CALL dump_col('QI_IN',     qi0)
  CALL dump_col('QS_IN',     qs0)
  CALL dump_col('FICE_IN',   fice0)
  CALL dump_col('FRAIN_IN',  frain0)
  CALL dump_col('FRIMEF_IN', frimef0)
  CALL dump_col('PII',       pii0)
  CALL dump_col('DEN',       den0)
  CALL dump_col('P',         p0)
  CALL dump_col('DELZ',      dz0)
  ! outputs (post-Ferrier). qv is returned as MIXING RATIO (ETAMP_NEW converts
  ! back from specific humidity at the end), and the species are recovered from
  ! CWM via the updated F_ICE/F_RAIN.
  CALL dump3('T_OUT',      th_phy, pi_phy)   ! theta*pi
  CALL dump_col2('QV_OUT', qv)
  CALL dump_col2('QC_OUT', qc)
  CALL dump_col2('QR_OUT', qr)
  CALL dump_qi('QI_OUT',   qt, f_ice_phy)    ! recovered ice = CWM*F_ICE
  CALL dump_col2('QS_OUT', qs)
  CALL dump_col2('QT_OUT', qt)
  CALL dump_col2('FICE_OUT',   f_ice_phy)
  CALL dump_col2('FRAIN_OUT',  f_rain_phy)
  CALL dump_col2('FRIMEF_OUT', f_rimef_phy)
  WRITE(*,'(A,ES23.15)') 'RAINNC=',  rainnc(1,1)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=', rainncv(1,1)
  WRITE(*,'(A,ES23.15)') 'SR=',      sr(1,1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col2(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col2

  SUBROUTINE dump3(name, th, pii)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: th, pii
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', th(1,kk,1)*pii(1,kk,1)
    END DO
  END SUBROUTINE dump3

  SUBROUTINE dump_qi(name, cwm, fice)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: cwm, fice
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', cwm(1,kk,1)*fice(1,kk,1)
    END DO
  END SUBROUTINE dump_qi

  ! ------------------------------------------------------------------
  ! Predeclared single columns. case_id:
  !  1 = warm moist BL, supersaturated low levels (cloud + light rain)
  !  2 = deep mixed-phase: warm below, ice/snow aloft, melting layer
  !  3 = cold ice/snow column (all subfreezing), ice-supersaturated
  !  4 = convective core: large qr/qi/qs + updraft, riming
  !  5 = subsaturated mid-level with rain/snow falling -> evap/sublim/melt
  !  6 = clean/near-zero hydrometeor column with slight supersaturation
  ! Hydrostatic pressure from a temperature/qv integration; bottom-up index
  ! (k=1 lowest model layer). F_ICE/F_RAIN/F_RIMEF are seeded consistent with
  ! the partitioning (F_ICE = qi/(qi+qc+qr+qs lumped...), see below).
  ! ------------------------------------------------------------------
  SUBROUTINE build_column(cid)
    INTEGER, INTENT(IN) :: cid
    REAL :: psfc, tsfc, theta_sfc, ztop
    REAL :: th_k, t_k, p_k, tv_k, z_k, es, qsw, rh_k
    REAL :: lapse, rh_ml, rh_trop, zml
    REAL :: cwmk, qik, qck, qrk, qsk, qliq, qice
    REAL, DIMENSION(KX) :: zz
    REAL, DIMENSION(KX) :: qi_seed
    INTEGER :: kk

    ztop = 16000.0
    DO kk = 1, KX
      zz(kk) = ztop * ( (REAL(kk)-0.5)/REAL(KX) )**1.15
    END DO

    SELECT CASE (cid)
    CASE (1)
      psfc=1000.0E2; tsfc=298.0; zml=1500.0; lapse=5.0E-3; rh_ml=1.02; rh_trop=0.40
    CASE (2)
      psfc=1000.0E2; tsfc=287.0; zml=600.0;  lapse=6.0E-3; rh_ml=0.98; rh_trop=0.60
    CASE (3)
      psfc=850.0E2;  tsfc=258.0; zml=400.0;  lapse=5.5E-3; rh_ml=1.05; rh_trop=0.70
    CASE (4)
      psfc=1000.0E2; tsfc=296.0; zml=1000.0; lapse=6.5E-3; rh_ml=1.00; rh_trop=0.65
    CASE (5)
      psfc=950.0E2;  tsfc=283.0; zml=300.0;  lapse=6.0E-3; rh_ml=0.55; rh_trop=0.30
    CASE (6)
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
        tv_k = t_k*(1.0+0.608*qv(1,kk-1,1))
        p_k  = p_k * EXP(-G*(zz(kk)-zz(kk-1))/(R_D*tv_k))
      END IF
      t_k = th_k*(p_k/P1000)**ROVCP
      es  = 610.78*EXP(17.27*(t_k-273.15)/(t_k-35.86))
      qsw = 0.622*es/(p_k-es)
      qv(1,kk,1)     = MAX(rh_k*qsw, 1.0E-8)
      th_phy(1,kk,1) = th_k
      pi_phy(1,kk,1) = (p_k/P1000)**ROVCP
      p_phy(1,kk,1)  = p_k
      tv_k = t_k*(1.0+0.608*qv(1,kk,1))
      rho_phy(1,kk,1)= p_k/(R_D*tv_k)
      IF (kk == 1) THEN
        dz8w(1,kk,1) = 2.0*zz(1)
      ELSE
        dz8w(1,kk,1) = zz(kk)-zz(kk-1)
      END IF
    END DO

    ! Seed hydrometeors per regime: qck=cloud water, qrk=rain, qik=cloud ice,
    ! qsk=snow. Then build CWM and F_ICE/F_RAIN/F_RIMEF the WRF way.
    qc=0.; qr=0.; qs=0.; qi_seed=0.
    f_ice_phy=0.; f_rain_phy=0.; f_rimef_phy=1.
    DO kk = 1, KX
      z_k = zz(kk)
      qck=0.; qrk=0.; qik=0.; qsk=0.
      SELECT CASE (cid)
      CASE (1)
        IF (z_k < 3000.0)  qck = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0)  qrk = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
      CASE (2)
        IF (z_k < 4000.0)  qck = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k < 5000.0)  qrk = 6.0E-4*EXP(-((z_k-1200.0)/1500.0)**2)
        IF (z_k > 4000.0)  qik = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 3500.0)  qsk = 8.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
      CASE (3)
        qik = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qsk = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
      CASE (4)
        IF (z_k < 5000.0)  qck = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        IF (z_k < 6000.0)  qrk = 1.5E-3*EXP(-((z_k-2500.0)/2000.0)**2)
        IF (z_k > 4000.0)  qik = 4.0E-4*EXP(-((z_k-7500.0)/2500.0)**2)
        qsk = 1.0E-3*EXP(-((z_k-6000.0)/3000.0)**2)
      CASE (5)
        IF (z_k < 6000.0)  qrk = 7.0E-4*EXP(-((z_k-3000.0)/2000.0)**2)
        qsk = 6.0E-4*EXP(-((z_k-5000.0)/2500.0)**2)
      CASE (6)
        IF (z_k < 3000.0)  qck = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
      END SELECT

      ! Lump into the Ferrier representation. WRF defines (see ETAMP_NEW
      ! reconstitution): liquid = qc + qr, ice = qi + qs (snow IS ice here).
      ! CWM = qc+qr+qi+qs; F_ICE = (qi+qs)/CWM; F_RAIN = qr/(qc+qr).
      qliq = qck + qrk
      qice = qik + qsk
      cwmk = qliq + qice
      qc(1,kk,1) = qck
      qr(1,kk,1) = qrk
      qs(1,kk,1) = qsk
      qi_seed(kk)= qik
      IF (cwmk > 1.0E-12) THEN
        f_ice_phy(1,kk,1) = MIN(1.0, qice/cwmk)
      ELSE
        f_ice_phy(1,kk,1) = 0.0
      END IF
      IF (qliq > 1.0E-12) THEN
        f_rain_phy(1,kk,1) = MIN(1.0, qrk/qliq)
      ELSE
        f_rain_phy(1,kk,1) = 0.0
      END IF
      f_rimef_phy(1,kk,1) = 1.0
    END DO
    ! expose seeded ice through module-scope save array for the dump
    DO kk = 1, KX
      qi0(kk) = qi_seed(kk)
    END DO
  END SUBROUTINE build_column

END PROGRAM ferrier_oracle
