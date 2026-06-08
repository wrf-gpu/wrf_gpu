! =====================================================================
! v0.13 single-column WDM5 (WRF mp_physics=14) oracle driver.
!
! Drives the UNMODIFIED pristine WRF module_mp_wdm5.F:
!   wdm5init     (initializes every saved module constant)
!   wdm52D       (the full double-moment 5-class microphysics, one column)
!   effectRad_wdm5 (effective radii for radiation)
! on a prescribed single column (idim=1) and dumps the FULL input state
! (t,qv,qc,qr,qi,qs,Nn,Nc,Nr,den,p,delz,pii) and the FULL output state
! (t,qv,qc,qr,qi,qs,Nn,Nc,Nr after the call) plus surface accumulators
! (rain,rainncv,snow,snowncv,sr) and effective radii (re_cloud,re_ice,re_snow).
!
! WDM5 = WSM5-style single-moment ICE/SNOW physics (NO graupel, NO hail) +
! DOUBLE-MOMENT warm rain. The precipitating array qrs packs:
! qrs(:,:,1)=qr (rain), qrs(:,:,2)=qs (snow). Cloud water/ice are qci(:,:,1)=qc,
! qci(:,:,2)=qi. The number-concentration array ncr packs ncr(:,:,1)=Nn (CCN),
! ncr(:,:,2)=Nc (cloud), ncr(:,:,3)=Nr (rain) -- the WDM6 layout. The 2D entry
! wdm52D works on t(its:ite,kts:kte), q/den/p/delz(ims:ime,kms:kme).
!
! This is the GOLD oracle for the JAX port: it is the real Fortran WDM5
! scheme (module_mp_wdm5.F), not a re-implementation, so the JAX port cannot
! "self-compare". The init is the real wdm5init so every saved module constant
! (gammas, slope maxima, fall-speed prefactors, pidnc/pidnr/qck1/qc0) is
! exactly WRF.
!
! NOTE on dependency stubs: module_mp_wdm5.F `use`s module_mp_radar (only for
! the reflectivity diagnostic path, NOT mass/number physics) and
! module_model_constants (only RE_QC_BG/RE_QI_BG/RE_QS_BG). We compile the REAL
! share/module_model_constants.F and a tiny no-op module_wrf_error + global
! wrf_debug (the morrison_stub_modules.f90 reused from the v0.6.0 oracle). The
! microphysics tendencies (mass + Nc/Nr/Nn) are produced by the UNMODIFIED
! scheme; only the radar-reflectivity diagnostic (out of scope; gated by
! diagflag) is stubbed.
!
! Usage: ./wdm5_oracle <case_id>
! Output: flat key=value text dump on stdout (parsed by Python into JSON).
!
! kind_phys: the classic WRF module is REAL*4 (single precision) by default;
! the fp64 build sets KP=8 via -DWDM5_FP64 + -fdefault-real-8. The JAX port
! runs fp64. Parity is therefore to a PREDECLARED physical tolerance, never
! bitwise (see run_wdm5_parity / t3_wdm5_oracle.py).
! =====================================================================
PROGRAM wdm5_oracle
  USE module_mp_wdm5, ONLY : wdm5init, wdm52D, effectRad_wdm5
  IMPLICIT NONE

#ifdef WDM5_FP64
  INTEGER, PARAMETER :: KP = 8
#else
  INTEGER, PARAMETER :: KP = 4
#endif

  ! ---- WRF model constants (share/module_model_constants.F), bound exactly
  !      as in module_microphysics_driver.F CALL wdm5(...) ----
  REAL(KP), PARAMETER :: G      = 9.81
  REAL(KP), PARAMETER :: R_D    = 287.0
  REAL(KP), PARAMETER :: CP     = 7.0*R_D/2.0     ! cpd
  REAL(KP), PARAMETER :: R_V    = 461.6
  REAL(KP), PARAMETER :: CPV    = 4.0*R_V
  REAL(KP), PARAMETER :: CLIQ   = 4190.0
  REAL(KP), PARAMETER :: CICE   = 2106.0
  REAL(KP), PARAMETER :: PSAT   = 610.78
  REAL(KP), PARAMETER :: XLV    = 2.5E6           ! xlv0
  REAL(KP), PARAMETER :: XLS    = 2.85E6
  REAL(KP), PARAMETER :: XLF    = 3.50E5          ! xlf0
  REAL(KP), PARAMETER :: SVPT0  = 273.15          ! t0c
  REAL(KP), PARAMETER :: EP_1   = R_V/R_D - 1.0
  REAL(KP), PARAMETER :: EP_2   = R_D/R_V
  REAL(KP), PARAMETER :: EPSILON= 1.E-15          ! qmin
  REAL(KP), PARAMETER :: RHOAIR0= 1.28            ! den0
  REAL(KP), PARAMETER :: RHOWATER=1000.0          ! denr
  REAL(KP), PARAMETER :: DENS_SNOW=100.0          ! dens (wdm5init arg)
  REAL(KP), PARAMETER :: CCN0   = 1.0E8           ! ccn_conc default
  REAL(KP), PARAMETER :: P1000  = 1.0E5
  REAL(KP), PARAMETER :: ROVCP  = R_D/CP
  REAL(KP), PARAMETER :: RE_QC_BG = 2.49E-6
  REAL(KP), PARAMETER :: RE_QI_BG = 4.99E-6
  REAL(KP), PARAMETER :: RE_QS_BG = 9.99E-6

  INTEGER, PARAMETER :: KX  = 40
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX
  INTEGER, PARAMETER :: ids=1, ide=1, jds=1, jde=1, kds=1, kde=KX
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX

  REAL(KP), DIMENSION(its:ite,kts:kte)   :: t
  REAL(KP), DIMENSION(its:ite,kts:kte,2) :: qci   ! 1=qc, 2=qi
  REAL(KP), DIMENSION(its:ite,kts:kte,2) :: qrs   ! 1=qr, 2=qs
  REAL(KP), DIMENSION(its:ite,kts:kte,3) :: ncr   ! 1=Nn, 2=Nc, 3=Nr
  REAL(KP), DIMENSION(ims:ime,kms:kme)   :: q, den, p, delz
  REAL(KP), DIMENSION(ims:ime)           :: rain, rainncv, sr
  REAL(KP), DIMENSION(ims:ime)           :: snow, snowncv

  REAL(KP), DIMENSION(kts:kte)           :: pii_col
  REAL(KP), DIMENSION(kts:kte)           :: re_qc, re_qi, re_qs
  REAL(KP), DIMENSION(kts:kte)           :: t1d, qc1d, nc1d, qi1d, qs1d, den1d

  ! saved copies of the inputs for dumping
  REAL(KP), DIMENSION(kts:kte) :: t0, q0, qc0, qr0, qi0, qs0
  REAL(KP), DIMENSION(kts:kte) :: nn0, nc0, nr0, pii0

  REAL(KP) :: DT
  INTEGER  :: k, case_id
  CHARACTER(LEN=32) :: arg

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 90.0   ! representative convection-permitting microphysics dt (s)

  ! ---- initialize WDM5 module constants ----
  CALL wdm5init(RHOAIR0, RHOWATER, DENS_SNOW, CLIQ, CPV, CCN0, .FALSE.)

  ! ---- build the chosen column ----
  CALL build_column(case_id, t, q, qci, qrs, ncr, den, p, delz, pii_col)

  ! save inputs (the EXACT pre-call values the scheme receives)
  DO k = kts, kte
    t0(k)  = t(1,k)
    q0(k)  = q(1,k)
    qc0(k) = qci(1,k,1); qi0(k) = qci(1,k,2)
    qr0(k) = qrs(1,k,1); qs0(k) = qrs(1,k,2)
    nn0(k) = ncr(1,k,1); nc0(k) = ncr(1,k,2); nr0(k) = ncr(1,k,3)
    pii0(k)= pii_col(k)
  END DO

  rain=0.; rainncv=0.; sr=0.
  snow=0.; snowncv=0.

  ! ---- call the real WDM5 microphysics (single column, single j) ----
  CALL wdm52D(t, q, qci, qrs, ncr, den, p, delz                  &
             ,DT, G, CP, CPV, CCN0, R_D, R_V, SVPT0              &
             ,EP_1, EP_2, EPSILON                                &
             ,XLS, XLV, XLF, RHOAIR0, RHOWATER                   &
             ,CLIQ, CICE, PSAT                                   &
             ,jts                                                &  ! lat
             ,rain, rainncv                                      &
             ,sr                                                 &
             ,ids,ide, jds,jde, kds,kde                          &
             ,ims,ime, jms,jme, kms,kme                          &
             ,its,ite, jts,jte, kts,kte                          &
             ,snow, snowncv                                       )

  ! ---- effective radii (uses POST-microphysics t,qc,nc,qi,qs) ----
  re_qc = RE_QC_BG; re_qi = RE_QI_BG; re_qs = RE_QS_BG
  DO k = kts, kte
    t1d(k)  = t(1,k)
    qc1d(k) = qci(1,k,1); qi1d(k) = qci(1,k,2); qs1d(k) = qrs(1,k,2)
    nc1d(k) = ncr(1,k,2); den1d(k)= den(1,k)
  END DO
  CALL effectRad_wdm5(t1d, qc1d, nc1d, qi1d, qs1d, den1d,    &
                      EPSILON, SVPT0, re_qc, re_qi, re_qs,   &
                      kts, kte, 1, 1)
  ! apply the driver's bounds (module_mp_wdm5.F wdm5 wrapper)
  DO k = kts, kte
    re_qc(k) = MAX(RE_QC_BG, MIN(re_qc(k),  50.E-6))
    re_qi(k) = MAX(RE_QI_BG, MIN(re_qi(k), 125.E-6))
    re_qs(k) = MAX(RE_QS_BG, MIN(re_qs(k), 999.E-6))
  END DO

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15E3)') 'DT=', DT
  CALL dump_col('T_IN',   t0)
  CALL dump_col('QV_IN',  q0)
  CALL dump_col('QC_IN',  qc0)
  CALL dump_col('QR_IN',  qr0)
  CALL dump_col('QI_IN',  qi0)
  CALL dump_col('QS_IN',  qs0)
  CALL dump_col('NN_IN',  nn0)
  CALL dump_col('NC_IN',  nc0)
  CALL dump_col('NR_IN',  nr0)
  CALL dump_col('PII',    pii0)
  CALL dump_col2('DEN',   den)
  CALL dump_col2('P',     p)
  CALL dump_col2('DELZ',  delz)
  CALL dump_col2('T_OUT',  t)
  CALL dump_col2q('QV_OUT', q)
  CALL dump_col2c('QC_OUT', qci, 1)
  CALL dump_col2c('QI_OUT', qci, 2)
  CALL dump_col2c('QR_OUT', qrs, 1)
  CALL dump_col2c('QS_OUT', qrs, 2)
  CALL dump_col3('NN_OUT', ncr, 1)
  CALL dump_col3('NC_OUT', ncr, 2)
  CALL dump_col3('NR_OUT', ncr, 3)
  CALL dump_col('RE_CLOUD', re_qc)
  CALL dump_col('RE_ICE',   re_qi)
  CALL dump_col('RE_SNOW',  re_qs)
  WRITE(*,'(A,ES23.15E3)') 'RAIN=',    rain(1)
  WRITE(*,'(A,ES23.15E3)') 'RAINNCV=', rainncv(1)
  WRITE(*,'(A,ES23.15E3)') 'SNOW=',    snow(1)
  WRITE(*,'(A,ES23.15E3)') 'SNOWNCV=', snowncv(1)
  WRITE(*,'(A,ES23.15E3)') 'SR=',      sr(1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(KP), DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15E3)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col2(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(KP), DIMENSION(its:ite,kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15E3)') name,'[',kk,']=', arr(1,kk)
    END DO
  END SUBROUTINE dump_col2

  SUBROUTINE dump_col2q(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(KP), DIMENSION(ims:ime,kms:kme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15E3)') name,'[',kk,']=', arr(1,kk)
    END DO
  END SUBROUTINE dump_col2q

  SUBROUTINE dump_col3(name, arr, m)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(KP), DIMENSION(its:ite,kts:kte,3), INTENT(IN) :: arr
    INTEGER, INTENT(IN) :: m
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15E3)') name,'[',kk,']=', arr(1,kk,m)
    END DO
  END SUBROUTINE dump_col3

  SUBROUTINE dump_col2c(name, arr, m)  ! for the 2-component qci/qrs arrays
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(KP), DIMENSION(its:ite,kts:kte,2), INTENT(IN) :: arr
    INTEGER, INTENT(IN) :: m
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15E3)') name,'[',kk,']=', arr(1,kk,m)
    END DO
  END SUBROUTINE dump_col2c

  ! ------------------------------------------------------------------
  ! Predeclared single columns. case_id:
  !  1 = warm moist BL, supersaturated low levels, dense cloud + rain
  !      -> double-moment warm-rain autoconversion(praut/nraut), accretion
  !         (pracw/nracw), self-collection (nccol/nrcol), warm sedimentation
  !  2 = deep mixed-phase with melting layer near surface
  !      -> snow melting (psmlt/nsmlt), accretion, sedimentation
  !  3 = cold ice/snow column (all subfreezing), ice-supersaturated
  !      -> ice nucleation/deposition (pigen/pidep/psdep), psaut
  !  4 = convective core: large qr/qi/qs + dense cloud
  !      -> riming (psacw/nsacw), psaci, freezing of rain to snow (psfrz/nsfrz)
  !  5 = subsaturated mid-level rain/snow falling into dry air
  !      -> evaporation (prevp<0 + Nrevp NR->NCCN), psevp, partial melt
  !  6 = clean column, slight supersaturation, weak cloud
  !      -> CCN activation (pcact/ncact), condensation (pcond), small-drop
  !         rain->cloud conversion (avedia<=di82)
  ! Hydrostatic pressure from a temperature/qv integration; bottom-up index
  ! (k=1 lowest model layer) matching WDM5 ordering. Nc/Nr seeded physically
  ! (Nc ~ 3e8/kg in cloud, Nr ~ 1e5/kg in rain), Nn (CCN) = ccn0.
  ! ------------------------------------------------------------------
  SUBROUTINE build_column(cid, tt, qq, qcip, qrsp, ncrp, dd, pp, dz, exner)
    INTEGER, INTENT(IN) :: cid
    REAL(KP), DIMENSION(its:ite,kts:kte),   INTENT(OUT) :: tt
    REAL(KP), DIMENSION(ims:ime,kms:kme),   INTENT(OUT) :: qq, dd, pp, dz
    REAL(KP), DIMENSION(its:ite,kts:kte,2), INTENT(OUT) :: qcip, qrsp
    REAL(KP), DIMENSION(its:ite,kts:kte,3), INTENT(OUT) :: ncrp
    REAL(KP), DIMENSION(kts:kte),           INTENT(OUT) :: exner
    REAL(KP) :: psfc, tsfc, theta_sfc, ztop
    REAL(KP) :: th_k, t_k, p_k, tv_k, z_k, es, qsw, rh_k
    REAL(KP) :: lapse, rh_ml, rh_trop, zml
    REAL(KP), DIMENSION(KX) :: zz
    REAL(KP) :: qc_k, qr_k
    INTEGER  :: kk

    ztop = 16000.0
    DO kk = 1, KX
      zz(kk) = ztop * ( (REAL(kk,KP)-0.5)/REAL(KX,KP) )**1.15
    END DO

    SELECT CASE (cid)
    CASE (1)   ! warm moist BL, supersaturated low levels
      psfc=1000.0E2; tsfc=298.0; zml=1500.0; lapse=5.0E-3; rh_ml=1.02; rh_trop=0.40
    CASE (2)   ! deep mixed-phase with melting layer
      psfc=1000.0E2; tsfc=287.0; zml=600.0;  lapse=6.0E-3; rh_ml=0.98; rh_trop=0.60
    CASE (3)   ! cold ice/snow column, ice-supersaturated
      psfc=850.0E2;  tsfc=258.0; zml=400.0;  lapse=5.5E-3; rh_ml=1.05; rh_trop=0.70
    CASE (4)   ! convective core
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
      es  = 610.78*EXP(17.27*(t_k-273.15)/(t_k-35.86))
      qsw = 0.622*es/(p_k-es)
      qq(1,kk)  = MAX(rh_k*qsw, 1.0E-8)
      tt(1,kk)  = t_k
      pp(1,kk)  = p_k
      tv_k = t_k*(1.0+0.608*qq(1,kk))
      dd(1,kk)  = p_k/(R_D*tv_k)
      exner(kk) = (p_k/P1000)**ROVCP
      IF (kk == 1) THEN
        dz(1,kk) = 2.0*zz(1)
      ELSE
        dz(1,kk) = zz(kk)-zz(kk-1)
      END IF
    END DO

    ! Seed hydrometeors + number concentrations per regime.
    qcip = 0.; qrsp = 0.
    ncrp(:,:,1) = CCN0   ! Nn (CCN background; wdm52D clamps to >=0)
    ncrp(:,:,2) = 0.     ! Nc
    ncrp(:,:,3) = 0.     ! Nr
    DO kk = 1, KX
      z_k = zz(kk)
      qc_k = 0.; qr_k = 0.
      SELECT CASE (cid)
      CASE (1)   ! warm: cloud + a little rain in the low cloud layer
        IF (z_k < 3000.0)  qc_k = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0)  qr_k = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
        qcip(1,kk,1) = qc_k; qrsp(1,kk,1) = qr_k
      CASE (2)   ! mixed-phase: cloud low, rain mid, ice/snow aloft
        IF (z_k < 4000.0)  qc_k = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k < 5000.0)  qr_k = 6.0E-4*EXP(-((z_k-1200.0)/1500.0)**2)
        qcip(1,kk,1) = qc_k; qrsp(1,kk,1) = qr_k
        IF (z_k > 4000.0)  qcip(1,kk,2) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 3500.0)  qrsp(1,kk,2) = 8.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
      CASE (3)   ! cold: ice + snow aloft, no warm rain
        qcip(1,kk,2) = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qrsp(1,kk,2) = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
      CASE (4)   ! convective core: big qr,qi,qs + cloud
        IF (z_k < 5000.0)  qc_k = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        IF (z_k < 6000.0)  qr_k = 1.5E-3*EXP(-((z_k-2500.0)/2000.0)**2)
        qcip(1,kk,1) = qc_k; qrsp(1,kk,1) = qr_k
        IF (z_k > 4000.0)  qcip(1,kk,2) = 4.0E-4*EXP(-((z_k-7500.0)/2500.0)**2)
        qrsp(1,kk,2) = 1.5E-3*EXP(-((z_k-6000.0)/3000.0)**2)
      CASE (5)   ! subsaturated: rain + snow falling into dry air
        IF (z_k < 6000.0)  qr_k = 7.0E-4*EXP(-((z_k-3000.0)/2000.0)**2)
        qrsp(1,kk,1) = qr_k
        qrsp(1,kk,2) = 6.0E-4*EXP(-((z_k-5000.0)/2500.0)**2)
      CASE (6)   ! clean: trace cloud only -> activation/condensation path
        IF (z_k < 3000.0)  qc_k = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
        qcip(1,kk,1) = qc_k
      END SELECT
      ! seed Nc where cloud exists (~3e8/kg maritime droplet conc), Nr where
      ! rain exists (~1e5/kg) so the number tendencies are genuinely exercised.
      IF (qcip(1,kk,1) > 1.0E-6) ncrp(1,kk,2) = 3.0E8
      IF (qrsp(1,kk,1) > 1.0E-6) ncrp(1,kk,3) = 1.0E5
    END DO
  END SUBROUTINE build_column

END PROGRAM wdm5_oracle
