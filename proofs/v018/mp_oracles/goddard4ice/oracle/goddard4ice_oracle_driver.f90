! =====================================================================
! v0.17 single-column Goddard GCE 4-ice (WRF mp_physics=7,
! GSFCGCE_4ICE_NUWRF) oracle driver.
!
! Drives the UNMODIFIED WRF phys/module_mp_gsfcgce_4ice_nuwrf.F scheme:
!   gsfcgce_4ice_nuwrf  -- the public entry. It INTERNALLY calls
!                          consat_s (constants init, every call -- there is
!                          NO separate init entry) and saticel_s (the
!                          saturation-adjustment + 4-ice microphysics core),
!                          and on the FIRST itimestep runs radar_init.
! on a prescribed single column and dumps the FULL input state
! (t,qv,qc,qr,qi,qs,qg,qh,pii,den,p,delz) and the FULL output state
! (t,qv,qc,qr,qi,qs,qg,qh after the call) plus surface accumulators
! (rainnc,rainncv,snownc,snowncv,graupelnc,graupelncv,hailnc,hailncv,sr)
! and the 6 categorical effective radii.
!
! This is the GOLD oracle for the future JAX mp=7 port: it is the real
! Fortran Goddard 4-ice scheme, not a re-implementation, so the JAX port
! cannot "self-compare". consat_s runs every call so every saved internal
! constant (gamma-function moments, fall-speed / slope prefactors for the
! FOUR ice species incl. the separate hail category) is exactly WRF.
!
! 4-ice prognostic set (WRF Registry gsfcgce_4ice):
!   moist: qv, qc(=ql cloud water), qr, qi, qs, qg, qh   (hail is its OWN
!   category -- this is the distinction vs the 3-ice mp=7 lin/gsfcgce that
!   has only graupel; here graupel AND hail co-exist).
!
! Usage: ./goddard4ice_oracle <case_id>   (1..6 regimes, see build_column)
! Output: flat key=value text dump on stdout (parsed by Python into JSON).
!
! Default REAL = REAL*4 (single precision); the JAX port runs fp64. Parity is
! therefore to a PREDECLARED physical tolerance, never bitwise. A
! -DDOUBLE_PRECISION build (-fdefault-real-8) provides the fp64 reference
! used for the categorical effective-radius diagnostics only.
! =====================================================================
PROGRAM goddard4ice_oracle
  USE module_mp_gsfcgce_4ice_nuwrf, ONLY : gsfcgce_4ice_nuwrf
  IMPLICIT NONE

  ! ---- WRF model constants, bound exactly as the microphysics driver does ----
  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: SVPT0  = 273.15
  REAL, PARAMETER :: P1000  = 1.0E5
  REAL, PARAMETER :: ROVCP  = R_D/CP
  REAL, PARAMETER :: RHOWATER = 1000.0
  REAL, PARAMETER :: RHOSNOW  = 100.0

  INTEGER, PARAMETER :: KX  = 40
  ! One ACTIVE interior column (i=j=1). Use a 2-wide MEMORY tile so any WRF
  ! init/debug probe stays in bounds and so no uninitialized memory is read;
  ! the Goddard loop bound is i = ii+ic-1 with ic<=min(CHUNK,ite-ii+1), i.e.
  ! strictly within [its,ite], so with its=ite=1 only column (1,*,1) is
  ! integrated. The inactive memory column (2,*,*) is zero-filled.
  INTEGER, PARAMETER :: KX1 = KX + 1
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX1
  INTEGER, PARAMETER :: ims=1, ime=2, jms=1, jme=2, kms=1, kme=KX
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  ! single column (1,KX,1) prognostic + forcing arrays (i,k,j layout)
  REAL, DIMENSION(ims:ime, kms:kme, jms:jme) :: &
        th, qv, ql, qr, qi, qs, qg, qh, &
        rho, pii, p, z, dz8w, w, refl_10cm
  ! latent-heating budget tracers (INTENT(INOUT)) + their accumulators
  REAL, DIMENSION(ims:ime, kms:kme, jms:jme) :: &
        physc, physe, physd, physs, physm, physf, &
        acphysc, acphyse, acphysd, acphyss, acphysm, acphysf
  ! 3D precip diagnostics (INTENT(INOUT))
  REAL, DIMENSION(ims:ime, kms:kme, jms:jme) :: &
        preci3d, precs3d, precg3d, prech3d, precr3d
  ! categorical effective radii (INTENT(INOUT))
  REAL, DIMENSION(ims:ime, kms:kme, jms:jme) :: &
        re_cloud_gsfc, re_rain_gsfc, re_ice_gsfc, &
        re_snow_gsfc, re_graupel_gsfc, re_hail_gsfc
  ! surface fields
  REAL, DIMENSION(ims:ime, jms:jme) :: &
        rainnc, rainncv, snownc, snowncv, sr, &
        graupelnc, graupelncv, hailnc, hailncv
  REAL, DIMENSION(ims:ime, jms:jme) :: xland, ht

  ! saved copies of the inputs for dumping (active column only)
  REAL, DIMENSION(kts:kte) :: t0, qv0, qc0, qr0, qi0, qs0, qg0, qh0
  REAL, DIMENSION(kts:kte) :: pii0, p0, dz0, rho0

  REAL :: DT, DX
  INTEGER :: k, case_id
  CHARACTER(LEN=32) :: arg

  ! ---- parse case id ----
  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 90.0      ! representative convection-permitting microphysics dt (s)
  DX = 3000.0    ! grid spacing (m); used by the conv/stra separation diag

  ! ---- build the chosen column (sets th, qv, ql, qr, qi, qs, qg, qh,
  !      rho, pii, p, z, dz8w, w) ----
  CALL build_column(case_id)

  ! land point: xland MUST be 1. (land) or 2. (water); the scheme aborts
  ! otherwise (auto_conversion path). ht = surface terrain height (m).
  xland = 1.0
  ht    = 0.0

  ! save inputs (the EXACT pre-call values the scheme receives).
  DO k = kts, kte
    t0(k)   = th(1,k,1)*pii(1,k,1)
    qv0(k)  = qv(1,k,1)
    qc0(k)  = ql(1,k,1)
    qr0(k)  = qr(1,k,1)
    qi0(k)  = qi(1,k,1)
    qs0(k)  = qs(1,k,1)
    qg0(k)  = qg(1,k,1)
    qh0(k)  = qh(1,k,1)
    pii0(k) = pii(1,k,1)
    p0(k)   = p(1,k,1)
    dz0(k)  = dz8w(1,k,1)
    rho0(k) = rho(1,k,1)
  END DO

  ! accumulators / diagnostics start at 0.
  rainnc=0.; rainncv=0.; snownc=0.; snowncv=0.; sr=0.
  graupelnc=0.; graupelncv=0.; hailnc=0.; hailncv=0.
  refl_10cm=0.
  physc=0.; physe=0.; physd=0.; physs=0.; physm=0.; physf=0.
  acphysc=0.; acphyse=0.; acphysd=0.; acphyss=0.; acphysm=0.; acphysf=0.
  preci3d=0.; precs3d=0.; precg3d=0.; prech3d=0.; precr3d=0.
  re_cloud_gsfc=0.; re_rain_gsfc=0.; re_ice_gsfc=0.
  re_snow_gsfc=0.; re_graupel_gsfc=0.; re_hail_gsfc=0.

  ! ---- call the real Goddard 4-ice microphysics ----
  ! Positional order EXACTLY per the entry (module line 170): th,qv,ql,qr,qi,
  ! qs,qh, then rho,pii,p,dt_in,z, ht,dz8w,grav,w, rhowater,rhosnow,
  ! itimestep,xland,dx, dims..., surface accumulators, refl/diag, hailnc/hailncv,
  ! f_qg,qg, the 6 phys + 6 acphys, the 6 effective radii, the 5 precip3d.
  ! f_qg=.TRUE. activates the graupel category; diagflag=.FALSE.,
  ! do_radar_ref=0 (no reflectivity diag this oracle).
  CALL gsfcgce_4ice_nuwrf( th, qv, ql, qr, qi, qs, qh,                    &
                  rho, pii, p, DT, z,                                     &
                  ht, dz8w, G, w,                                         &
                  RHOWATER, RHOSNOW,                                      &
                  1, xland, DX,                                           &
                  ids,ide, jds,jde, kds,kde,                             &
                  ims,ime, jms,jme, kms,kme,                             &
                  its,ite, jts,jte, kts,kte,                             &
                  rainnc, rainncv,                                       &
                  snownc, snowncv, sr,                                   &
                  graupelnc, graupelncv,                                 &
                  refl_10cm, .FALSE., 0,                                 &
                  hailnc, hailncv,                                       &
                  .TRUE., qg,                                            &
                  physc, physe, physd, physs, physm, physf,             &
                  acphysc, acphyse, acphysd, acphyss, acphysm, acphysf, &
                  re_cloud_gsfc, re_rain_gsfc, re_ice_gsfc,             &
                  re_snow_gsfc, re_graupel_gsfc, re_hail_gsfc,          &
                  preci3d, precs3d, precg3d, prech3d, precr3d )

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  ! inputs
  CALL dump_col('T_IN',  t0)
  CALL dump_col('QV_IN', qv0)
  CALL dump_col('QC_IN', qc0)
  CALL dump_col('QR_IN', qr0)
  CALL dump_col('QI_IN', qi0)
  CALL dump_col('QS_IN', qs0)
  CALL dump_col('QG_IN', qg0)
  CALL dump_col('QH_IN', qh0)
  CALL dump_col('PII',   pii0)
  CALL dump_col('DEN',   rho0)
  CALL dump_col('P',     p0)
  CALL dump_col('DELZ',  dz0)
  ! outputs (post-Goddard-4ice)
  CALL dump_3d('T_OUT',  th, .TRUE.)
  CALL dump_3d('QV_OUT', qv, .FALSE.)
  CALL dump_3d('QC_OUT', ql, .FALSE.)
  CALL dump_3d('QR_OUT', qr, .FALSE.)
  CALL dump_3d('QI_OUT', qi, .FALSE.)
  CALL dump_3d('QS_OUT', qs, .FALSE.)
  CALL dump_3d('QG_OUT', qg, .FALSE.)
  CALL dump_3d('QH_OUT', qh, .FALSE.)
  CALL dump_3d('RE_CLOUD',   re_cloud_gsfc,   .FALSE.)
  CALL dump_3d('RE_RAIN',    re_rain_gsfc,    .FALSE.)
  CALL dump_3d('RE_ICE',     re_ice_gsfc,     .FALSE.)
  CALL dump_3d('RE_SNOW',    re_snow_gsfc,    .FALSE.)
  CALL dump_3d('RE_GRAUPEL', re_graupel_gsfc, .FALSE.)
  CALL dump_3d('RE_HAIL',    re_hail_gsfc,    .FALSE.)
  WRITE(*,'(A,ES23.15)') 'RAINNC=',     rainnc(1,1)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=',    rainncv(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNC=',     snownc(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNCV=',    snowncv(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNC=',  graupelnc(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNCV=', graupelncv(1,1)
  WRITE(*,'(A,ES23.15)') 'HAILNC=',     hailnc(1,1)
  WRITE(*,'(A,ES23.15)') 'HAILNCV=',    hailncv(1,1)
  WRITE(*,'(A,ES23.15)') 'SR=',         sr(1,1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  ! Dump a (1,KX,1) field. If is_theta, convert to T via pii for the T_OUT key.
  SUBROUTINE dump_3d(name, arr, is_theta)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    LOGICAL, INTENT(IN) :: is_theta
    INTEGER :: kk
    REAL :: val
    DO kk = kts, kte
      IF (is_theta) THEN
        val = arr(1,kk,1)*pii(1,kk,1)
      ELSE
        val = arr(1,kk,1)
      END IF
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', val
    END DO
  END SUBROUTINE dump_3d

  ! ------------------------------------------------------------------
  ! Predeclared single columns. case_id (SAME regimes as the WSM7 / Thompson
  ! oracles so the hail families are exercised over comparable thermodynamics):
  !  1 = warm moist BL, supersaturated low levels
  !  2 = deep mixed-phase: warm below, ice/snow/graupel/hail aloft, melting
  !  3 = cold ice/snow column (all subfreezing), supersaturated wrt ice
  !  4 = hail/graupel-dominant convective core: large qr/qi/qs/qg/qh + updraft
  !  5 = subsaturated mid-level with rain/snow/graupel/hail falling
  !  6 = clean/near-zero hydrometeor column with slight supersaturation
  ! Bottom-up index (k=1 lowest model layer) matching Goddard ordering.
  ! ------------------------------------------------------------------
  SUBROUTINE build_column(cid)
    INTEGER, INTENT(IN) :: cid
    REAL :: psfc, tsfc, theta_sfc, ztop
    REAL :: th_k, t_k, p_k, tv_k, z_k, es, qsw, rh_k
    REAL :: lapse, rh_ml, rh_trop, zml, rho_k
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
    CASE (5)   ! subsaturated mid-level, falling rain/snow/graupel/hail
      psfc=950.0E2;  tsfc=283.0; zml=300.0;  lapse=6.0E-3; rh_ml=0.55; rh_trop=0.30
    CASE (6)   ! clean column, slight supersaturation
      psfc=1000.0E2; tsfc=295.0; zml=2000.0; lapse=5.0E-3; rh_ml=1.01; rh_trop=0.50
    CASE DEFAULT
      psfc=1000.0E2; tsfc=295.0; zml=1000.0; lapse=5.5E-3; rh_ml=0.95; rh_trop=0.50
    END SELECT

    ! zero everything first (whole arrays incl. the inactive memory column 2,
    ! so the scheme never reads uninitialized memory in the i=2 / j=2 planes).
    th=0.; qv=0.; ql=0.; qr=0.; qi=0.; qs=0.; qg=0.; qh=0.
    rho=0.; pii=1.; p=0.; z=0.; dz8w=1.; w=0.

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
      qv(1,kk,1)  = MAX(rh_k*qsw, 1.0E-8)
      pii(1,kk,1) = (p_k/P1000)**ROVCP
      th(1,kk,1)  = t_k/pii(1,kk,1)
      p(1,kk,1)   = p_k
      z(1,kk,1)   = z_k
      tv_k = t_k*(1.0+0.608*qv(1,kk,1))
      rho_k = p_k/(R_D*tv_k)
      rho(1,kk,1) = rho_k
      IF (kk == 1) THEN
        dz8w(1,kk,1) = 2.0*zz(1)
      ELSE
        dz8w(1,kk,1) = zz(kk)-zz(kk-1)
      END IF
    END DO

    ! Seed hydrometeors per regime: cloud water (ql), rain (qr), cloud ice
    ! (qi), snow (qs), graupel (qg), hail (qh). Case 4 = strong hail/graupel
    ! convective core.
    DO kk = 1, KX
      z_k = zz(kk)
      SELECT CASE (cid)
      CASE (1)   ! warm: cloud + a little rain
        IF (z_k < 3000.0)  ql(1,kk,1) = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0)  qr(1,kk,1) = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
      CASE (2)   ! mixed-phase: cloud/rain low, ice/snow/graupel/hail aloft
        IF (z_k < 4000.0)  ql(1,kk,1) = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k < 5000.0)  qr(1,kk,1) = 6.0E-4*EXP(-((z_k-1200.0)/1500.0)**2)
        IF (z_k > 4000.0)  qi(1,kk,1) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 3500.0)  qs(1,kk,1) = 8.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        IF (z_k > 3500.0)  qg(1,kk,1) = 5.0E-4*EXP(-((z_k-5500.0)/2500.0)**2)
        IF (z_k > 4000.0)  qh(1,kk,1) = 3.0E-4*EXP(-((z_k-6000.0)/2500.0)**2)
      CASE (3)   ! cold: ice + snow aloft, some graupel/hail
        qi(1,kk,1) = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qs(1,kk,1) = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
        IF (z_k > 7000.0) qg(1,kk,1) = 1.0E-4*EXP(-((z_k-9000.0)/2500.0)**2)
        IF (z_k > 7000.0) qh(1,kk,1) = 8.0E-5*EXP(-((z_k-9000.0)/2500.0)**2)
      CASE (4)   ! hail/graupel convective core: big qr,qi,qs,qg,qh + cloud
        IF (z_k < 5000.0)  ql(1,kk,1) = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        IF (z_k < 6000.0)  qr(1,kk,1) = 1.5E-3*EXP(-((z_k-2500.0)/2000.0)**2)
        IF (z_k > 4000.0)  qi(1,kk,1) = 4.0E-4*EXP(-((z_k-7500.0)/2500.0)**2)
        qs(1,kk,1) = 1.0E-3*EXP(-((z_k-6000.0)/3000.0)**2)
        qg(1,kk,1) = 2.0E-3*EXP(-((z_k-6000.0)/2500.0)**2)
        qh(1,kk,1) = 1.5E-3*EXP(-((z_k-6500.0)/2500.0)**2)
      CASE (5)   ! subsaturated: rain + snow + graupel + hail falling into dry air
        IF (z_k < 6000.0)  qr(1,kk,1) = 7.0E-4*EXP(-((z_k-3000.0)/2000.0)**2)
        qs(1,kk,1) = 6.0E-4*EXP(-((z_k-5000.0)/2500.0)**2)
        IF (z_k > 5000.0) qg(1,kk,1) = 2.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 5000.0) qh(1,kk,1) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
      CASE (6)   ! clean: trace cloud only -> condensation path
        IF (z_k < 3000.0)  ql(1,kk,1) = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
      END SELECT
    END DO
  END SUBROUTINE build_column

END PROGRAM goddard4ice_oracle
