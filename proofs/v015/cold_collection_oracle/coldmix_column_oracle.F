! Standalone COLD MIXED-PHASE single-column Thompson oracle harness (WRF v4.7.1).
!
! Drives the REAL WRF Thompson microphysics (module_mp_thompson::mp_gt_driver)
! on a deliberately COLD mixed-phase column where SUPERCOOLED RAIN coexists with
! SNOW and GRAUPEL below 0 C across a deep layer -- the exact regime that
! activates the rain-collecting-snow (qr_acr_qs), rain-collecting-graupel
! (qr_acr_qg) and Bigg rain-freezing (freezeH2O) processes that the existing
! warm-biased precip oracle column never exercises.  This produces the cold
! savepoint the JAX Thompson cold-collection lane is validated against.
!
! The Switzerland 72h case is a JANUARY ALPINE case; its RAINNC bias (+5 mm)
! is consistent with a missing cold rain sink.  This column makes that sink
! lane observable in a single bit-exact WRF step.
!
! Build flags must match WRF's: -fconvert=big-endian -frecord-marker=4, real=4.
PROGRAM coldmix_column_oracle
  USE module_mp_thompson, ONLY : thompson_init, mp_gt_driver
  USE module_wrfgpu2_oracle, ONLY : oracle_open_scheme, oracle_close_scheme, &
                                    oracle_dump3d, oracle_dump2d
  IMPLICIT NONE

  INTEGER, PARAMETER :: NCOL = 8, NROW = 1, NLEV = 44
  INTEGER, PARAMETER :: ids=1, ide=NCOL+1, jds=1, jde=NROW+1, kds=1, kde=NLEV+1
  INTEGER, PARAMETER :: ims=1, ime=NCOL,   jms=1, jme=NROW,   kms=1, kme=NLEV
  INTEGER, PARAMETER :: its=1, ite=NCOL,   jts=1, jte=NROW,   kts=1, kte=NLEV

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: qv, qc, qr, qi, qs, qg, ni, nr, th
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: pii, p, w, dz8w
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: refl_10cm, re_cloud, re_ice, re_snow
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: hgt3d
  REAL, DIMENSION(ims:ime,jms:jme) :: RAINNC, RAINNCV, SNOWNC, SNOWNCV
  REAL, DIMENSION(ims:ime,jms:jme) :: GRAUPELNC, GRAUPELNCV, SR

  REAL, PARAMETER :: P0=100000.0, RD=287.04, CP=1004.5, G=9.81, RV=461.5
  REAL :: dt, zlev, tk, esat, qvs, frac, rh, pk, exn, tsfc
  INTEGER :: i, k, j, itimestep
  LOGICAL :: diagflag
  INTEGER :: has_reqc, has_reqi, has_reqs, do_radar_ref, ke_diag

  dt = 18.0
  itimestep = 1
  diagflag = .FALSE.
  do_radar_ref = 0
  ke_diag = NLEV
  has_reqc = 0; has_reqi = 0; has_reqs = 0

  ! ----- Build a COLD mixed-phase column -----
  ! Surface temperature near/below freezing (winter Alpine).  Supercooled rain
  ! and cloud water occupy a deep sub-freezing layer that OVERLAPS the snow and
  ! graupel layers, so rain-collects-snow / rain-collects-graupel / Bigg
  ! rain-freezing are all active.
  DO j = jts, jte
   DO i = its, ite
    zlev = 0.0
    ! per-column surface temperature 274 .. 270 K (straddles 0 C; colder cols
    ! push the whole supercooled-rain layer further below freezing)
    tsfc = 274.0 - 0.5*REAL(i-1)
    DO k = kts, kte
      p(i,k,j) = 101000.0 * EXP(-REAL(k-1)/14.0)
      pk = p(i,k,j)
      tk = tsfc - 0.0065 * zlev
      IF (tk < 212.0) tk = 212.0
      exn = (pk/P0)**(RD/CP)
      th(i,k,j) = tk / exn
      pii(i,k,j) = exn
      dz8w(i,k,j) = MIN(650.0, 30.0 + 620.0*(REAL(k-1)/REAL(NLEV-1))**1.3)
      IF (tk > 273.15) THEN
        esat = 611.2 * EXP(17.67*(tk-273.15)/(tk-29.65))
      ELSE
        esat = 611.2 * EXP(21.875*(tk-273.15)/(tk-7.66))
      END IF
      qvs = 0.622*esat/MAX(pk-esat, 1.0)
      frac = 0.55 + 0.05*REAL(i-1)
      IF (k <= 34) THEN
        rh = MIN(1.05, frac + 0.45)
      ELSE
        rh = 0.30
      END IF
      qv(i,k,j) = rh * qvs
      qc(i,k,j) = 0.0; qr(i,k,j) = 0.0; qi(i,k,j) = 0.0
      qs(i,k,j) = 0.0; qg(i,k,j) = 0.0
      ni(i,k,j) = 0.0; nr(i,k,j) = 0.0

      ! SUPERCOOLED rain + cloud water over a DEEP layer that reaches well below
      ! 0 C (k 3..30 spans the whole lower/mid troposphere; with tsfc<=274 K the
      ! bulk of this layer is sub-freezing).
      IF (k >= 3 .AND. k <= 30) THEN
        qc(i,k,j) = 5.0e-4
        qr(i,k,j) = 1.0e-3 * (0.6 + 0.1*REAL(i-1))   ! active (super)cooled rain
        nr(i,k,j) = 8.0e3
      END IF
      ! snow over a deep cold layer OVERLAPPING the rain layer
      IF (tk <= 273.15 .AND. k >= 8 .AND. k <= 34) THEN
        qs(i,k,j) = 8.0e-4 * (0.5 + 0.1*REAL(i-1))   ! active snow
      END IF
      ! graupel over the mid mixed-phase region OVERLAPPING the rain layer
      IF (tk <= 273.15 .AND. tk > 245.0 .AND. k >= 10 .AND. k <= 28) THEN
        qg(i,k,j) = 1.2e-3 * (0.5 + 0.1*REAL(i-1))   ! active graupel
      END IF
      ! cloud ice aloft
      IF (tk <= 250.0 .AND. k >= 24) THEN
        qi(i,k,j) = 2.0e-4
        ni(i,k,j) = 1.0e5
      END IF
      w(i,k,j) = 0.0
      refl_10cm(i,k,j) = 0.0
      re_cloud(i,k,j) = 0.0; re_ice(i,k,j) = 0.0; re_snow(i,k,j) = 0.0
      hgt3d(i,k,j) = zlev
      zlev = zlev + dz8w(i,k,j)
    END DO
   END DO
  END DO

  RAINNC = 0.0; RAINNCV = 0.0; SNOWNC = 0.0; SNOWNCV = 0.0
  GRAUPELNC = 0.0; GRAUPELNCV = 0.0; SR = 0.0

  CALL thompson_init(HGT=hgt3d,                                       &
       IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde,          &
       IMS=ims, IME=ime, JMS=jms, JME=jme, KMS=kms, KME=kme,          &
       ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte)

  CALL oracle_open_scheme('microphysics_coldmix','thompson','in', 1, itimestep, NCOL, NLEV, NROW)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','qv','kg kg-1','mass','water vapor mixing ratio', qv, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','qc','kg kg-1','mass','cloud water mixing ratio', qc, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','qr','kg kg-1','mass','rain mixing ratio', qr, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','qi','kg kg-1','mass','cloud ice mixing ratio', qi, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','qs','kg kg-1','mass','snow mixing ratio', qs, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','qg','kg kg-1','mass','graupel mixing ratio', qg, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','ni','kg-1','mass','cloud ice number conc', ni, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','nr','kg-1','mass','rain number conc', nr, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','th','K','mass','potential temperature', th, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','pii','-','mass','Exner function (pi)', pii, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','p','Pa','mass','pressure', p, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','w','m s-1','w','vertical velocity (full lev)', w, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','in','dz8w','m','mass','layer thickness dz', dz8w, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump2d('microphysics_coldmix','thompson','in','rainnc','mm','sfc','accum total precip', RAINNC, ims,ime,jms,jme, its,ite,jts,jte)
  CALL oracle_dump2d('microphysics_coldmix','thompson','in','snownc','mm','sfc','accum snow precip', SNOWNC, ims,ime,jms,jme, its,ite,jts,jte)
  CALL oracle_dump2d('microphysics_coldmix','thompson','in','graupelnc','mm','sfc','accum graupel precip', GRAUPELNC, ims,ime,jms,jme, its,ite,jts,jte)
  CALL oracle_close_scheme()

  CALL mp_gt_driver(                                  &
       QV=qv, QC=qc, QR=qr, QI=qi, QS=qs, QG=qg,      &
       NI=ni, NR=nr, TH=th, PII=pii, P=p, W=w, DZ=dz8w,&
       DT_IN=dt, ITIMESTEP=itimestep,                 &
       RAINNC=RAINNC, RAINNCV=RAINNCV,                &
       SNOWNC=SNOWNC, SNOWNCV=SNOWNCV,                &
       GRAUPELNC=GRAUPELNC, GRAUPELNCV=GRAUPELNCV, SR=SR, &
       refl_10cm=refl_10cm, diagflag=diagflag,        &
       ke_diag=ke_diag, do_radar_ref=do_radar_ref,    &
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, &
       has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs, &
       ids=ids,ide=ide, jds=jds,jde=jde, kds=kds,kde=kde, &
       ims=ims,ime=ime, jms=jms,jme=jme, kms=kms,kme=kme, &
       its=its,ite=ite, jts=jts,jte=jte, kts=kts,kte=kte)

  CALL oracle_open_scheme('microphysics_coldmix','thompson','out', 1, itimestep, NCOL, NLEV, NROW)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','qv','kg kg-1','mass','water vapor mixing ratio', qv, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','qc','kg kg-1','mass','cloud water mixing ratio', qc, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','qr','kg kg-1','mass','rain mixing ratio', qr, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','qi','kg kg-1','mass','cloud ice mixing ratio', qi, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','qs','kg kg-1','mass','snow mixing ratio', qs, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','qg','kg kg-1','mass','graupel mixing ratio', qg, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','ni','kg-1','mass','cloud ice number conc', ni, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','nr','kg-1','mass','rain number conc', nr, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','th','K','mass','potential temperature', th, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','pii','-','mass','Exner function (pi)', pii, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','p','Pa','mass','pressure', p, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump3d('microphysics_coldmix','thompson','out','dz8w','m','mass','layer thickness dz', dz8w, ims,ime,kms,kme,jms,jme, its,ite,kts,kte,jts,jte)
  CALL oracle_dump2d('microphysics_coldmix','thompson','out','rainnc','mm','sfc','accum total precip', RAINNC, ims,ime,jms,jme, its,ite,jts,jte)
  CALL oracle_dump2d('microphysics_coldmix','thompson','out','rainncv','mm','sfc','step total precip', RAINNCV, ims,ime,jms,jme, its,ite,jts,jte)
  CALL oracle_dump2d('microphysics_coldmix','thompson','out','snownc','mm','sfc','accum snow precip', SNOWNC, ims,ime,jms,jme, its,ite,jts,jte)
  CALL oracle_dump2d('microphysics_coldmix','thompson','out','graupelnc','mm','sfc','accum graupel precip', GRAUPELNC, ims,ime,jms,jme, its,ite,jts,jte)
  CALL oracle_dump2d('microphysics_coldmix','thompson','out','sr','-','sfc','snow ratio', SR, ims,ime,jms,jme, its,ite,jts,jte)
  CALL oracle_close_scheme()

  WRITE(*,'(A)') '=== COLDMIX COLUMN ORACLE: mp_gt_driver completed ==='
  WRITE(*,'(A,4(1X,E12.4))') 'max RAINNC SNOWNC GRAUPELNC RAINNCV:', &
       MAXVAL(RAINNC), MAXVAL(SNOWNC), MAXVAL(GRAUPELNC), MAXVAL(RAINNCV)
  DO i = its, ite
    WRITE(*,'(A,I2,A,4(1X,E12.4))') 'col ', i, ' RAINNC SNOWNC GRAUPELNC SR:', &
         RAINNC(i,1), SNOWNC(i,1), GRAUPELNC(i,1), SR(i,1)
  END DO
END PROGRAM coldmix_column_oracle
