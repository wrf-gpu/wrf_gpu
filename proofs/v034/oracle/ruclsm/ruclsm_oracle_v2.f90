PROGRAM ruclsm_oracle_v2
  !---------------------------------------------------------------------------
  ! v0.34 EXTENDED pristine-WRF single-column oracle for the RUC land-surface
  ! model (sf_surface_physics=3).  Calls the UNMODIFIED WRF driver subroutine
  ! LSMRUC (phys/module_sf_ruclsm.F) on a 1x1x1 grid, one column per regime,
  ! NSTEPS consecutive calls (ktau = 1..NSTEPS), and dumps the FULL INOUT/OUT
  ! state after EVERY call.  Template: proofs/v017/oracle/ruclsm/
  ! ruclsm_oracle_driver.f90 (regimes 1-5 keep the v017 inputs).
  !
  ! Module soil/veg tables are filled exactly like WRF lsminit, by the real
  ! RUCLSM_SOILVEGPARM('USGS-RUC','STAS-RUC') reading the pristine
  ! VEGPARM.TBL / SOILPARM.TBL / GENPARM.TBL copied into the build dir.
  ! Physical constants passed to LSMRUC are the WRF module_model_constants
  ! values that module_surface_driver passes (cp, rcp, g, xlv, stbolt).
  !
  ! Dump format (one value per line, parsed by ruclsm_dump_to_json_v2.py):
  !   NAME=value                 global scalars
  !   REGIME_NAME[r]=string      regime label
  !   IN_NAME[r]=value           regime input (state BEFORE the first call)
  !   IN_NAME[r][k]=value        regime input profile (k = soil level / category)
  !   NAME[r][s]=value           state after call s (s = 1..NSTEPS)
  !   NAME[r][s][k]=value        soil profile after call s (k = 1..NSL[r])
  ! Reals: ES24.16E3 (17 significant digits, 3-digit exponent, always 'E').
  ! Integers: I0.  The IN_ prefix disambiguates input profiles from step
  ! scalars (both carry two indices).
  !
  ! Optional arg 1 = precision label (echoed).  Optional arg 2 = 'skip:N' or
  ! 'only:N'.  The build script runs every regime in its OWN process
  ! ('only:N') so that a WRF fatal (wrf_error_fatal -> STOP 1) in one regime
  ! is recorded per regime and cannot truncate the others; a fresh process
  ! also prevents heap/stack carry-over between regimes.
  !---------------------------------------------------------------------------
  USE module_sf_ruclsm, ONLY : LSMRUC, RUCLSM_SOILVEGPARM, RUCLSMINIT, ifortbl
  USE module_model_constants, ONLY : cp, rcp, g, xlv, stbolt
  IMPLICIT NONE

  INTEGER, PARAMETER :: IMS=1, IME=1, JMS=1, JME=1, KMS=1, KME=1
  INTEGER, PARAMETER :: IDS=1, IDE=2, JDS=1, JDE=2, KDS=1, KDE=2
  INTEGER, PARAMETER :: ITS=1, ITE=1, JTS=1, JTE=1, KTS=1, KTE=1

  INTEGER, PARAMETER :: NLCAT  = 24     ! USGS land categories (landusef)
  INTEGER, PARAMETER :: NSCAT  = 19     ! STAS-RUC soil categories (soilctop)
  INTEGER, PARAMETER :: NREG   = 51     ! number of regimes
  INTEGER, PARAMETER :: NSTEPS = 12     ! LSMRUC calls per regime
  INTEGER, PARAMETER :: NSLMAX = 9
  INTEGER, PARAMETER :: ISWATER_C = 16, ISICE_C = 24
  REAL,    PARAMETER :: XICE_THRESHOLD_C = 0.5

  ! soil grids: v017 6-level (regimes 1-23, 27-30), WRF init_soil_depth_3
  ! 6-level (regime 31) and 9-level HRRR (regimes 24-26).
  REAL, DIMENSION(6), PARAMETER :: ZS6_V017 = (/ 0.0, 0.05, 0.20, 0.40, 1.0, 2.0 /)
  REAL, DIMENSION(6), PARAMETER :: ZS6_WRF  = (/ 0.0, 0.05, 0.20, 0.40, 1.60, 3.00 /)
  REAL, DIMENSION(9), PARAMETER :: ZS9_HRRR = (/ 0.0, 0.01, 0.04, 0.10, 0.30, 0.60, &
                                                 1.00, 1.60, 3.00 /)

  ! ---- regime configuration (filled by config_regime) ----
  CHARACTER(LEN=64) :: c_name
  INTEGER :: c_nsl, c_ivgtyp, c_isltyp, c_mosaic_lu, c_mosaic_soil
  INTEGER :: c_sh2o_mode            ! 0: sh2o=soilmois, smfr3d=0; 1: RUCLSMINIT
  LOGICAL :: c_frpcpn
  REAL    :: c_zs(NSLMAX)
  REAL    :: c_dt, c_xland, c_xice
  REAL    :: c_rainbl, c_rainncv, c_snowncv, c_graupelncv, c_frzfrac
  REAL    :: c_t3d, c_qv, c_qc, c_p8w, c_rho, c_z3d, c_glw, c_gsw, c_chs, c_tbot
  REAL    :: c_shdmin, c_shdmax, c_albbck, c_snoalb, c_alb, c_emiss, c_znt, c_z0
  REAL    :: c_lai, c_vegfra, c_canwat, c_mavail
  REAL    :: c_snow, c_snowh, c_snowc, c_soilt, c_soilt1, c_tsnav
  REAL    :: c_qvg, c_qsg, c_qcg, c_qsfc
  REAL    :: c_tso_top, c_tso_bot, c_sm_top, c_sm_bot
  REAL    :: c_landusef(NLCAT), c_soilctop(NSCAT)
  LOGICAL :: c_tsnav_set

  REAL, PARAMETER :: UNSET = -999.0

  INTEGER :: ireg, sel_mode, sel_reg, ios
  CHARACTER(LEN=32) :: precision_mode, selarg
  CHARACTER(LEN=4)  :: mminlu_c

  precision_mode = 'fp64'
  IF (COMMAND_ARGUMENT_COUNT() >= 1) CALL GET_COMMAND_ARGUMENT(1, precision_mode)
  sel_mode = 0; sel_reg = 0
  IF (COMMAND_ARGUMENT_COUNT() >= 2) THEN
     CALL GET_COMMAND_ARGUMENT(2, selarg)
     IF (selarg(1:5) == 'skip:') THEN
        sel_mode = 1; READ(selarg(6:), *, IOSTAT=ios) sel_reg
     ELSE IF (selarg(1:5) == 'only:') THEN
        sel_mode = 2; READ(selarg(6:), *, IOSTAT=ios) sel_reg
     END IF
  END IF
  mminlu_c = 'USGS'

  ! ---- Populate module soil/veg tables exactly like WRF lsminit ----
  CALL RUCLSM_SOILVEGPARM('USGS-RUC', 'STAS-RUC')

  ! ---- global header ----
  WRITE(*,'(A)')    'CASE=ruclsm_v2'
  WRITE(*,'(A,A)')  'PRECISION_MODE=', TRIM(precision_mode)
  CALL wgi('REAL_KIND', KIND(1.0))
  CALL wgi('NREG', NREG)
  CALL wgi('NSTEPS', NSTEPS)
  CALL wgi('NLCAT', NLCAT)
  CALL wgi('NSCAT', NSCAT)
  CALL wgi('ISWATER', ISWATER_C)
  CALL wgi('ISICE', ISICE_C)
  CALL wgi('FULL_WRF_EXE', 0)
  WRITE(*,'(A)')    'MMINLU=USGS'
  WRITE(*,'(A)')    'MMINLURUC=USGS-RUC'
  WRITE(*,'(A)')    'MMINSL=STAS-RUC'
  CALL wgr('CONST_CP', cp)
  CALL wgr('CONST_ROVCP', rcp)
  CALL wgr('CONST_G0', g)
  CALL wgr('CONST_LV', xlv)
  CALL wgr('CONST_STBOLT', stbolt)
  CALL wgr('CONST_XICE_THRESHOLD', XICE_THRESHOLD_C)

  DO ireg = 1, NREG
     IF (sel_mode == 1 .AND. ireg == sel_reg) CYCLE
     IF (sel_mode == 2 .AND. ireg /= sel_reg) CYCLE
     CALL config_regime(ireg)
     CALL run_regime(ireg)
  END DO

CONTAINS

  !==========================================================================
  SUBROUTINE run_regime(rid)
    INTEGER, INTENT(IN) :: rid

    REAL    :: dt
    INTEGER :: ktau, nsl, istep, k, kk, iforest, nroot, ioob
    LOGICAL :: myj, frpcpn, rdlai2d
    INTEGER :: spp_lsm, mosaic_lu, mosaic_soil, iswater, isice, lakemodel

    REAL, DIMENSION(IMS:IME, KMS:KME, JMS:JME) :: pattern_spp_lsm, field_sf
    REAL, DIMENSION(IMS:IME, KMS:KME, JMS:JME) :: z3d, p8w, t3d, qv3d, qc3d, rho3d
    REAL, DIMENSION(IMS:IME, JMS:JME) :: rainbl, snow, snowh, snowc
    REAL, DIMENSION(IMS:IME, JMS:JME) :: frzfrac, rhosnf, precipfr
    REAL, DIMENSION(IMS:IME, JMS:JME) :: graupelncv, snowncv, rainncv
    REAL, DIMENSION(IMS:IME, JMS:JME) :: glw, gsw, emiss, chklowq, chs
    REAL, DIMENSION(IMS:IME, JMS:JME) :: flqc, flhc, mavail, canwat, vegfra
    REAL, DIMENSION(IMS:IME, JMS:JME) :: alb, znt, z0, snoalb, albbck, lai
    REAL, DIMENSION(IMS:IME, JMS:JME) :: qsfc, qsg, qvg, qcg, dew, soilt1, tsnav
    REAL, DIMENSION(IMS:IME, JMS:JME) :: tbot, xland, xice
    REAL, DIMENSION(IMS:IME, JMS:JME) :: smavail, smmax, soilt, hfx, qfx, lh
    REAL, DIMENSION(IMS:IME, JMS:JME) :: sfcrunoff, udrunoff, acrunoff, sfcexc
    REAL, DIMENSION(IMS:IME, JMS:JME) :: sfcevp, grdflx, snowfallac, acsnow, snom
    REAL, DIMENSION(IMS:IME, JMS:JME) :: shdmin, shdmax, lakemask
    INTEGER, DIMENSION(IMS:IME, JMS:JME) :: ivgtyp, isltyp
    REAL, DIMENSION(IMS:IME, NLCAT, JMS:JME) :: landusef
    REAL, DIMENSION(IMS:IME, NSCAT, JMS:JME) :: soilctop
    REAL, ALLOCATABLE, DIMENSION(:,:,:) :: soilmois, sh2o, tso, smfr3d, keepfr3dflag
    REAL, ALLOCATABLE, DIMENSION(:) :: zs_loc

    nsl = c_nsl
    ALLOCATE(soilmois(IMS:IME,1:nsl,JMS:JME), sh2o(IMS:IME,1:nsl,JMS:JME),   &
             tso(IMS:IME,1:nsl,JMS:JME), smfr3d(IMS:IME,1:nsl,JMS:JME),      &
             keepfr3dflag(IMS:IME,1:nsl,JMS:JME), zs_loc(1:nsl))

    ! -------- scalar / flag setup (single rank, EM_CORE, no SPP) --------
    dt          = c_dt
    myj         = .FALSE.
    rdlai2d     = .FALSE.
    frpcpn      = c_frpcpn
    spp_lsm     = 0
    mosaic_lu   = c_mosaic_lu
    mosaic_soil = c_mosaic_soil
    iswater     = ISWATER_C
    isice       = ISICE_C
    lakemodel   = 0
    pattern_spp_lsm = 0.0
    field_sf        = 0.0
    lakemask        = 0.0

    ! -------- atmospheric forcing (single level, constant over the steps) ----
    z3d  (1,KMS,1) = c_z3d
    rho3d(1,KMS,1) = c_rho
    t3d  (1,KMS,1) = c_t3d
    qv3d (1,KMS,1) = c_qv
    qc3d (1,KMS,1) = c_qc
    p8w  (1,KMS,1) = c_p8w
    rainbl(1,1)     = c_rainbl
    rainncv(1,1)    = c_rainncv
    snowncv(1,1)    = c_snowncv
    graupelncv(1,1) = c_graupelncv
    frzfrac(1,1)    = c_frzfrac
    glw(1,1)        = c_glw
    gsw(1,1)        = c_gsw
    chs(1,1)        = c_chs
    tbot(1,1)       = c_tbot
    xland(1,1)      = c_xland
    xice(1,1)       = c_xice
    shdmin(1,1)     = c_shdmin
    shdmax(1,1)     = c_shdmax
    albbck(1,1)     = c_albbck
    ivgtyp(1,1)     = c_ivgtyp
    isltyp(1,1)     = c_isltyp

    ! -------- initial surface state --------
    snow(1,1)   = c_snow
    snowh(1,1)  = c_snowh
    snowc(1,1)  = c_snowc
    emiss(1,1)  = c_emiss
    mavail(1,1) = c_mavail
    canwat(1,1) = c_canwat
    vegfra(1,1) = c_vegfra
    alb(1,1)    = c_alb
    snoalb(1,1) = c_snoalb
    znt(1,1)    = c_znt
    z0(1,1)     = c_z0
    lai(1,1)    = c_lai
    soilt(1,1)  = c_soilt
    soilt1(1,1) = c_soilt1
    tsnav(1,1)  = c_tsnav
    qvg(1,1)    = c_qvg
    qsg(1,1)    = c_qsg
    qcg(1,1)    = c_qcg
    qsfc(1,1)   = c_qsfc
    dew(1,1)    = 0.0

    ! -------- initial soil column: linear in depth between top and bottom ----
    DO k = 1, nsl
       zs_loc(k)       = c_zs(k)
       tso(1,k,1)      = c_tso_top + (c_tso_bot - c_tso_top) * (c_zs(k) / c_zs(nsl))
       soilmois(1,k,1) = c_sm_top  + (c_sm_bot  - c_sm_top)  * (c_zs(k) / c_zs(nsl))
       sh2o(1,k,1)     = soilmois(1,k,1)
       smfr3d(1,k,1)   = 0.0
       keepfr3dflag(1,k,1) = 0.0
    END DO

    DO k = 1, NLCAT
       landusef(1,k,1) = c_landusef(k)
    END DO
    DO k = 1, NSCAT
       soilctop(1,k,1) = c_soilctop(k)
    END DO

    ! -------- optional WRF cold-start soil-ice partition (lsminit path) -----
    IF (c_sh2o_mode == 1) THEN
       CALL RUCLSMINIT(sh2o, smfr3d, tso, soilmois, isltyp, ivgtyp,          &
                       mminlu_c, xice, mavail, nsl, iswater, isice,          &
                       znt, .FALSE., .FALSE.,                                 &
                       IDS,IDE, JDS,JDE, KDS,KDE,                             &
                       IMS,IME, JMS,JME, KMS,KME,                             &
                       ITS,ITE, JTS,JTE, KTS,KTE)
    END IF

    ! exchange coefficients from the INITIAL mavail (non-myj path:
    ! qkms = flqc/rho/mavail, tkms = flhc/rho/(cp*(1+0.84*qv)))
    flqc(1,1) = c_chs * c_rho * mavail(1,1)
    flhc(1,1) = c_chs * c_rho * cp

    ! diagnostics / accumulators (initial values; most re-zeroed at ktau=1)
    smavail=0.; smmax=0.; sfcrunoff=0.; udrunoff=0.; acrunoff=0.
    sfcexc=0.; sfcevp=0.; grdflx=0.; snowfallac=0.; acsnow=0.; snom=0.
    chklowq=1.; hfx=0.; qfx=0.; lh=0.; rhosnf=-1.e3; precipfr=0.

    ! -------- derived (driver-side mirror of LSMRUC nroot logic) ----------
    iforest = ifortbl(c_ivgtyp)
    nroot = 4
    IF (iforest > 2) THEN
       DO k = 2, nsl
          IF (zs_loc(k) >= 0.4) THEN
             nroot = k; EXIT
          END IF
       END DO
    ELSE
       DO k = 2, nsl
          IF (zs_loc(k) >= 1.1) THEN
             nroot = k; EXIT
          END IF
       END DO
    END IF
    ioob = 0
    IF (nroot + 1 > nsl) ioob = 1

    ! -------- dump inputs --------
    WRITE(*,'(A,I0,A,A)') 'REGIME_NAME[', rid, ']=', TRIM(c_name)
    CALL wini('NSL', rid, nsl)
    DO k = 1, nsl
       CALL winp('ZS', rid, k, zs_loc(k))
    END DO
    CALL winr('DT', rid, dt)
    CALL wini('KTAU_START', rid, 1)
    CALL wini('FRPCPN', rid, MERGE(1, 0, frpcpn))
    CALL wini('MYJ', rid, 0)
    CALL wini('RDLAI2D', rid, 0)
    CALL wini('SPP_LSM', rid, spp_lsm)
    CALL wini('LAKEMODEL', rid, lakemodel)
    CALL winr('LAKEMASK', rid, lakemask(1,1))
    CALL wini('MOSAIC_LU', rid, mosaic_lu)
    CALL wini('MOSAIC_SOIL', rid, mosaic_soil)
    CALL wini('IVGTYP', rid, ivgtyp(1,1))
    CALL wini('ISLTYP', rid, isltyp(1,1))
    CALL wini('ISWATER', rid, iswater)
    CALL wini('ISICE', rid, isice)
    CALL wini('SH2O_INIT_MODE', rid, c_sh2o_mode)
    CALL wini('IFORTBL_DERIVED', rid, iforest)
    CALL wini('NROOT_DERIVED', rid, nroot)
    CALL wini('ZSHALF_NROOT1_OOB', rid, ioob)
    CALL winr('XLAND', rid, xland(1,1))
    CALL winr('XICE', rid, xice(1,1))
    CALL winr('XICE_THRESHOLD', rid, XICE_THRESHOLD_C)
    CALL winr('RAINBL', rid, rainbl(1,1))
    CALL winr('RAINNCV', rid, rainncv(1,1))
    CALL winr('SNOWNCV', rid, snowncv(1,1))
    CALL winr('GRAUPELNCV', rid, graupelncv(1,1))
    CALL winr('FRZFRAC', rid, frzfrac(1,1))
    CALL winr('T3D', rid, t3d(1,KMS,1))
    CALL winr('QV3D', rid, qv3d(1,KMS,1))
    CALL winr('QC3D', rid, qc3d(1,KMS,1))
    CALL winr('P8W', rid, p8w(1,KMS,1))
    CALL winr('RHO3D', rid, rho3d(1,KMS,1))
    CALL winr('Z3D', rid, z3d(1,KMS,1))
    CALL winr('GLW', rid, glw(1,1))
    CALL winr('GSW', rid, gsw(1,1))
    CALL winr('CHS', rid, chs(1,1))
    CALL winr('FLQC', rid, flqc(1,1))
    CALL winr('FLHC', rid, flhc(1,1))
    CALL winr('TBOT', rid, tbot(1,1))
    CALL winr('SHDMIN', rid, shdmin(1,1))
    CALL winr('SHDMAX', rid, shdmax(1,1))
    CALL winr('ALBBCK', rid, albbck(1,1))
    CALL winr('PATTERN_SPP_LSM', rid, pattern_spp_lsm(1,KMS,1))
    CALL winr('FIELD_SF', rid, field_sf(1,KMS,1))
      CALL winr('SOILT', rid, soilt(1,1));       CALL winr('SOILT1', rid, soilt1(1,1))
      CALL winr('TSNAV', rid, tsnav(1,1));       CALL winr('QVG', rid, qvg(1,1))
      CALL winr('QSG', rid, qsg(1,1));           CALL winr('QCG', rid, qcg(1,1))
      CALL winr('DEW', rid, dew(1,1));           CALL winr('QSFC', rid, qsfc(1,1))
      CALL winr('HFX', rid, hfx(1,1));           CALL winr('QFX', rid, qfx(1,1))
      CALL winr('LH', rid, lh(1,1));             CALL winr('GRDFLX', rid, grdflx(1,1))
      CALL winr('SFCRUNOFF', rid, sfcrunoff(1,1)); CALL winr('UDRUNOFF', rid, udrunoff(1,1))
      CALL winr('ACRUNOFF', rid, acrunoff(1,1)); CALL winr('SFCEXC', rid, sfcexc(1,1))
      CALL winr('SFCEVP', rid, sfcevp(1,1));     CALL winr('SMAVAIL', rid, smavail(1,1))
      CALL winr('SMMAX', rid, smmax(1,1));       CALL winr('SNOWFALLAC', rid, snowfallac(1,1))
      CALL winr('ACSNOW', rid, acsnow(1,1));     CALL winr('SNOM', rid, snom(1,1))
      CALL winr('RHOSNF', rid, rhosnf(1,1));     CALL winr('PRECIPFR', rid, precipfr(1,1))
      CALL winr('CHKLOWQ', rid, chklowq(1,1));   CALL winr('SNOW', rid, snow(1,1))
      CALL winr('SNOWH', rid, snowh(1,1));       CALL winr('SNOWC', rid, snowc(1,1))
      CALL winr('CANWAT', rid, canwat(1,1));     CALL winr('ALB', rid, alb(1,1))
      CALL winr('EMISS', rid, emiss(1,1));       CALL winr('ZNT', rid, znt(1,1))
      CALL winr('Z0', rid, z0(1,1));             CALL winr('LAI', rid, lai(1,1))
      CALL winr('MAVAIL', rid, mavail(1,1));     CALL winr('VEGFRA', rid, vegfra(1,1))
      CALL winr('SNOALB', rid, snoalb(1,1))
      DO kk = 1, nsl
         CALL winp('TSO', rid, kk, tso(1,kk,1))
         CALL winp('SOILMOIS', rid, kk, soilmois(1,kk,1))
         CALL winp('SH2O', rid, kk, sh2o(1,kk,1))
         CALL winp('SMFR3D', rid, kk, smfr3d(1,kk,1))
         CALL winp('KEEPFR3DFLAG', rid, kk, keepfr3dflag(1,kk,1))
      END DO
      IF (mosaic_lu == 1) THEN
         DO kk = 1, NLCAT
            CALL winp('LANDUSEF', rid, kk, landusef(1,kk,1))
         END DO
      END IF
      IF (mosaic_soil == 1) THEN
         DO kk = 1, NSCAT
            CALL winp('SOILCTOP', rid, kk, soilctop(1,kk,1))
         END DO
      END IF

    ! -------- integrate --------
    DO istep = 1, NSTEPS
       ktau = istep
       CALL LSMRUC(spp_lsm,                                              &
                   pattern_spp_lsm, field_sf,                            &
                   dt, ktau, nsl,                                        &
                   lakemodel, lakemask,                                  &
                   graupelncv, snowncv, rainncv,                         &
                   zs_loc, rainbl, snow, snowh, snowc, frzfrac, frpcpn,  &
                   rhosnf, precipfr,                                     &
                   z3d, p8w, t3d, qv3d, qc3d, rho3d,                     &
                   glw, gsw, emiss, chklowq, chs,                        &
                   flqc, flhc, mavail, canwat, vegfra, alb, znt,         &
                   z0, snoalb, albbck, lai,                              &
                   mminlu_c, landusef, NLCAT, mosaic_lu,                 &
                   mosaic_soil, soilctop, NSCAT,                         &
                   qsfc, qsg, qvg, qcg, dew, soilt1, tsnav,              &
                   tbot, ivgtyp, isltyp, xland,                          &
                   iswater, isice, xice, XICE_THRESHOLD_C,               &
                   cp, rcp, g, xlv, stbolt,                              &
                   soilmois, sh2o, smavail, smmax,                       &
                   tso, soilt, hfx, qfx, lh,                             &
                   sfcrunoff, udrunoff, acrunoff, sfcexc,                &
                   sfcevp, grdflx, snowfallac, acsnow, snom,             &
                   smfr3d, keepfr3dflag,                                 &
                   myj, shdmin, shdmax, rdlai2d,                         &
                   IDS,IDE, JDS,JDE, KDS,KDE,                            &
                   IMS,IME, JMS,JME, KMS,KME,                            &
                   ITS,ITE, JTS,JTE, KTS,KTE)
      CALL wsti('KTAU', rid, istep, ktau)
      CALL wst('SOILT', rid, istep, soilt(1,1));       CALL wst('SOILT1', rid, istep, soilt1(1,1))
      CALL wst('TSNAV', rid, istep, tsnav(1,1));       CALL wst('QVG', rid, istep, qvg(1,1))
      CALL wst('QSG', rid, istep, qsg(1,1));           CALL wst('QCG', rid, istep, qcg(1,1))
      CALL wst('DEW', rid, istep, dew(1,1));           CALL wst('QSFC', rid, istep, qsfc(1,1))
      CALL wst('HFX', rid, istep, hfx(1,1));           CALL wst('QFX', rid, istep, qfx(1,1))
      CALL wst('LH', rid, istep, lh(1,1));             CALL wst('GRDFLX', rid, istep, grdflx(1,1))
      CALL wst('SFCRUNOFF', rid, istep, sfcrunoff(1,1)); CALL wst('UDRUNOFF', rid, istep, udrunoff(1,1))
      CALL wst('ACRUNOFF', rid, istep, acrunoff(1,1)); CALL wst('SFCEXC', rid, istep, sfcexc(1,1))
      CALL wst('SFCEVP', rid, istep, sfcevp(1,1));     CALL wst('SMAVAIL', rid, istep, smavail(1,1))
      CALL wst('SMMAX', rid, istep, smmax(1,1));       CALL wst('SNOWFALLAC', rid, istep, snowfallac(1,1))
      CALL wst('ACSNOW', rid, istep, acsnow(1,1));     CALL wst('SNOM', rid, istep, snom(1,1))
      CALL wst('RHOSNF', rid, istep, rhosnf(1,1));     CALL wst('PRECIPFR', rid, istep, precipfr(1,1))
      CALL wst('CHKLOWQ', rid, istep, chklowq(1,1));   CALL wst('SNOW', rid, istep, snow(1,1))
      CALL wst('SNOWH', rid, istep, snowh(1,1));       CALL wst('SNOWC', rid, istep, snowc(1,1))
      CALL wst('CANWAT', rid, istep, canwat(1,1));     CALL wst('ALB', rid, istep, alb(1,1))
      CALL wst('EMISS', rid, istep, emiss(1,1));       CALL wst('ZNT', rid, istep, znt(1,1))
      CALL wst('Z0', rid, istep, z0(1,1));             CALL wst('LAI', rid, istep, lai(1,1))
      CALL wst('MAVAIL', rid, istep, mavail(1,1));     CALL wst('VEGFRA', rid, istep, vegfra(1,1))
      CALL wst('SNOALB', rid, istep, snoalb(1,1))
      DO kk = 1, nsl
         CALL wstp('TSO', rid, istep, kk, tso(1,kk,1))
         CALL wstp('SOILMOIS', rid, istep, kk, soilmois(1,kk,1))
         CALL wstp('SH2O', rid, istep, kk, sh2o(1,kk,1))
         CALL wstp('SMFR3D', rid, istep, kk, smfr3d(1,kk,1))
         CALL wstp('KEEPFR3DFLAG', rid, istep, kk, keepfr3dflag(1,kk,1))
      END DO
    END DO

    DEALLOCATE(soilmois, sh2o, tso, smfr3d, keepfr3dflag, zs_loc)


  END SUBROUTINE run_regime

  !==========================================================================
  SUBROUTINE set_defaults()
    c_name = 'unset'
    c_nsl = 6
    c_zs = 0.0
    c_zs(1:6) = ZS6_V017
    c_dt = 180.0
    c_frpcpn = .FALSE.
    c_mosaic_lu = 0; c_mosaic_soil = 0
    c_sh2o_mode = 0
    c_ivgtyp = 7; c_isltyp = 6
    c_xland = 1.0; c_xice = 0.0
    c_rainbl = 0.0; c_rainncv = 0.0; c_snowncv = 0.0; c_graupelncv = 0.0
    c_frzfrac = 0.0
    c_t3d = 288.0; c_qv = 0.006; c_qc = 0.0; c_p8w = 98000.0; c_rho = 1.2
    c_z3d = 60.0
    c_glw = 300.0; c_gsw = 0.0; c_chs = 0.015
    c_shdmin = 1.0; c_shdmax = 80.0
    c_albbck = 0.18; c_snoalb = 0.70; c_alb = 0.18; c_emiss = 0.97
    c_znt = 0.05; c_z0 = 0.05; c_lai = 2.0; c_vegfra = 50.0; c_canwat = 0.0
    c_mavail = 0.5
    c_snow = 0.0; c_snowh = 0.0; c_snowc = 0.0
    c_soilt = 288.0
    c_soilt1 = 0.0          ! WRF cold-start convention: LSMRUC initializes at ktau=1
    c_tsnav = 0.0; c_tsnav_set = .FALSE.
    c_qvg = 0.0; c_qsg = 0.0; c_qcg = 0.0; c_qsfc = 0.0
    c_tso_top = UNSET; c_tso_bot = UNSET; c_tbot = UNSET
    c_sm_top = 0.25; c_sm_bot = 0.30
    c_landusef = 0.0; c_soilctop = 0.0
  END SUBROUTINE set_defaults

  SUBROUTINE finalize_config()
    IF (c_tso_top == UNSET) c_tso_top = c_soilt
    IF (c_tso_bot == UNSET) c_tso_bot = c_tso_top
    IF (c_tbot == UNSET)    c_tbot    = c_tso_bot
    IF (.NOT. c_tsnav_set)  c_tsnav   = c_soilt - 273.15
    IF (c_mosaic_lu == 0)   c_landusef(c_ivgtyp) = 1.0     ! dominant category
    IF (c_mosaic_soil == 0) c_soilctop(c_isltyp) = 1.0
  END SUBROUTINE finalize_config

  !==========================================================================
  ! Regime table.  Units per LSMRUC header: p8w [Pa], gsw ABSORBED SW [W/m2],
  ! glw downward LW [W/m2], vegfra 0-100, xland 1 land / 2 water, rainbl /
  ! rainncv / snowncv / graupelncv = per-call amounts [mm], snow SWE [mm],
  ! snowh depth [m].  v017 regimes 1-5 (and 20) use soilt1 = soilt;
  ! all other regimes use soilt1 = 0 (LSMRUC ktau=1 initialization) unless a
  ! soilt1 is prescribed.
  SUBROUTINE config_regime(rid)
    INTEGER, INTENT(IN) :: rid
    INTEGER :: base

    base = rid
    SELECT CASE (rid)
    CASE (20);       base = 1
    CASE (24, 31);   base = 15
    CASE (25);       base = 10
    CASE (26, 27);   base = 6
    CASE (30);       base = 29
    CASE (32);       base = 10
    CASE (33);       base = 7
    CASE (34);       base = 14
    CASE (35);       base = 9
    CASE (36, 37);   base = 4
    CASE (38, 39, 40); base = 11
    CASE (41);       base = 10
    CASE (42);       base = 8
    CASE (43);       base = 9
    CASE (44);       base = 7
    CASE (45);       base = 11
    CASE (46);       base = 4
    CASE (47);       base = 16
    CASE (48);       base = 7
    CASE (49);       base = 8
    CASE (50);       base = 8
    CASE (51);       base = 11
    END SELECT

    CALL set_defaults()
    CALL config_base(base)

    SELECT CASE (rid)
    CASE (20)
       c_name = 'small_dt_unstable'
       c_dt   = 20.0
    CASE (24)
       c_name = 'forest_snow_hrrr9'
       c_nsl = 9; c_zs(1:9) = ZS9_HRRR
    CASE (25)
       c_name = 'melting_snow_hrrr9'
       c_nsl = 9; c_zs(1:9) = ZS9_HRRR
    CASE (26)
       c_name = 'frozen_soil_hrrr9'
       c_nsl = 9; c_zs(1:9) = ZS9_HRRR
    CASE (27)
       c_name = 'frozen_soil_ruclsminit'
       c_sh2o_mode = 1
    CASE (30)
       c_name = 'sea_ice_snow'
       c_snow = 50.0; c_snowh = 0.20; c_snowc = 1.0
       c_soilt = 255.0; c_alb = 0.75
    CASE (31)
       c_name = 'forest_snow_wrf6'
       c_nsl = 6; c_zs(1:6) = ZS6_WRF
    ! --- surface melt (SNOWTEMP nmelt=1 second energy iteration): skin above
    ! freezing under the snow, warm moist air, strong sun (lane o1-ruc coverage add)
    CASE (32)
       c_name = 'surface_melt_full_cover'
       c_t3d = 288.0; c_glw = 360.0; c_gsw = 400.0; c_qv = 0.0080; c_chs = 0.030
       c_soilt = 274.0; c_soilt1 = 273.0; c_tso_top = 273.5; c_alb = 0.55
    CASE (33)
       c_name = 'surface_melt_thin_snow'
       c_t3d = 286.0; c_glw = 350.0; c_gsw = 450.0; c_qv = 0.0070; c_chs = 0.030
       c_soilt = 274.5; c_tso_top = 274.0; c_tso_bot = 278.0; c_alb = 0.45
    CASE (34)
       c_name = 'surface_melt_patchy_mosaic'
       c_t3d = 287.0; c_glw = 350.0; c_gsw = 500.0; c_qv = 0.0075; c_chs = 0.030
       c_soilt = 276.0; c_tso_top = 275.0; c_tso_bot = 280.0
    CASE (35)
       c_name = 'surface_melt_two_layer_hrrr9'
       c_nsl = 9; c_zs(1:9) = ZS9_HRRR
       c_t3d = 286.0; c_glw = 340.0; c_gsw = 450.0; c_qv = 0.0070; c_chs = 0.030
       c_soilt = 274.0; c_soilt1 = 272.0; c_tso_top = 273.0; c_tso_bot = 277.0
    ! --- regimes 36-51: E154 coverage extension by critic rv-ruc (2026-10-10): convective/liquid
    ! frpcpn precipitation split, bottom melt, canopy drip into snow, snow fully gone, mid-range
    ! snow roughness, fresh-snow albedo correction, saturated soil, dew, sublimation (gcov-guided)
    CASE (36)
       c_name = 'ext_warm_rain_frpcpn'
       c_frpcpn = .TRUE.; c_frzfrac = 0.0; c_rainbl = 2.0; c_rainncv = 2.0
    CASE (37)
       c_name = 'ext_convective_warm_frpcpn'
       c_frpcpn = .TRUE.; c_frzfrac = 0.0; c_rainbl = 3.5; c_rainncv = 2.0
    CASE (38)
       c_name = 'ext_convective_cold_frzfrac'
       c_rainbl = 2.5; c_rainncv = 1.5
    CASE (39)
       c_name = 'ext_convective_cold_frzfrac0'
       c_rainbl = 2.5; c_rainncv = 1.5; c_frzfrac = 0.0; c_snowncv = 0.0; c_graupelncv = 0.0; c_t3d = 268.0
    CASE (40)
       c_name = 'ext_convective_warmair_frzfrac'
       c_rainbl = 2.5; c_rainncv = 1.5; c_t3d = 276.0; c_frzfrac = 0.5; c_snowncv = 0.6; c_graupelncv = 0.1
    CASE (41)
       c_name = 'ext_rain_on_melting_snow_frpcpn'
       c_frpcpn = .TRUE.; c_frzfrac = 0.0; c_rainbl = 2.5; c_rainncv = 1.5; c_t3d = 282.0
    CASE (42)
       c_name = 'ext_bottom_melt_one_layer'
       c_dt = 900.0; c_soilt = 272.0; c_tso_top = 279.0; c_tso_bot = 283.0; c_tbot = 283.0
    CASE (43)
       c_name = 'ext_bottom_melt_two_layer'
       c_dt = 900.0; c_soilt = 270.0; c_tso_top = 278.0; c_tso_bot = 282.0; c_tbot = 282.0
    CASE (44)
       c_name = 'ext_bottom_melt_thin_all_melts'
       c_dt = 900.0; c_soilt = 272.0; c_tso_top = 280.0; c_tso_bot = 283.0; c_tbot = 283.0
    CASE (45)
       c_name = 'ext_bottom_melt_newsnow_limit'
       c_soilt = 271.0; c_tso_top = 279.0; c_tso_bot = 283.0; c_tbot = 283.0
    CASE (46)
       c_name = 'ext_saturated_soil_runoff'
       c_sm_top = 0.43; c_sm_bot = 0.43; c_rainbl = 20.0; c_rainncv = 20.0
    CASE (47)
       c_name = 'ext_dew_bare_soil'
       c_vegfra = 0.0; c_shdmin = 0.0; c_shdmax = 1.0
    CASE (48)
       c_name = 'ext_tiny_snow_all_melts'
       c_snow = 0.3; c_snowh = 0.003; c_snowc = 0.3; c_gsw = 500.0; c_t3d = 276.0; c_soilt = 274.0; c_tso_top = 276.0; c_dt = 900.0
    CASE (49)
       c_name = 'ext_snow_sublimation_dry'
       c_snow = 0.5; c_snowh = 0.004; c_snowc = 0.5; c_gsw = 600.0; c_qv = 0.0002; c_t3d = 262.0; c_soilt = 268.0; c_tso_top = 266.0; c_tso_bot = 270.0; c_tbot = 270.0; c_dt = 900.0; c_chs = 0.03
    CASE (50)
       c_name = 'ext_snow_znt_mid_range'
       c_snow = 50.0; c_snowh = 0.20; c_snowc = 1.0
    CASE (51)
       c_name = 'ext_fresh_snow_low_albedo_forest_hrrr9'
       c_nsl = 9; c_zs(1:9) = ZS9_HRRR
       c_ivgtyp = 14; c_snoalb = 0.30; c_alb = 0.30; c_rainbl = 3.0; c_rainncv = 3.0; c_snowncv = 2.2; c_graupelncv = 0.2
    END SELECT

    CALL finalize_config()
  END SUBROUTINE config_regime

  SUBROUTINE config_base(rid)
    INTEGER, INTENT(IN) :: rid
    SELECT CASE (rid)
    ! ----------------- v017 regimes (identical inputs) -----------------
    CASE (1)
       c_name='unstable_day_grassland_loam'
       c_ivgtyp=7; c_isltyp=6; c_gsw=600.0; c_glw=350.0; c_t3d=298.0
       c_qv=0.0090; c_rho=1.16; c_p8w=99500.0; c_rainbl=0.0; c_vegfra=60.0
       c_chs=0.025; c_tbot=290.0; c_mavail=0.55; c_soilt=300.0
       c_sm_top=0.20; c_sm_bot=0.25
       c_tso_top=c_soilt; c_tso_bot=c_tbot; c_soilt1=c_soilt; c_rainncv=c_rainbl
    CASE (2)
       c_name='stable_night_grassland_loam'
       c_ivgtyp=7; c_isltyp=6; c_gsw=0.0; c_glw=300.0; c_t3d=286.0
       c_qv=0.0070; c_rho=1.20; c_p8w=95000.0; c_rainbl=0.0; c_vegfra=60.0
       c_chs=0.010; c_tbot=288.0; c_mavail=0.55; c_soilt=287.0
       c_sm_top=0.20; c_sm_bot=0.25
       c_tso_top=c_soilt; c_tso_bot=c_tbot; c_soilt1=c_soilt; c_rainncv=c_rainbl
    CASE (3)
       c_name='unstable_day_cropland_sandyloam'
       c_ivgtyp=2; c_isltyp=3; c_gsw=700.0; c_glw=360.0; c_t3d=301.0
       c_qv=0.0110; c_rho=1.10; c_p8w=96000.0; c_rainbl=0.0; c_vegfra=70.0
       c_chs=0.030; c_tbot=292.0; c_mavail=0.40; c_soilt=305.0
       c_sm_top=0.12; c_sm_bot=0.18
       c_tso_top=c_soilt; c_tso_bot=c_tbot; c_soilt1=c_soilt; c_rainncv=c_rainbl
    CASE (4)
       c_name='wet_precip_cropland_loam'
       c_ivgtyp=2; c_isltyp=6; c_gsw=300.0; c_glw=360.0; c_t3d=295.0
       c_qv=0.0140; c_rho=1.16; c_p8w=99000.0; c_rainbl=2.0; c_vegfra=65.0
       c_chs=0.020; c_tbot=292.0; c_mavail=0.70; c_soilt=296.0
       c_sm_top=0.32; c_sm_bot=0.36
       c_tso_top=c_soilt; c_tso_bot=c_tbot; c_soilt1=c_soilt; c_rainncv=c_rainbl
    CASE (5)
       c_name='near_neutral_shrubland_sandyloam'
       c_ivgtyp=8; c_isltyp=3; c_gsw=200.0; c_glw=330.0; c_t3d=292.0
       c_qv=0.0085; c_rho=1.18; c_p8w=95000.0; c_rainbl=0.0; c_vegfra=40.0
       c_chs=0.018; c_tbot=291.0; c_mavail=0.45; c_soilt=292.2
       c_sm_top=0.15; c_sm_bot=0.20
       c_tso_top=c_soilt; c_tso_bot=c_tbot; c_soilt1=c_soilt; c_rainncv=c_rainbl
    ! ----------------- v0.34 regimes -----------------
    CASE (6)
       c_name='frozen_soil_cold_night'
       c_ivgtyp=7; c_isltyp=6; c_gsw=0.0; c_glw=230.0; c_t3d=263.0
       c_qv=0.0015; c_rho=1.297; c_p8w=98000.0; c_chs=0.008; c_vegfra=30.0
       c_mavail=0.95; c_soilt=265.0; c_tso_top=265.0; c_tso_bot=275.0; c_tbot=276.0
       c_sm_top=0.30; c_sm_bot=0.30
    CASE (7)
       c_name='thin_snow'
       c_ivgtyp=7; c_isltyp=6; c_gsw=150.0; c_glw=250.0; c_t3d=268.0
       c_qv=0.0025; c_rho=1.272; c_p8w=98000.0; c_chs=0.012; c_vegfra=30.0
       c_mavail=0.76; c_soilt=268.0; c_tso_top=268.0; c_tso_bot=275.0
       c_sm_top=0.25; c_sm_bot=0.30
       c_snow=5.0; c_snowh=0.02; c_snowc=0.6; c_alb=0.45
    CASE (8)
       c_name='single_layer_snow'
       c_ivgtyp=7; c_isltyp=4; c_gsw=100.0; c_glw=240.0; c_t3d=266.0
       c_qv=0.0022; c_rho=1.269; c_p8w=97000.0; c_chs=0.010; c_vegfra=30.0
       c_mavail=0.65; c_soilt=265.0; c_soilt1=267.0; c_tso_top=270.0; c_tso_bot=277.0
       c_sm_top=0.25; c_sm_bot=0.30
       c_snow=30.0; c_snowh=0.12; c_snowc=1.0; c_alb=0.60
    CASE (9)
       c_name='two_layer_deep_snow'
       c_ivgtyp=20; c_isltyp=4; c_gsw=50.0; c_glw=220.0; c_t3d=260.0
       c_qv=0.0012; c_rho=1.286; c_p8w=96000.0; c_chs=0.008; c_vegfra=20.0
       c_mavail=0.73; c_soilt=259.0; c_soilt1=262.0; c_tso_top=271.0; c_tso_bot=276.0
       c_sm_top=0.28; c_sm_bot=0.28
       c_snow=150.0; c_snowh=0.6; c_snowc=1.0; c_alb=0.65
    CASE (10)
       c_name='melting_snow_warm_day'
       c_ivgtyp=7; c_isltyp=6; c_gsw=450.0; c_glw=320.0; c_t3d=281.0
       c_qv=0.005; c_rho=1.218; c_p8w=98500.0; c_chs=0.020; c_vegfra=30.0
       c_mavail=0.95; c_soilt=273.0; c_soilt1=272.8; c_tso_top=273.5; c_tso_bot=279.0
       c_sm_top=0.30; c_sm_bot=0.32
       c_snow=40.0; c_snowh=0.15; c_snowc=1.0; c_alb=0.55
    CASE (11)
       c_name='mixed_snowfall_frpcpn'
       c_frpcpn=.TRUE.
       c_ivgtyp=2; c_isltyp=6; c_gsw=50.0; c_glw=300.0; c_t3d=271.0
       c_qv=0.0031; c_rho=1.258; c_p8w=98000.0; c_chs=0.012; c_vegfra=20.0
       c_rainbl=1.5; c_rainncv=1.5; c_snowncv=1.1; c_graupelncv=0.1; c_frzfrac=0.85
       c_mavail=0.86; c_soilt=271.0; c_tso_top=272.0; c_tso_bot=277.0
       c_sm_top=0.27; c_sm_bot=0.30
       c_snow=10.0; c_snowh=0.05; c_snowc=0.8; c_alb=0.50
    CASE (12)
       c_name='fresh_snow_on_bare_ground'
       c_frpcpn=.FALSE.
       c_ivgtyp=2; c_isltyp=3; c_gsw=30.0; c_glw=290.0; c_t3d=270.0
       c_qv=0.0028; c_rho=1.263; c_p8w=98000.0; c_chs=0.012; c_vegfra=20.0
       c_rainbl=2.0; c_rainncv=2.0
       c_mavail=1.0; c_soilt=271.0; c_tso_top=273.0; c_tso_bot=278.0
       c_sm_top=0.25; c_sm_bot=0.28
    CASE (13)
       c_name='rain_on_frozen_soil'
       c_ivgtyp=7; c_isltyp=9; c_gsw=80.0; c_glw=320.0; c_t3d=277.0
       c_qv=0.006; c_rho=1.235; c_p8w=98500.0; c_chs=0.015; c_vegfra=30.0
       c_rainbl=3.0; c_rainncv=3.0
       c_mavail=1.0; c_soilt=272.0; c_tso_top=271.0; c_tso_bot=276.0
       c_sm_top=0.32; c_sm_bot=0.32
    CASE (14)
       c_name='patchy_snow_mosaic'
       c_ivgtyp=8; c_isltyp=3; c_gsw=300.0; c_glw=280.0; c_t3d=272.0
       c_qv=0.003; c_rho=1.253; c_p8w=98000.0; c_chs=0.018; c_vegfra=25.0
       c_mavail=0.90; c_soilt=272.0; c_tso_top=274.0; c_tso_bot=280.0
       c_sm_top=0.22; c_sm_bot=0.28
       c_snow=3.0; c_snowh=0.015; c_snowc=0.3; c_alb=0.30
    CASE (15)
       c_name='forest_snow'
       c_ivgtyp=14; c_isltyp=6; c_gsw=200.0; c_glw=260.0; c_t3d=270.0
       c_qv=0.0025; c_rho=1.224; c_p8w=95000.0; c_chs=0.020; c_vegfra=70.0
       c_mavail=0.86; c_soilt=269.0; c_tso_top=271.0; c_tso_bot=276.0
       c_sm_top=0.28; c_sm_bot=0.30
       c_snow=60.0; c_snowh=0.25; c_snowc=1.0; c_alb=0.35; c_snoalb=0.75
    CASE (16)
       c_name='dew_night_condensation'
       c_ivgtyp=7; c_isltyp=6; c_gsw=0.0; c_glw=360.0; c_t3d=285.0
       c_qv=0.0115; c_rho=1.214; c_p8w=100000.0; c_chs=0.006; c_vegfra=60.0
       c_mavail=0.95; c_soilt=283.0; c_tso_top=284.0; c_tso_bot=288.0
       c_sm_top=0.30; c_sm_bot=0.32
    CASE (17)
       c_name='urban_snow'
       c_ivgtyp=1; c_isltyp=6; c_gsw=100.0; c_glw=250.0; c_t3d=268.0
       c_qv=0.0025; c_rho=1.285; c_p8w=99000.0; c_chs=0.015; c_vegfra=10.0
       c_mavail=0.76; c_soilt=268.0; c_tso_top=270.0; c_tso_bot=276.0
       c_sm_top=0.25; c_sm_bot=0.28
       c_snow=20.0; c_snowh=0.08; c_snowc=1.0; c_alb=0.45
    CASE (18)
       c_name='dry_sand_hot'
       c_ivgtyp=19; c_isltyp=1; c_gsw=800.0; c_glw=380.0; c_t3d=308.0
       c_qv=0.004; c_rho=1.095; c_p8w=97000.0; c_chs=0.025; c_vegfra=2.0
       c_mavail=0.28; c_soilt=318.0; c_tso_top=318.0; c_tso_bot=300.0
       c_sm_top=0.05; c_sm_bot=0.10; c_alb=0.25
    CASE (19)
       c_name='clay_wet'
       c_ivgtyp=6; c_isltyp=12; c_gsw=250.0; c_glw=350.0; c_t3d=293.0
       c_qv=0.012; c_rho=1.169; c_p8w=99000.0; c_chs=0.015; c_vegfra=70.0
       c_rainbl=1.0; c_rainncv=1.0
       c_mavail=1.0; c_soilt=294.0; c_tso_top=294.0; c_tso_bot=288.0
       c_sm_top=0.40; c_sm_bot=0.45
    CASE (21)
       c_name='water_point'
       c_xland=2.0; c_ivgtyp=16; c_isltyp=14; c_gsw=500.0; c_glw=350.0
       c_t3d=288.0; c_qv=0.008; c_rho=1.216; c_p8w=101000.0; c_chs=0.012
       c_vegfra=0.0; c_mavail=1.0; c_soilt=290.0; c_tso_top=290.0; c_tso_bot=290.0
       c_sm_top=1.0; c_sm_bot=1.0; c_alb=0.08; c_albbck=0.08; c_emiss=0.98
       c_znt=0.0001; c_z0=0.0001; c_lai=0.01
    CASE (22)
       c_name='mosaic_lu_soil'
       c_mosaic_lu=1; c_mosaic_soil=1
       c_landusef(2)=0.5; c_landusef(7)=0.3; c_landusef(11)=0.2
       c_soilctop(3)=0.6; c_soilctop(6)=0.4
       c_ivgtyp=2; c_isltyp=3; c_vegfra=80.0; c_shdmin=10.0; c_shdmax=90.0
       c_gsw=600.0; c_glw=360.0; c_t3d=296.0; c_qv=0.010; c_rho=1.141
       c_p8w=97500.0; c_chs=0.022; c_mavail=0.15
       c_soilt=300.0; c_tso_top=300.0; c_tso_bot=290.0
       c_sm_top=0.08; c_sm_bot=0.08
    CASE (23)
       c_name='glacier_land_ice'
       c_ivgtyp=24; c_isltyp=16; c_gsw=300.0; c_glw=200.0; c_t3d=255.0
       c_qv=0.0008; c_rho=1.297; c_p8w=95000.0; c_chs=0.010; c_vegfra=0.0
       c_mavail=1.0; c_soilt=255.0; c_tso_top=255.0; c_tso_bot=260.0
       c_sm_top=0.35; c_sm_bot=0.35
       c_snow=500.0; c_snowh=1.5; c_snowc=1.0; c_alb=0.75; c_snoalb=0.75
    CASE (28)
       c_name='mosaic_crop_irrigation'
       c_mosaic_lu=1; c_mosaic_soil=1
       c_landusef(3)=0.4; c_landusef(5)=0.3; c_landusef(7)=0.2; c_landusef(11)=0.1
       c_soilctop(3)=0.6; c_soilctop(6)=0.4
       c_ivgtyp=3; c_isltyp=3; c_vegfra=80.0; c_shdmin=10.0; c_shdmax=90.0
       c_gsw=650.0; c_glw=370.0; c_t3d=300.0; c_qv=0.008; c_rho=1.127
       c_p8w=97500.0; c_chs=0.022; c_mavail=0.15
       c_soilt=305.0; c_tso_top=305.0; c_tso_bot=292.0
       c_sm_top=0.08; c_sm_bot=0.08
    CASE (29)
       c_name='sea_ice_bare'
       c_xland=1.0; c_xice=1.0; c_ivgtyp=24; c_isltyp=16
       c_gsw=100.0; c_glw=220.0; c_t3d=255.0; c_qv=0.0009; c_rho=1.379
       c_p8w=101000.0; c_chs=0.010; c_vegfra=0.0; c_mavail=1.0
       c_soilt=258.0; c_tso_top=258.0; c_tso_bot=271.4; c_tbot=271.4
       c_sm_top=1.0; c_sm_bot=1.0
       c_alb=0.65; c_albbck=0.65; c_snoalb=0.75; c_emiss=0.98
    CASE DEFAULT
       WRITE(0,*) 'config_base: unknown regime ', rid
       STOP 2
    END SELECT
  END SUBROUTINE config_base

  !==========================================================================
  ! writers
  SUBROUTINE wgi(name, iv)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: iv
    WRITE(*,'(A,A,I0)') TRIM(name), '=', iv
  END SUBROUTINE wgi

  SUBROUTINE wgr(name, v)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, INTENT(IN) :: v
    WRITE(*,'(A,A,ES24.16E3)') TRIM(name), '=', v
  END SUBROUTINE wgr

  SUBROUTINE wini(name, r, iv)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: r, iv
    WRITE(*,'(A,A,A,I0,A,I0)') 'IN_', TRIM(name), '[', r, ']=', iv
  END SUBROUTINE wini

  SUBROUTINE winr(name, r, v)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: r
    REAL, INTENT(IN) :: v
    WRITE(*,'(A,A,A,I0,A,ES24.16E3)') 'IN_', TRIM(name), '[', r, ']=', v
  END SUBROUTINE winr

  SUBROUTINE winp(name, r, k, v)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: r, k
    REAL, INTENT(IN) :: v
    WRITE(*,'(A,A,A,I0,A,I0,A,ES24.16E3)') 'IN_', TRIM(name), '[', r, '][', k, ']=', v
  END SUBROUTINE winp

  SUBROUTINE wsti(name, r, s, iv)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: r, s, iv
    WRITE(*,'(A,A,I0,A,I0,A,I0)') TRIM(name), '[', r, '][', s, ']=', iv
  END SUBROUTINE wsti

  SUBROUTINE wst(name, r, s, v)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: r, s
    REAL, INTENT(IN) :: v
    WRITE(*,'(A,A,I0,A,I0,A,ES24.16E3)') TRIM(name), '[', r, '][', s, ']=', v
  END SUBROUTINE wst

  SUBROUTINE wstp(name, r, s, k, v)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: r, s, k
    REAL, INTENT(IN) :: v
    WRITE(*,'(A,A,I0,A,I0,A,I0,A,ES24.16E3)') TRIM(name), '[', r, '][', s, '][', k, ']=', v
  END SUBROUTINE wstp

END PROGRAM ruclsm_oracle_v2
