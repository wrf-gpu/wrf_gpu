PROGRAM ruclsm_oracle
  !---------------------------------------------------------------------------
  ! Standalone fp64 oracle for the WRF RUC land-surface model
  ! (sf_surface_physics=3).  Calls the UNMODIFIED WRF driver subroutine LSMRUC
  ! (phys/module_sf_ruclsm.F:84) directly on a 1x1x1 grid, once per regime,
  ! over a short multi-step integration.  LSMRUC internally calls SOILVEGIN
  ! (which derives the soil/veg constants from the module-level tables) and
  ! SFCTMP (the column solver), so the ENTIRE WRF driver path runs unmodified.
  !
  ! The module-level soil/veg parameter tables (hc, bb, maxsmc, drysmc, satdk,
  ! satpsi, refsmc, wltsmc, qtz; z0tbl, lemitbl, pctbl, laitbl, ifortbl, ...)
  ! are populated EXACTLY as WRF's lsminit does, by calling the real
  ! RUCLSM_SOILVEGPARM('USGS-RUC','STAS-RUC') once before the regime loop.
  ! That reads the unmodified VEGPARM.TBL / SOILPARM.TBL / GENPARM.TBL files
  ! copied into the build dir.
  !
  ! Soil column: nsl=6 RUC levels zs=[0,0.05,0.20,0.40,1.0,2.0] m.
  ! Land point, USGS veg + STASGO(STAS-RUC) soil, no snow.
  ! Emits flat `VAR[i]=value` lines (inputs + INOUT carry + outputs) for parsing
  ! into JSON savepoints.  i indexes the regime/case (one column each here).
  !---------------------------------------------------------------------------
  USE module_sf_ruclsm, ONLY : LSMRUC, RUCLSM_SOILVEGPARM
  IMPLICIT NONE

  ! ---- grid dimensions: single point, single atmospheric level ----
  INTEGER, PARAMETER :: IMS=1, IME=1, JMS=1, JME=1, KMS=1, KME=1
  INTEGER, PARAMETER :: IDS=1, IDE=2, JDS=1, JDE=2, KDS=1, KDE=2
  INTEGER, PARAMETER :: ITS=1, ITE=1, JTS=1, JTE=1, KTS=1, KTE=1

  INTEGER, PARAMETER :: NSL  = 6        ! soil levels
  INTEGER, PARAMETER :: NLCAT = 24      ! USGS land categories (landusef)
  INTEGER, PARAMETER :: NSCAT = 19      ! STAS-RUC soil categories (soilctop)
  INTEGER, PARAMETER :: NREG  = 5       ! number of regimes (columns reported)
  INTEGER, PARAMETER :: NSTEPS = 6      ! integration steps per regime

  ! WRF model constants (passed explicitly into LSMRUC argument list)
  REAL, PARAMETER :: CP_C     = 1004.5
  REAL, PARAMETER :: ROVCP_C  = 0.2857
  REAL, PARAMETER :: G0_C     = 9.81
  REAL, PARAMETER :: LV_C     = 2.5E6
  REAL, PARAMETER :: STBOLT_C = 5.67051E-8
  REAL, PARAMETER :: XICE_THRESHOLD_C = 0.5
  REAL, PARAMETER :: P1000   = 100000.0

  ! ---- soil level depths (m), zs(1)=0 surface; RUC 6-level ----
  REAL, DIMENSION(NSL) :: ZS

  ! ---- per-regime stored INPUTS (for the savepoint) ----
  REAL,    DIMENSION(NREG) :: R_DT, R_SOLDN_GSW, R_GLW, R_EMISS, R_TABS
  REAL,    DIMENSION(NREG) :: R_QV, R_QC, R_RHO, R_P8W, R_Z3D
  REAL,    DIMENSION(NREG) :: R_RAINBL, R_VEGFRA, R_CHS, R_FLHC, R_FLQC
  REAL,    DIMENSION(NREG) :: R_TBOT, R_XLAND, R_MAVAIL0
  INTEGER, DIMENSION(NREG) :: R_IVGTYP, R_ISLTYP
  REAL,    DIMENSION(NREG) :: R_SOILT0, R_SOILMOIS0_1, R_TSO0_1
  CHARACTER(LEN=64), DIMENSION(NREG) :: R_NAME

  ! ---- per-regime stored OUTPUTS (post-integration) ----
  REAL, DIMENSION(NREG) :: O_SOILT, O_HFX, O_QFX, O_LH, O_GRDFLX
  REAL, DIMENSION(NREG) :: O_SMAVAIL, O_SMMAX, O_SFCRUNOFF, O_UDRUNOFF
  REAL, DIMENSION(NREG) :: O_QSFC, O_QSG, O_QVG, O_QCG, O_DEW
  REAL, DIMENSION(NREG) :: O_SNOW, O_SNOWH, O_ALB, O_EMISS, O_ZNT, O_LAI, O_MAVAIL
  REAL, DIMENSION(NREG) :: O_SFCEVP, O_SFCEXC, O_TSNAV, O_SOILT1
  REAL, DIMENSION(NREG, NSL) :: O_TSO, O_SOILMOIS, O_SH2O

  INTEGER :: ireg, istep, k
  CHARACTER(LEN=32) :: arg, precision_mode
  CHARACTER(LEN=4)  :: mminlu_local

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
     CALL GET_COMMAND_ARGUMENT(1, precision_mode)
  ELSE
     precision_mode = 'fp64'
  END IF

  ! ---- RUC 6-level soil depths (m) ----
  ZS = (/ 0.0, 0.05, 0.20, 0.40, 1.0, 2.0 /)

  ! ---- Populate module soil/veg tables exactly like WRF lsminit ----
  ! USGS land-use -> 'USGS-RUC'; STASGO soil -> 'STAS-RUC'.
  CALL RUCLSM_SOILVEGPARM('USGS-RUC', 'STAS-RUC')

  ! ---- Drive each regime ----
  DO ireg = 1, NREG
     CALL run_regime(ireg)
  END DO

  ! ---- Emit savepoint ----
  CALL emit_savepoint()

CONTAINS

  !==========================================================================
  SUBROUTINE run_regime(rid)
    INTEGER, INTENT(IN) :: rid

    ! LSMRUC argument arrays (all 1x1 or 1x1xK / 1xNSLx1 etc.)
    REAL    :: dt
    INTEGER :: ktau
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
    REAL, DIMENSION(IMS:IME, NSL, JMS:JME) :: soilmois, sh2o, tso
    REAL, DIMENSION(IMS:IME, NSL, JMS:JME) :: smfr3d, keepfr3dflag
    REAL, DIMENSION(IMS:IME, NLCAT, JMS:JME) :: landusef
    REAL, DIMENSION(IMS:IME, NSCAT, JMS:JME) :: soilctop
    REAL, DIMENSION(NSL) :: zs_loc

    CHARACTER(LEN=4) :: mminlu
    REAL :: ivg_r, isl_r, soldn_in, glw_in, emiss_in, tabs_in, qv_in, qc_in
    REAL :: rho_in, p8w_in, z3d_in, rainbl_in, vegfra_in, chs_in, tbot_in
    REAL :: xland_in, mavail_in, soilt_in, sm1_in, sm2_in, theta1
    REAL :: dz, p_top
    REAL :: tso_init(NSL), sm_init(NSL)
    CHARACTER(LEN=64) :: nm

    ! -------- regime configuration --------
    CALL config_regime(rid, nm, ivg_r, isl_r, soldn_in, glw_in, emiss_in,    &
                        tabs_in, qv_in, qc_in, rho_in, p8w_in, z3d_in,        &
                        rainbl_in, vegfra_in, chs_in, tbot_in, xland_in,      &
                        mavail_in, soilt_in, sm1_in, sm2_in, frpcpn)

    ! -------- scalar / flag setup (single-rank, EM_CORE, no SPP) --------
    dt          = 180.0
    myj         = .FALSE.      ! use flqc/flhc path for qkms/tkms
    rdlai2d     = .FALSE.
    spp_lsm     = 0
    mosaic_lu   = 0            ! dominant-category veg
    mosaic_soil = 0            ! dominant-category soil
    iswater     = 16          ! USGS water category
    isice       = 24          ! USGS snow/ice category
    lakemodel   = 0
    mminlu      = 'USGS'

    pattern_spp_lsm = 0.0
    field_sf        = 0.0

    ! -------- atmospheric forcing (single surface level at kms) --------
    z3d (1,KMS,1) = z3d_in     ! thickness 1st full sigma level to surface (m)
    rho3d(1,KMS,1) = rho_in
    t3d (1,KMS,1) = tabs_in
    qv3d(1,KMS,1) = qv_in
    qc3d(1,KMS,1) = qc_in
    p8w (1,KMS,1) = p8w_in     ! [Pa]

    rainbl(1,1)   = rainbl_in  ! accumulated rain in [mm] over the pbl step
    rainncv(1,1)  = rainbl_in  ! grid-scale precip this step [mm]
    snowncv(1,1)  = 0.0
    graupelncv(1,1)= 0.0
    frzfrac(1,1)  = 0.0        ! no frozen precip
    snow (1,1)    = 0.0
    snowh(1,1)    = 0.0
    snowc(1,1)    = 0.0

    glw (1,1)     = glw_in
    gsw (1,1)     = soldn_in   ! absorbed shortwave at ground (W/m^2)
    emiss(1,1)    = emiss_in
    chs (1,1)     = chs_in
    ! exchange coefficients: flqc [kg/m^2/s], flhc [W/m^2/s/K].
    ! For non-myj path: qkms=flqc/rho/mavail, tkms=flhc/rho/(cp*(1+0.84*qv)).
    flqc(1,1)     = chs_in * rho_in * mavail_in
    flhc(1,1)     = chs_in * rho_in * CP_C
    mavail(1,1)   = mavail_in
    canwat(1,1)   = 0.0
    vegfra(1,1)   = vegfra_in  ! 0-100

    alb (1,1)     = 0.18
    albbck(1,1)   = 0.18
    snoalb(1,1)   = 0.70
    znt (1,1)     = 0.05
    z0  (1,1)     = 0.05
    lai (1,1)     = 2.0
    emiss(1,1)    = emiss_in

    tbot(1,1)     = tbot_in
    xland(1,1)    = xland_in
    xice(1,1)     = 0.0
    shdmin(1,1)   = 1.0
    shdmax(1,1)   = 80.0
    lakemask(1,1) = 0.0

    ivgtyp(1,1)   = NINT(ivg_r)
    isltyp(1,1)   = NINT(isl_r)

    ! -------- initial surface / soil state --------
    soilt(1,1)    = soilt_in
    soilt1(1,1)   = soilt_in     ! snow-soil interface T (no snow -> = soilt)
    tsnav(1,1)    = soilt_in - 273.15
    qvg(1,1)      = 0.0          ! let LSMRUC init from qsg*mavail on ktau=1
    qsg(1,1)      = 0.0
    qcg(1,1)      = 0.0
    dew(1,1)      = 0.0
    qsfc(1,1)     = 0.0

    ! linear soil temperature profile from skin to tbot; moisture top->deep
    DO k = 1, NSL
       tso_init(k) = soilt_in + (tbot_in - soilt_in) * (ZS(k) / ZS(NSL))
       sm_init(k)  = sm1_in + (sm2_in - sm1_in) * (ZS(k) / ZS(NSL))
    END DO
    DO k = 1, NSL
       tso(1,k,1)      = tso_init(k)
       soilmois(1,k,1) = sm_init(k)
       sh2o(1,k,1)     = sm_init(k)     ! all liquid (warm soil)
       smfr3d(1,k,1)   = 0.0
       keepfr3dflag(1,k,1) = 0.0
       zs_loc(k)       = ZS(k)
    END DO

    ! dominant-category land use / soil fractions (not used when mosaic=0,
    ! but must be defined; set the dominant type to 1.0).
    landusef = 0.0
    soilctop = 0.0
    landusef(1, NINT(ivg_r), 1) = 1.0
    soilctop(1, NINT(isl_r), 1) = 1.0

    ! diagnostics / accumulators
    smavail=0.; smmax=0.; sfcrunoff=0.; udrunoff=0.; acrunoff=0.
    sfcexc=0.; sfcevp=0.; grdflx=0.; snowfallac=0.; acsnow=0.; snom=0.
    chklowq=1.; hfx=0.; qfx=0.; lh=0.; rhosnf=-1.e3; precipfr=0.

    ! -------- store the regime inputs for the savepoint --------
    R_NAME(rid)       = nm
    R_DT(rid)         = dt
    R_SOLDN_GSW(rid)  = soldn_in
    R_GLW(rid)        = glw_in
    R_EMISS(rid)      = emiss_in
    R_TABS(rid)       = tabs_in
    R_QV(rid)         = qv_in
    R_QC(rid)         = qc_in
    R_RHO(rid)        = rho_in
    R_P8W(rid)        = p8w_in
    R_Z3D(rid)        = z3d_in
    R_RAINBL(rid)     = rainbl_in
    R_VEGFRA(rid)     = vegfra_in
    R_CHS(rid)        = chs_in
    R_FLHC(rid)       = flhc(1,1)
    R_FLQC(rid)       = flqc(1,1)
    R_TBOT(rid)       = tbot_in
    R_XLAND(rid)      = xland_in
    R_MAVAIL0(rid)    = mavail_in
    R_IVGTYP(rid)     = NINT(ivg_r)
    R_ISLTYP(rid)     = NINT(isl_r)
    R_SOILT0(rid)     = soilt_in
    R_SOILMOIS0_1(rid)= sm_init(1)
    R_TSO0_1(rid)     = tso_init(1)

    ! -------- integrate NSTEPS surface steps --------
    DO istep = 1, NSTEPS
       ktau = istep
       CALL LSMRUC(spp_lsm,                                              &
                   pattern_spp_lsm, field_sf,                            &
                   dt, ktau, NSL,                                        &
                   lakemodel, lakemask,                                  &
                   graupelncv, snowncv, rainncv,                         &
                   zs_loc, rainbl, snow, snowh, snowc, frzfrac, frpcpn,  &
                   rhosnf, precipfr,                                     &
                   z3d, p8w, t3d, qv3d, qc3d, rho3d,                     &
                   glw, gsw, emiss, chklowq, chs,                        &
                   flqc, flhc, mavail, canwat, vegfra, alb, znt,         &
                   z0, snoalb, albbck, lai,                              &
                   mminlu, landusef, NLCAT, mosaic_lu,                   &
                   mosaic_soil, soilctop, NSCAT,                         &
                   qsfc, qsg, qvg, qcg, dew, soilt1, tsnav,              &
                   tbot, ivgtyp, isltyp, xland,                         &
                   iswater, isice, xice, XICE_THRESHOLD_C,               &
                   CP_C, ROVCP_C, G0_C, LV_C, STBOLT_C,                  &
                   soilmois, sh2o, smavail, smmax,                       &
                   tso, soilt, hfx, qfx, lh,                             &
                   sfcrunoff, udrunoff, acrunoff, sfcexc,                &
                   sfcevp, grdflx, snowfallac, acsnow, snom,             &
                   smfr3d, keepfr3dflag,                                 &
                   myj, shdmin, shdmax, rdlai2d,                         &
                   IDS,IDE, JDS,JDE, KDS,KDE,                            &
                   IMS,IME, JMS,JME, KMS,KME,                            &
                   ITS,ITE, JTS,JTE, KTS,KTE)
    END DO

    ! -------- capture outputs --------
    O_SOILT(rid)    = soilt(1,1)
    O_HFX(rid)      = hfx(1,1)
    O_QFX(rid)      = qfx(1,1)
    O_LH(rid)       = lh(1,1)
    O_GRDFLX(rid)   = grdflx(1,1)
    O_SMAVAIL(rid)  = smavail(1,1)
    O_SMMAX(rid)    = smmax(1,1)
    O_SFCRUNOFF(rid)= sfcrunoff(1,1)
    O_UDRUNOFF(rid) = udrunoff(1,1)
    O_QSFC(rid)     = qsfc(1,1)
    O_QSG(rid)      = qsg(1,1)
    O_QVG(rid)      = qvg(1,1)
    O_QCG(rid)      = qcg(1,1)
    O_DEW(rid)      = dew(1,1)
    O_SNOW(rid)     = snow(1,1)
    O_SNOWH(rid)    = snowh(1,1)
    O_ALB(rid)      = alb(1,1)
    O_EMISS(rid)    = emiss(1,1)
    O_ZNT(rid)      = znt(1,1)
    O_LAI(rid)      = lai(1,1)
    O_MAVAIL(rid)   = mavail(1,1)
    O_SFCEVP(rid)   = sfcevp(1,1)
    O_SFCEXC(rid)   = sfcexc(1,1)
    O_TSNAV(rid)    = tsnav(1,1)
    O_SOILT1(rid)   = soilt1(1,1)
    DO k = 1, NSL
       O_TSO(rid,k)      = tso(1,k,1)
       O_SOILMOIS(rid,k) = soilmois(1,k,1)
       O_SH2O(rid,k)     = sh2o(1,k,1)
    END DO
  END SUBROUTINE run_regime

  !==========================================================================
  ! Per-regime forcing. Units per LSMRUC comment block:
  !   p8w [Pa], gsw absorbed SW [W/m^2], glw downward LW [W/m^2],
  !   vegfra 0-100, xland 1=land/2=water, rainbl accumulated mm over pbl step.
  SUBROUTINE config_regime(rid, nm, ivg, isl, soldn, glw, emiss, tabs, qv, qc, &
                           rho, p8w, z3d, rainbl, vegfra, chs, tbot, xland,    &
                           mavail, soilt, sm1, sm2, frpcpn)
    INTEGER, INTENT(IN) :: rid
    CHARACTER(LEN=*), INTENT(OUT) :: nm
    REAL, INTENT(OUT) :: ivg, isl, soldn, glw, emiss, tabs, qv, qc, rho, p8w
    REAL, INTENT(OUT) :: z3d, rainbl, vegfra, chs, tbot, xland, mavail
    REAL, INTENT(OUT) :: soilt, sm1, sm2
    LOGICAL, INTENT(OUT) :: frpcpn

    frpcpn = .FALSE.
    qc     = 0.0
    z3d    = 60.0       ! thickness 1st sigma level to surface (m); conflx=30m
    xland  = 1.0        ! land
    emiss  = 0.97
    SELECT CASE (rid)
    CASE (1)
       nm     = 'unstable_day_grassland_loam'
       ivg    = 7.0     ! USGS grassland
       isl    = 6.0     ! STASGO loam
       soldn  = 600.0   ! strong absorbed SW
       glw    = 350.0
       tabs   = 298.0
       qv     = 0.0090
       rho    = 1.16
       p8w    = 99500.0
       rainbl = 0.0
       vegfra = 60.0
       chs    = 0.025   ! exchange coeff (m/s)
       tbot   = 290.0
       mavail = 0.55
       soilt  = 300.0
       sm1    = 0.20
       sm2    = 0.25
    CASE (2)
       nm     = 'stable_night_grassland_loam'
       ivg    = 7.0
       isl    = 6.0
       soldn  = 0.0     ! night
       glw    = 300.0
       tabs   = 286.0
       qv     = 0.0070
       rho    = 1.20
       p8w    = 95000.0
       rainbl = 0.0
       vegfra = 60.0
       chs    = 0.010
       tbot   = 288.0
       mavail = 0.55
       soilt  = 287.0
       sm1    = 0.20
       sm2    = 0.25
    CASE (3)
       nm     = 'unstable_day_cropland_sandyloam'
       ivg    = 2.0     ! USGS dryland cropland and pasture
       isl    = 3.0     ! STASGO sandy loam
       soldn  = 700.0
       glw    = 360.0
       tabs   = 301.0
       qv     = 0.0110
       rho    = 1.10
       p8w    = 96000.0
       rainbl = 0.0
       vegfra = 70.0
       chs    = 0.030
       tbot   = 292.0
       mavail = 0.40
       soilt  = 305.0
       sm1    = 0.12
       sm2    = 0.18
    CASE (4)
       nm     = 'wet_precip_cropland_loam'
       ivg    = 2.0
       isl    = 6.0
       soldn  = 300.0
       glw    = 360.0
       tabs   = 295.0
       qv     = 0.0140
       rho    = 1.16
       p8w    = 99000.0
       rainbl = 2.0     ! 2 mm over the step
       vegfra = 65.0
       chs    = 0.020
       tbot   = 292.0
       mavail = 0.70
       soilt  = 296.0
       sm1    = 0.32
       sm2    = 0.36
    CASE DEFAULT
       nm     = 'near_neutral_shrubland_sandyloam'
       ivg    = 8.0     ! USGS shrubland
       isl    = 3.0     ! sandy loam
       soldn  = 200.0
       glw    = 330.0
       tabs   = 292.0
       qv     = 0.0085
       rho    = 1.18
       p8w    = 95000.0
       rainbl = 0.0
       vegfra = 40.0
       chs    = 0.018
       tbot   = 291.0
       mavail = 0.45
       soilt  = 292.2
       sm1    = 0.15
       sm2    = 0.20
    END SELECT
  END SUBROUTINE config_regime

  !==========================================================================
  SUBROUTINE emit_savepoint()
    INTEGER :: ii, kk
    WRITE(*,'(A,I0)') 'CASE=', 1
    WRITE(*,'(A,A)')  'PRECISION_MODE=', TRIM(precision_mode)
    WRITE(*,'(A,I0)') 'N=', NREG
    WRITE(*,'(A,I0)') 'NSL=', NSL
    WRITE(*,'(A,I0)') 'NSTEPS=', NSTEPS
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0

    DO ii = 1, NREG
       ! --- regime label as REGIME_NAME_<i> scalar ---
       WRITE(*,'(A,I0,A,A)') 'REGIME_NAME_', ii, '=', TRIM(R_NAME(ii))
       ! --- INPUTS ---
       CALL dump('DT',       ii, R_DT(ii))
       CALL dumpi('IVGTYP',  ii, R_IVGTYP(ii))
       CALL dumpi('ISLTYP',  ii, R_ISLTYP(ii))
       CALL dump('GSW',      ii, R_SOLDN_GSW(ii))
       CALL dump('GLW',      ii, R_GLW(ii))
       CALL dump('EMISS_IN', ii, R_EMISS(ii))
       CALL dump('TABS',     ii, R_TABS(ii))
       CALL dump('QV',       ii, R_QV(ii))
       CALL dump('QC',       ii, R_QC(ii))
       CALL dump('RHO',      ii, R_RHO(ii))
       CALL dump('P8W',      ii, R_P8W(ii))
       CALL dump('Z3D',      ii, R_Z3D(ii))
       CALL dump('RAINBL',   ii, R_RAINBL(ii))
       CALL dump('VEGFRA',   ii, R_VEGFRA(ii))
       CALL dump('CHS',      ii, R_CHS(ii))
       CALL dump('FLHC',     ii, R_FLHC(ii))
       CALL dump('FLQC',     ii, R_FLQC(ii))
       CALL dump('TBOT',     ii, R_TBOT(ii))
       CALL dump('XLAND',    ii, R_XLAND(ii))
       CALL dump('MAVAIL_IN',ii, R_MAVAIL0(ii))
       CALL dump('SOILT_IN', ii, R_SOILT0(ii))
       CALL dump('SOILMOIS_IN_1', ii, R_SOILMOIS0_1(ii))
       CALL dump('TSO_IN_1', ii, R_TSO0_1(ii))
       ! --- OUTPUTS (scalars) ---
       CALL dump('SOILT',    ii, O_SOILT(ii))
       CALL dump('HFX',      ii, O_HFX(ii))
       CALL dump('QFX',      ii, O_QFX(ii))
       CALL dump('LH',       ii, O_LH(ii))
       CALL dump('GRDFLX',   ii, O_GRDFLX(ii))
       CALL dump('SMAVAIL',  ii, O_SMAVAIL(ii))
       CALL dump('SMMAX',    ii, O_SMMAX(ii))
       CALL dump('SFCRUNOFF',ii, O_SFCRUNOFF(ii))
       CALL dump('UDRUNOFF', ii, O_UDRUNOFF(ii))
       CALL dump('QSFC',     ii, O_QSFC(ii))
       CALL dump('QSG',      ii, O_QSG(ii))
       CALL dump('QVG',      ii, O_QVG(ii))
       CALL dump('QCG',      ii, O_QCG(ii))
       CALL dump('DEW',      ii, O_DEW(ii))
       CALL dump('SNOW',     ii, O_SNOW(ii))
       CALL dump('SNOWH',    ii, O_SNOWH(ii))
       CALL dump('ALB',      ii, O_ALB(ii))
       CALL dump('EMISS',    ii, O_EMISS(ii))
       CALL dump('ZNT',      ii, O_ZNT(ii))
       CALL dump('LAI',      ii, O_LAI(ii))
       CALL dump('MAVAIL',   ii, O_MAVAIL(ii))
       CALL dump('SFCEVP',   ii, O_SFCEVP(ii))
       CALL dump('SFCEXC',   ii, O_SFCEXC(ii))
       CALL dump('TSNAV',    ii, O_TSNAV(ii))
       CALL dump('SOILT1',   ii, O_SOILT1(ii))
       ! --- OUTPUT soil-layer profiles ---
       DO kk = 1, NSL
          CALL dump2('TSO',      ii, kk, O_TSO(ii,kk))
          CALL dump2('SOILMOIS', ii, kk, O_SOILMOIS(ii,kk))
          CALL dump2('SH2O',     ii, kk, O_SH2O(ii,kk))
       END DO
    END DO
  END SUBROUTINE emit_savepoint

  SUBROUTINE dump(name, idx, value)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx
    REAL, INTENT(IN) :: value
    WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name),'[',idx,']=', value
  END SUBROUTINE dump

  SUBROUTINE dumpi(name, idx, ivalue)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx, ivalue
    WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name),'[',idx,']=', REAL(ivalue)
  END SUBROUTINE dumpi

  ! soil-layer value: NAME[i][k]=value
  SUBROUTINE dump2(name, idx, kdx, value)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx, kdx
    REAL, INTENT(IN) :: value
    WRITE(*,'(A,A,I0,A,I0,A,ES23.15)') TRIM(name),'[',idx,'][',kdx,']=', value
  END SUBROUTINE dump2

END PROGRAM ruclsm_oracle
