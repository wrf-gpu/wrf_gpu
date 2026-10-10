! =====================================================================
! v0.23 F2 single-column NSSL 2-moment (WRF mp_physics=18) oracle driver.
!
! Drives the UNMODIFIED WRF NSSL 2-moment scheme
! (phys/module_mp_nssl_2mom.F) exactly as WRF invokes it for the DEFAULT
! mp_physics=18 configuration:
!
!   nssl_2mom_init(...)   mirrors phys/module_physics_init.F:4632-4678
!       nssl_params(1..15) from the Registry namelist defaults
!       (Registry/Registry.EM_COMMON:2410-2428), ipctmp=5 (2-moment,
!       nssl_2moment_on=1, nssl_3moment=0), mixphase=0 (dead argument),
!       nssl_density_on=.T. (nssl_density_on=2: graupel+hail volume),
!       nssl_hail_on=.T., nssl_ccn_on=.T., icdx=6, icdxhl=6,
!       ccn_is_ccna=0 on entry (init flips it to 1 because the module
!       default irenuc=5 predicts activated CCN; module line 1506-1514).
!       Default resolution of the -1 namelist flags for mp_physics=18 is
!       share/module_check_a_mundo.F:3449-3472 (ccn_on=1, 2moment_on=1,
!       hail_on=1, density_on=2).
!
!   nssl_2mom_driver(...) mirrors phys/module_microphysics_driver.F:
!       2237-2303 (CASE (NSSL_2MOM)). NOTE THE NAME MAPPING:
!       WRF qg (moist) -> NSSL argument QH  = GRAUPEL
!       WRF qh (moist) -> NSSL argument QHL = HAIL
!       scalar qndrop->CCW, qnr->CRW, qni->CCI, qns->CSW, qng->CHW,
!       qnh->CHL, qnn->CN (CCN; with irenuc=5 the field is re-purposed
!       as ACTIVATED CCN "CCNA" inside the scheme), qvolg->VHW,
!       qvolh->VHL. All number fields are number MIXING RATIOS (#/kg);
!       volume fields are m3/kg (Registry.EM_COMMON:519-546).
!
! The 6-case sounding generator is copied from
! proofs/v060/oracle/morrison_oracle_driver.f90 (same columns, same
! hydrometeor seeds). Morrison QG seeds NSSL graupel (QH); NSSL hail
! (QHL) starts at zero (no Morrison analogue; the scheme grows hail from
! graupel via wet growth, ihlcnh=3). Extra NSSL fields are initialized
! consistently:
!   CCW (droplet number)  = MIN( QC/cwmas9, qccn_bg ), cwmas9 =
!       1000*(pi/6)*(2*9e-6)^3 kg = mass of a 9-micron-radius droplet --
!       the same mean droplet mass the scheme itself assumes when
!       deriving Nc from qc at cold start (calcnfromq, module line 5340,
!       cwmas09), capped by the background CCN mixing ratio
!       qccn_bg = nssl_cccn/rho00 = 0.5e9/1.225 (module lines 907, 2121).
!   CRW = QR/5.0E-9, CCI = QI/1.0E-10, CSW = QS/2.0E-8, CHW = QH/5.0E-8
!       (identical mean-mass seeding to the Morrison oracle's NR/NI/NS/NG).
!   CHL = 0, VHL = 0 (consistent with QHL = 0).
!   VHW (graupel volume) = QH/500. (rho_qh init default, so the seeded
!       graupel starts exactly at its reference density).
!   CN  = 0. With the default irenuc=5 config the driver itself zeroes
!       the CN field at itimestep==1 (module lines 2717-2738: the qnn
!       array holds activated CCN and starts from zero at cold start;
!       the internal background CCN is the constant qccn).
!
! ITIMESTEP=1 semantics (cold start, exactly like WRF's first step):
!   - CN zeroed (see above),
!   - calcnfromq (module line 3115-3117) runs: where a number (or
!     volume) field is ~zero but mass is significant it derives the
!     number from single-moment assumptions; where mass is below qxmin
!     the species is folded back to vapor. With the consistent seeds
!     above the derivation branches are inert except on trace cells.
!
! Usage: ./nssl2mom_oracle <case_id>
! Output: flat key=value text dump on stdout (parsed by dump_to_json.py).
! =====================================================================
! ---------------------------------------------------------------------
! v0.3.4 lane o1-nssl EXTENSION (proofs/v034/oracle/nssl2mom):
!   cases 1..6  : byte-identical setup to the v023 oracle (itimestep=1, DT=60)
!   cases 7..14 : warm-start (itimestep=2, CN = activated CCN = CCW where cloud)
!                 and hail-seeded columns, DT 18/54/60 s, see case table below.
!   Optional argv(2) = 'dumpmv' -> print every module variable after init
!   (requires the print-only instrumented module copy).
! ---------------------------------------------------------------------
PROGRAM nssl2mom_oracle
  USE module_mp_nssl_2mom, ONLY : nssl_2mom_init, nssl_2mom_driver, qccn
#ifdef O1_INSTR
  USE module_mp_nssl_2mom, ONLY : o1_dump_module_vars
#endif
  IMPLICIT NONE

  ! WRF model constants used to build the column (share/module_model_constants.F)
  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: P1000  = 1.0E5
  REAL, PARAMETER :: ROVCP  = R_D/CP

  ! NSSL-consistent seeding constants (see header)
  REAL, PARAMETER :: CWMAS9   = 1000.0*0.523599*(2.0*9.0E-6)**3 ! kg, 9-um-radius droplet (calcnfromq cwmas09)
  REAL, PARAMETER :: QCCN_BG  = 0.5E9/1.225                     ! #/kg, nssl_cccn/rho00
  REAL, PARAMETER :: RHO_QH0  = 500.0                           ! kg/m3, nssl_rho_qh default (graupel)

  INTEGER, PARAMETER :: KX  = 40
  INTEGER, PARAMETER :: ids=1,ide=2, jds=1,jde=2, kds=1,kde=KX+1
  INTEGER, PARAMETER :: ims=1,ime=1, jms=1,jme=1, kms=1,kme=KX
  INTEGER, PARAMETER :: its=1,ite=1, jts=1,jte=1, kts=1,kte=KX

  ! 3D (1,KX,1) arrays for the driver (i,k,j orientation)
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: TH,QV,QC,QR,QI,QS,QH,QHL
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: CCW,CRW,CCI,CSW,CHW,CHL
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: CN,VHW,VHL
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: PII,P,DZ,W,DN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: DBZ,RE_C,RE_I,RE_S
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: QRCUTEN,QSCUTEN,QICUTEN,QCCUTEN
  REAL, DIMENSION(ims:ime,jms:jme) :: RAINNC,RAINNCV,SNOWNC,SNOWNCV
  REAL, DIMENSION(ims:ime,jms:jme) :: GRPLNC,GRPLNCV,HAILNC,HAILNCV,SR
  REAL, DIMENSION(ims:ime,jms:jme) :: HAILMXK1,HAILMX2D

  ! saved input copies (column, k=1..KX)
  REAL, DIMENSION(KX) :: TH0,QV0,QC0,QR0,QI0,QS0,QH0,QHL0
  REAL, DIMENSION(KX) :: CCW0,CRW0,CCI0,CSW0,CHW0,CHL0,CN0,VHW0,VHL0
  REAL, DIMENSION(KX) :: PII0,P0,DZ0,W0,DN0

  REAL, DIMENSION(20) :: nssl_params
  REAL :: DT
  INTEGER :: k, case_id, itimestep, ccn_is_ccna
  INTEGER :: thermo_id, hail_seed, cn_mode
  CHARACTER(LEN=32) :: arg, arg2

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 60.0          ! representative microphysics dt (s)
  itimestep = 1
  thermo_id = case_id
  hail_seed = 0
  cn_mode   = 0
  ! v034 extension case table (cases 1..6 unchanged)
  SELECT CASE (case_id)
  CASE (7)   ! hail-seeded convective core (case-4 thermodynamics), cold start
    thermo_id = 4; hail_seed = 1
  CASE (8)   ! case 4 warm start, PROD d02 dt
    thermo_id = 4; itimestep = 2; DT = 18.0; cn_mode = 1
  CASE (9)   ! case 2 (mixed phase, melting) warm start
    thermo_id = 2; itimestep = 2; DT = 18.0; cn_mode = 1
  CASE (10)  ! case 3 (cold ice/snow) warm start
    thermo_id = 3; itimestep = 2; DT = 18.0; cn_mode = 1
  CASE (11)  ! case 1 (warm rain) warm start, PROD d01 dt
    thermo_id = 1; itimestep = 2; DT = 54.0; cn_mode = 1
  CASE (12)  ! case 5 (evaporation/sublimation into dry air) warm start
    thermo_id = 5; itimestep = 2; DT = 54.0; cn_mode = 1
  CASE (13)  ! hail-seeded warm start, dt 18
    thermo_id = 4; hail_seed = 1; itimestep = 2; DT = 18.0; cn_mode = 1
  CASE (14)  ! hail-seeded warm start, warm surface melting hail, dt 54
    thermo_id = 1; hail_seed = 2; itimestep = 2; DT = 54.0; cn_mode = 1
  CASE (15)  ! supercooled rain + cloud in the cold case-3 column (Bigg rain/droplet freezing), warm start
    thermo_id = 3; hail_seed = 3; itimestep = 2; DT = 18.0; cn_mode = 1
  CASE (16)  ! large graupel in dense supercooled cloud (wet growth, graupel->hail conversion), warm start
    thermo_id = 4; hail_seed = 4; itimestep = 2; DT = 18.0; cn_mode = 1
  CASE (17)  ! small hail/graupel/snow particles sublimating in dry air, dt 60 (depletion limiters)
    thermo_id = 5; hail_seed = 5; itimestep = 2; DT = 60.0; cn_mode = 1
  END SELECT

  ! ---- init: mirrors module_physics_init.F:4642-4678 with the Registry
  !      namelist defaults (Registry.EM_COMMON:2410-2428) ----
  nssl_params(:)  = 0.0
  nssl_params(1)  = 0.5E9   ! nssl_cccn   (base CCN conc, #/m3)
  nssl_params(2)  = 0.0     ! nssl_alphah (graupel shape)
  nssl_params(3)  = 1.0     ! nssl_alphahl (hail shape)
  nssl_params(4)  = 4.0E5   ! nssl_cnoh   (graupel intercept)
  nssl_params(5)  = 4.0E4   ! nssl_cnohl  (hail intercept)
  nssl_params(6)  = 8.0E5   ! nssl_cnor   (rain intercept)
  nssl_params(7)  = 3.0E6   ! nssl_cnos   (snow intercept)
  nssl_params(8)  = 500.0   ! nssl_rho_qh (graupel density)
  nssl_params(9)  = 900.0   ! nssl_rho_qhl (hail density)
  nssl_params(10) = 100.0   ! nssl_rho_qs (snow density)
  nssl_params(11) = 0.0     ! ipelec (elec_physics off)
  nssl_params(12) = 12.0    ! nssl_isaund (registry.elec default; unused by init)
  ccn_is_ccna     = 0       ! Registry default; init sets it to 1 (irenuc=5)

  CALL nssl_2mom_init( nssl_params=nssl_params, ipctmp=5, mixphase=0,  &
       nssl_density_on=.TRUE., nssl_hail_on=.TRUE., nssl_ccn_on=.TRUE., &
       nssl_icdx=6, nssl_icdxhl=6, ccn_is_ccna=ccn_is_ccna )
#ifdef O1_INSTR
  IF (COMMAND_ARGUMENT_COUNT() >= 2) THEN
    CALL GET_COMMAND_ARGUMENT(2, arg2)
    IF (TRIM(arg2) == 'dumpmv') CALL o1_dump_module_vars()
  END IF
#endif

  ! ---- build the chosen column ----
  CALL build_column(thermo_id, TH, QV, QC, QR, QI, QS, QH, QHL,        &
                    CCW, CRW, CCI, CSW, CHW, CHL, CN, VHW, VHL,        &
                    DN, PII, P, DZ, W)
  CALL extend_column(hail_seed, cn_mode, DZ, QHL, CHL, VHL, CN, CCW, QR, CRW, QC, QH, CHW, VHW, QS, CSW)
  ! knife-edge probe (lane o1-nssl, dev only): scale QC at Fortran level 13 by (1+O1_EPS)
  BLOCK
    CHARACTER(LEN=64) :: ev
    INTEGER :: ln, st
    REAL :: eps
    CALL GET_ENVIRONMENT_VARIABLE('O1_EPS', ev, ln, st)
    IF (st == 0 .and. ln > 0) THEN
      READ(ev,*) eps
      QC(1,13,1) = QC(1,13,1)*(1.0+eps)
    END IF
  END BLOCK

  ! save inputs
  DO k = 1, KX
    TH0(k)=TH(1,k,1);  QV0(k)=QV(1,k,1);  QC0(k)=QC(1,k,1);  QR0(k)=QR(1,k,1)
    QI0(k)=QI(1,k,1);  QS0(k)=QS(1,k,1);  QH0(k)=QH(1,k,1);  QHL0(k)=QHL(1,k,1)
    CCW0(k)=CCW(1,k,1); CRW0(k)=CRW(1,k,1); CCI0(k)=CCI(1,k,1)
    CSW0(k)=CSW(1,k,1); CHW0(k)=CHW(1,k,1); CHL0(k)=CHL(1,k,1)
    CN0(k)=CN(1,k,1);   VHW0(k)=VHW(1,k,1); VHL0(k)=VHL(1,k,1)
    PII0(k)=PII(1,k,1); P0(k)=P(1,k,1); DZ0(k)=DZ(1,k,1); W0(k)=W(1,k,1)
    DN0(k)=DN(1,k,1)
  END DO

  DBZ = 0.0; RE_C = 0.0; RE_I = 0.0; RE_S = 0.0
  QRCUTEN = 0.0; QSCUTEN = 0.0; QICUTEN = 0.0; QCCUTEN = 0.0
  RAINNC=0.; RAINNCV=0.; SNOWNC=0.; SNOWNCV=0.
  GRPLNC=0.; GRPLNCV=0.; HAILNC=0.; HAILNCV=0.; SR=0.
  HAILMXK1=0.; HAILMX2D=0.

  ! ---- call the real NSSL 2-moment driver, mirroring
  !      module_microphysics_driver.F:2237-2303 (chem/3-moment/ssat
  !      optionals inactive by default and omitted; equivalent because
  !      ipconc=5 => lzr=lzh=lzhl=0 and nssl_ssat_output=0) ----
  CALL nssl_2mom_driver(                                            &
       ITIMESTEP=itimestep,                                         &
       TH=TH, QV=QV, QC=QC, QR=QR, QI=QI, QS=QS,                    &
       QH=QH, QHL=QHL,                                              &
       CCW=CCW, CRW=CRW, CCI=CCI, CSW=CSW, CHW=CHW, CHL=CHL,        &
       VHW=VHW, F_VHW=.TRUE., VHL=VHL, F_VHL=.TRUE.,                &
       CN=CN, F_CN=.TRUE.,                                          &
       PII=PII, P=P, W=W, DZ=DZ, DTP=DT, DN=DN,                     &
       RAINNC=RAINNC, RAINNCV=RAINNCV,                              &
       SNOWNC=SNOWNC, SNOWNCV=SNOWNCV,                              &
       HAILNC=HAILNC, HAILNCV=HAILNCV,                              &
       GRPLNC=GRPLNC, GRPLNCV=GRPLNCV,                              &
       SR=SR,                                                       &
       DBZ=DBZ,                                                     &
       NSSL_SSAT_OUTPUT=0,                                          &
       NSSL_PROGN=.FALSE.,                                          &
       DIAGFLAG=.TRUE., KE_DIAG=KX,                                 &
       CU_USED=0,                                                   &
       QRCUTEN=QRCUTEN, QSCUTEN=QSCUTEN,                            &
       QICUTEN=QICUTEN, QCCUTEN=QCCUTEN,                            &
       RE_CLOUD=RE_C, RE_ICE=RE_I, RE_SNOW=RE_S,                    &
       HAS_REQC=1, HAS_REQI=1, HAS_REQS=1,                          &
       HAIL_MAXK1=HAILMXK1, HAIL_MAX2D=HAILMX2D,                    &
       NWP_DIAGNOSTICS=0,                                           &
       IDS=ids,IDE=ide, JDS=jds,JDE=jde, KDS=kds,KDE=kde,           &
       IMS=ims,IME=ime, JMS=jms,JME=jme, KMS=kms,KME=kme,           &
       ITS=its,ITE=ite, JTS=jts,JTE=jte, KTS=kts,KTE=kte )

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,I0)') 'ITIMESTEP=', itimestep
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  WRITE(*,'(A,I0)') 'CCN_IS_CCNA=', ccn_is_ccna
  WRITE(*,'(A,ES23.15)') 'QCCN=', qccn
  ! inputs
  CALL dump_col('TH_IN',  TH0)
  CALL dump_col('QV_IN',  QV0)
  CALL dump_col('QC_IN',  QC0)
  CALL dump_col('QR_IN',  QR0)
  CALL dump_col('QI_IN',  QI0)
  CALL dump_col('QS_IN',  QS0)
  CALL dump_col('QH_IN',  QH0)   ! graupel (WRF qg)
  CALL dump_col('QHL_IN', QHL0)  ! hail    (WRF qh)
  CALL dump_col('CCW_IN', CCW0)  ! droplet number (WRF qndrop)
  CALL dump_col('CRW_IN', CRW0)  ! rain number    (WRF qnr)
  CALL dump_col('CCI_IN', CCI0)  ! ice number     (WRF qni)
  CALL dump_col('CSW_IN', CSW0)  ! snow number    (WRF qns)
  CALL dump_col('CHW_IN', CHW0)  ! graupel number (WRF qng)
  CALL dump_col('CHL_IN', CHL0)  ! hail number    (WRF qnh)
  CALL dump_col('CN_IN',  CN0)   ! CCN/CCNA       (WRF qnn)
  CALL dump_col('VHW_IN', VHW0)  ! graupel volume (WRF qvolg)
  CALL dump_col('VHL_IN', VHL0)  ! hail volume    (WRF qvolh)
  CALL dump_col('PII',    PII0)
  CALL dump_col('P',      P0)
  CALL dump_col('DZ',     DZ0)
  CALL dump_col('W',      W0)
  CALL dump_col('DN',     DN0)
  ! outputs (post-NSSL)
  CALL dump_col3('TH_OUT',  TH)
  CALL dump_col3('QV_OUT',  QV)
  CALL dump_col3('QC_OUT',  QC)
  CALL dump_col3('QR_OUT',  QR)
  CALL dump_col3('QI_OUT',  QI)
  CALL dump_col3('QS_OUT',  QS)
  CALL dump_col3('QH_OUT',  QH)
  CALL dump_col3('QHL_OUT', QHL)
  CALL dump_col3('CCW_OUT', CCW)
  CALL dump_col3('CRW_OUT', CRW)
  CALL dump_col3('CCI_OUT', CCI)
  CALL dump_col3('CSW_OUT', CSW)
  CALL dump_col3('CHW_OUT', CHW)
  CALL dump_col3('CHL_OUT', CHL)
  CALL dump_col3('CN_OUT',  CN)
  CALL dump_col3('VHW_OUT', VHW)
  CALL dump_col3('VHL_OUT', VHL)
  ! diagnostics
  CALL dump_col3('DBZ_OUT',      DBZ)
  CALL dump_col3('RE_CLOUD_OUT', RE_C)
  CALL dump_col3('RE_ICE_OUT',   RE_I)
  CALL dump_col3('RE_SNOW_OUT',  RE_S)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=', RAINNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNCV=', SNOWNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'GRPLNCV=', GRPLNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'HAILNCV=', HAILNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'SR=',      SR(1,1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(KX), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = 1, KX
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col3(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = 1, KX
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col3

  ! ------------------------------------------------------------------
  ! Predeclared single columns -- COPIED from the Morrison oracle
  ! (proofs/v060/oracle/morrison_oracle_driver.f90). Same thermodynamic
  ! profiles, same hydrometeor seeds (Morrison QG -> NSSL graupel QH).
  ! case_id:
  !  1 = warm moist BL, supersaturated low levels
  !  2 = deep mixed-phase with melting layer
  !  3 = cold ice/snow column (all subfreezing), ice-supersaturated
  !  4 = graupel-dominant convective core
  !  5 = subsaturated mid-level with rain/snow/graupel falling into dry air
  !  6 = clean column, slight liquid supersaturation, trace cloud only
  ! Bottom-up index (k=1 lowest model layer). 3D index (1,k,1).
  ! ------------------------------------------------------------------
  SUBROUTINE build_column(cid, tth, qqv, qqc, qqr, qqi, qqs, qqh, qqhl, &
                          nnc, nnr, nni, nns, nnh, nnhl, cnn, vvh, vvhl, &
                          ddrho, eexner, ppr, ddz, ww)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: &
         tth,qqv,qqc,qqr,qqi,qqs,qqh,qqhl,nnc,nnr,nni,nns,nnh,nnhl, &
         cnn,vvh,vvhl,ddrho,eexner,ppr,ddz,ww
    REAL :: psfc, tsfc, theta_sfc, ztop
    REAL :: th_k, t_k, p_k, tv_k, z_k, es, qsw, rh_k
    REAL :: lapse, rh_ml, rh_trop, zml, wmax
    REAL, DIMENSION(KX) :: zz
    INTEGER :: kk

    ztop = 16000.0
    DO kk = 1, KX
      zz(kk) = ztop * ( (REAL(kk)-0.5)/REAL(KX) )**1.15
    END DO

    SELECT CASE (cid)
    CASE (1)   ! warm moist BL, supersaturated low levels
      psfc=1000.0E2; tsfc=298.0; zml=1500.0; lapse=5.0E-3; rh_ml=1.02; rh_trop=0.40; wmax=1.0
    CASE (2)   ! deep mixed-phase with melting layer
      psfc=1000.0E2; tsfc=287.0; zml=600.0;  lapse=6.0E-3; rh_ml=0.98; rh_trop=0.60; wmax=2.0
    CASE (3)   ! cold ice/snow column, ice-supersaturated
      psfc=850.0E2;  tsfc=258.0; zml=400.0;  lapse=5.5E-3; rh_ml=1.05; rh_trop=0.70; wmax=0.5
    CASE (4)   ! graupel-dominant convective core
      psfc=1000.0E2; tsfc=296.0; zml=1000.0; lapse=6.5E-3; rh_ml=1.00; rh_trop=0.65; wmax=5.0
    CASE (5)   ! subsaturated mid-level, falling rain/snow/graupel
      psfc=950.0E2;  tsfc=283.0; zml=300.0;  lapse=6.0E-3; rh_ml=0.55; rh_trop=0.30; wmax=0.5
    CASE (6)   ! clean column, slight supersaturation
      psfc=1000.0E2; tsfc=295.0; zml=2000.0; lapse=5.0E-3; rh_ml=1.01; rh_trop=0.50; wmax=0.5
    CASE DEFAULT
      psfc=1000.0E2; tsfc=295.0; zml=1000.0; lapse=5.5E-3; rh_ml=0.95; rh_trop=0.50; wmax=1.0
    END SELECT

    theta_sfc = tsfc * (P1000/psfc)**ROVCP
    p_k = psfc
    DO kk = 1, KX
      z_k = zz(kk)
      IF (z_k <= zml) THEN
        th_k = theta_sfc
        rh_k = rh_ml
      ELSE
        th_k = theta_sfc + lapse*(z_k - zml)
        rh_k = rh_trop + (rh_ml-rh_trop)*EXP(-(z_k-zml)/3000.0)
      END IF
      IF (kk == 1) THEN
        t_k  = th_k*(psfc/P1000)**ROVCP
        tv_k = t_k
        p_k  = psfc * EXP(-G*zz(1)/(R_D*tv_k))
      ELSE
        t_k  = th_k*(p_k/P1000)**ROVCP
        tv_k = t_k*(1.0+0.608*qqv(1,kk-1,1))
        p_k  = p_k * EXP(-G*(zz(kk)-zz(kk-1))/(R_D*tv_k))
      END IF
      t_k = th_k*(p_k/P1000)**ROVCP
      ! saturation vapor pressure (Tetens, liquid) for the RH target
      es  = 610.78*EXP(17.27*(t_k-273.15)/(t_k-35.86))
      qsw = 0.622*es/(p_k-es)
      qqv(1,kk,1)  = MAX(rh_k*qsw, 1.0E-8)
      tv_k = t_k*(1.0+0.608*qqv(1,kk,1))
      eexner(1,kk,1) = (p_k/P1000)**ROVCP
      tth(1,kk,1)    = t_k/eexner(1,kk,1)          ! potential temperature
      ppr(1,kk,1)    = p_k
      ! DN: air density as WRF's phy_prep computes the rho passed to the
      ! microphysics driver (module_big_step_utilities_em.F:4856,
      ! rho = rho_dry*(1+qv) = p/(R_d*Tv) to eq-of-state accuracy).
      ! Unlike Morrison, NSSL USES this (number-conc scaling, fallout).
      ddrho(1,kk,1)  = p_k/(R_D*tv_k)
      IF (kk == 1) THEN
        ddz(1,kk,1) = 2.0*zz(1)
      ELSE
        ddz(1,kk,1) = zz(kk)-zz(kk-1)
      END IF
      ww(1,kk,1)   = wmax*EXP(-((z_k-zml-1000.0)/3000.0)**2)
    END DO

    ! Seed hydrometeors identically to the Morrison oracle (QG -> QH
    ! graupel); NSSL-only hail (QHL/CHL/VHL) starts at zero.
    qqc=0.; qqr=0.; qqi=0.; qqs=0.; qqh=0.; qqhl=0.
    nnc=0.; nnr=0.; nni=0.; nns=0.; nnh=0.; nnhl=0.
    cnn=0.; vvh=0.; vvhl=0.
    DO kk = 1, KX
      z_k = zz(kk)
      SELECT CASE (cid)
      CASE (1)   ! warm: cloud + a little rain low
        IF (z_k < 3000.0)  qqc(1,kk,1) = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0)  qqr(1,kk,1) = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
      CASE (2)   ! mixed-phase: cloud low, rain mid, ice/snow/graupel aloft
        IF (z_k < 4000.0)  qqc(1,kk,1) = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k < 5000.0)  qqr(1,kk,1) = 6.0E-4*EXP(-((z_k-1200.0)/1500.0)**2)
        IF (z_k > 4000.0)  qqi(1,kk,1) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 3500.0)  qqs(1,kk,1) = 8.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        IF (z_k > 3500.0)  qqh(1,kk,1) = 5.0E-4*EXP(-((z_k-5500.0)/2500.0)**2)
      CASE (3)   ! cold: ice + snow aloft
        qqi(1,kk,1) = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qqs(1,kk,1) = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
        IF (z_k > 7000.0) qqh(1,kk,1) = 1.0E-4*EXP(-((z_k-9000.0)/2500.0)**2)
      CASE (4)   ! graupel-dominant convective core
        IF (z_k < 5000.0)  qqc(1,kk,1) = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        IF (z_k < 6000.0)  qqr(1,kk,1) = 1.5E-3*EXP(-((z_k-2500.0)/2000.0)**2)
        IF (z_k > 4000.0)  qqi(1,kk,1) = 4.0E-4*EXP(-((z_k-7500.0)/2500.0)**2)
        qqs(1,kk,1) = 1.0E-3*EXP(-((z_k-6000.0)/3000.0)**2)
        qqh(1,kk,1) = 2.0E-3*EXP(-((z_k-6000.0)/2500.0)**2)
      CASE (5)   ! subsaturated: rain + snow + graupel falling into dry air
        IF (z_k < 6000.0)  qqr(1,kk,1) = 7.0E-4*EXP(-((z_k-3000.0)/2000.0)**2)
        qqs(1,kk,1) = 6.0E-4*EXP(-((z_k-5000.0)/2500.0)**2)
        IF (z_k > 5000.0) qqh(1,kk,1) = 2.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
      CASE (6)   ! clean: trace cloud only -> condensation path
        IF (z_k < 3000.0)  qqc(1,kk,1) = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
      END SELECT

      ! Number concentrations consistent with the seeded mass, using the
      ! Morrison-oracle mean particle masses for the shared classes and
      ! the NSSL cold-start droplet mass/CCN cap for cloud water.
      nnc(1,kk,1) = MIN( qqc(1,kk,1)/CWMAS9, QCCN_BG )
      nnr(1,kk,1) = qqr(1,kk,1)/5.0E-9
      nni(1,kk,1) = qqi(1,kk,1)/1.0E-10
      nns(1,kk,1) = qqs(1,kk,1)/2.0E-8
      nnh(1,kk,1) = qqh(1,kk,1)/5.0E-8
      ! Graupel particle volume mixing ratio at the reference graupel
      ! density (nssl_rho_qh=500), so seeded graupel starts on-density.
      vvh(1,kk,1) = qqh(1,kk,1)/RHO_QH0
    END DO
  END SUBROUTINE build_column

  ! ------------------------------------------------------------------
  ! v034 extension seeds (applied after build_column):
  !  hail_seed=1: hail 2.0e-3*exp(-((z-5500)/2000)**2) kg/kg (straddles the
  !               melting level of the case-4 sounding);
  !  hail_seed=2: hail 1.5e-3*exp(-((z-2500)/1500)**2) (warm case-1 sounding:
  !               melting/shedding/evaporation below 0C level).
  !  hail_seed=3: no hail; supercooled rain 8e-4*exp(-((z-3000)/1500)**2) (+qnr at 5e-9 kg mean mass)
  !               and cloud 4e-4*exp(-((z-2500)/1200)**2) (+qndrop at the 9-um mass, CCN cap) in the
  !               case-3 column (T < -18 C aloft): Bigg rain + droplet freezing paths.
  !  hail_seed=4: graupel 4e-3*exp(-((z-6000)/1200)**2) at 3.3e-5 kg mean mass (~5 mm at 500 kg/m3),
  !               volume at 500 kg/m3, supercooled cloud 2.5e-3*exp(-((z-6000)/1500)**2) (+qndrop).
  !  hail_seed=5: small hail 3e-5*exp(-((z-4000)/2000)**2) at 1e-7 kg, graupel 2e-5*exp(-((z-4500)/2000)**2)
  !               at 1e-8 kg (volume at 400), snow 2e-5*exp(-((z-5000)/2000)**2) at 1e-11 kg.
  !  mean hail particle mass 6.0e-5 kg (~5 mm at 900 kg/m3); volume at 900.
  !  cn_mode=1  : warm start: CN (activated CCN, #/kg) = CCW where cloud.
  ! ------------------------------------------------------------------
  SUBROUTINE extend_column(hseed, cmode, ddz, qqhl, nnhl, vvhl, cnn, nnc, qqr, nnr, qqc, qqh, nnh, vvh, qqs, nns)
    INTEGER, INTENT(IN) :: hseed, cmode
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: ddz
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(INOUT) :: qqhl, nnhl, vvhl, cnn, nnc, qqr, nnr, qqc
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(INOUT) :: qqh, nnh, vvh, qqs, nns
    REAL :: z_k
    INTEGER :: kk
    z_k = 0.0
    DO kk = 1, KX
      IF (kk == 1) THEN
        z_k = 0.5*ddz(1,1,1)
      ELSE
        z_k = z_k + ddz(1,kk,1)
      END IF
      IF (hseed == 1) qqhl(1,kk,1) = 2.0E-3*EXP(-((z_k-5500.0)/2000.0)**2)
      IF (hseed == 2) qqhl(1,kk,1) = 1.5E-3*EXP(-((z_k-2500.0)/1500.0)**2)
      IF (hseed == 3) THEN
        qqr(1,kk,1) = 8.0E-4*EXP(-((z_k-3000.0)/1500.0)**2)
        nnr(1,kk,1) = qqr(1,kk,1)/5.0E-9
        qqc(1,kk,1) = 4.0E-4*EXP(-((z_k-2500.0)/1200.0)**2)
        nnc(1,kk,1) = MIN( qqc(1,kk,1)/CWMAS9, QCCN_BG )
      END IF
      IF (hseed == 4) THEN
        qqh(1,kk,1) = 4.0E-3*EXP(-((z_k-6000.0)/1200.0)**2)
        nnh(1,kk,1) = qqh(1,kk,1)/3.3E-5
        vvh(1,kk,1) = qqh(1,kk,1)/500.0
        qqc(1,kk,1) = 2.5E-3*EXP(-((z_k-6000.0)/1500.0)**2)
        nnc(1,kk,1) = MIN( qqc(1,kk,1)/CWMAS9, QCCN_BG )
      END IF
      IF (hseed == 5) THEN
        qqhl(1,kk,1) = 3.0E-5*EXP(-((z_k-4000.0)/2000.0)**2)
        nnhl(1,kk,1) = qqhl(1,kk,1)/1.0E-7
        vvhl(1,kk,1) = qqhl(1,kk,1)/900.0
        qqh(1,kk,1) = 2.0E-5*EXP(-((z_k-4500.0)/2000.0)**2)
        nnh(1,kk,1) = qqh(1,kk,1)/1.0E-8
        vvh(1,kk,1) = qqh(1,kk,1)/400.0
        qqs(1,kk,1) = 2.0E-5*EXP(-((z_k-5000.0)/2000.0)**2)
        nns(1,kk,1) = qqs(1,kk,1)/1.0E-11
      END IF
      IF (hseed == 1 .or. hseed == 2) THEN
        nnhl(1,kk,1) = qqhl(1,kk,1)/6.0E-5
        vvhl(1,kk,1) = qqhl(1,kk,1)/900.0
      END IF
      IF (cmode == 1) cnn(1,kk,1) = nnc(1,kk,1)
    END DO
  END SUBROUTINE extend_column

END PROGRAM nssl2mom_oracle
