! =====================================================================
! v0.23 F2 single-column AEROSOL-AWARE Morrison 2-moment
! (WRF mp_physics=40) oracle driver.
!
! Drives the UNMODIFIED WRF aerosol-aware Morrison scheme
! (phys/module_mp_morr_two_moment_aero.F):
!   MORR_TWO_MOMENT_INIT_AERO(morr_rimed_ice=0)  -> IHAIL=0, graupel mode
!   MP_MORR_TWO_MOMENT_AERO(...)                 -> full 3D->1D->3D wrapper
! exactly as WRF's phys/module_microphysics_driver.F invokes it for
! mp_physics=40 (CASE (MORR_TM_AERO)) WITHOUT WRF-Chem (WRF_CHEM undefined).
!
! With aercu_opt=2 (the aerosol-aware operating point) the wrapper sets
!   INUM=0, iinum=0  -> droplet number NC is PROGNOSTIC (input+output)
!   IACT=4           -> Abdul-Razzak & Ghan (2000) activation from the
!                       prescribed 10-species CESM-style aerosol array AEROCU
!   INUC=2           -> Liu & Penner (2005) aerosol ice nucleation
!                       (sulfate homogeneous + soot/dust heterogeneous
!                        + Meyers mixed-phase)
! and builds per-mode aerosol mass (maerosol) and number (naer) from
! AEROCU (ug/m3, Registry order: dust1..4, seasalt, sulfate, BCphob,
! BCphil, OCphob, OCphil).
!
! The 6-case sounding generator (build_column) is copied AS-IS from the
! v0.6.0 base-Morrison oracle driver (morrison_oracle_driver.f90); only
! the aero-specific inputs (NC, KZH, AEROCU) are added on top (build_aero).
!
! DEVIATIONS FROM THE BASE-MORRISON ORACLE THAT A PORTER MUST KNOW:
!  * kme = KX+1 (base oracle used kme=KX): the aero wrapper computes
!    WVAR(i,k,j) = KZH(i,K+1,j)/20 for K=kts..kte, i.e. it reads the eddy
!    diffusivity one level ABOVE the tile top. WRF always has
!    kme = kde = kte+1, so this is the WRF-faithful memory layout. All 3D
!    state arrays carry one padding level (k=KX+1) that the scheme never
!    reads or writes (only KZH is read there); dumps cover k=1..KX except
!    KZH_IN which is dumped over k=1..KX+1.
!  * WVAR (sub-grid w std-dev used by activation) is KZH/20 clamped to
!    [0.1, 50] m/s -- NOT the constant 0.5 m/s of the base scheme.
!  * NC (droplet number, #/kg) is a 12th prognostic column (in+out).
!  * Extra diagnostic outputs: EFCG/EFIG/EFSG (radiation effective radii,
!    set because aercu_opt>0), WACT (=WVAR+W), CCN1_GS..CCN7_GS (CCN at
!    0.02/0.05/0.1/0.2/0.3/0.5/1.0 % supersaturation, #/m3).
!  * Optional WRF-Chem-only args (F_QNDROP path with qndrop, wetscav_on,
!    rainprod, evapprod, QLSINK, PRECR/I/S/G) are omitted (F_QNDROP passed
!    .FALSE. exactly as WRF does for mp_physics=40 without chem).
!
! Usage: ./morraero_oracle <case_id>
! Output: flat key=value text dump on stdout (parsed by Python into JSON).
! =====================================================================
PROGRAM morraero_oracle
  USE module_mp_morr_two_moment_aero, ONLY : MORR_TWO_MOMENT_INIT_AERO, &
                                             MP_MORR_TWO_MOMENT_AERO
  IMPLICIT NONE

  ! WRF model constants used to build the column (share/module_model_constants.F)
  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: P1000  = 1.0E5
  REAL, PARAMETER :: ROVCP  = R_D/CP

  INTEGER, PARAMETER :: KX  = 40
  ! kme = KX+1 (WRF-faithful; the scheme reads KZH at k+1, see header)
  INTEGER, PARAMETER :: ids=1,ide=2, jds=1,jde=2, kds=1,kde=KX+1
  INTEGER, PARAMETER :: ims=1,ime=1, jms=1,jme=1, kms=1,kme=KX+1
  INTEGER, PARAMETER :: its=1,ite=1, jts=1,jte=1, kts=1,kte=KX
  INTEGER, PARAMETER :: NAERCU = 10   ! no_src_types_cu MUST be 10 (module naer_cu=10)

  ! prescribed-aerosol operating point (see README.md)
  INTEGER, PARAMETER :: AERCU_OPT = 2
  REAL,    PARAMETER :: AERCU_FCT = 1.0
  INTEGER, PARAMETER :: PBL = 1      ! passed but never referenced by the scheme

  ! mean droplet mass for NC seeding: r = 10 um droplet, rho_w = 1000 kg/m3
  REAL, PARAMETER :: DROPMASS = 4.18879E-12

  ! 3D (1,KX+1,1) arrays for the wrapper
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: TH,QV,QC,QR,QI,QS,QG
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: NI,NS,NR,NG,NC
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RHO,PII,P,DZ,W,KZH
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: refl_10cm, mskf_refl_10cm
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: qrcuten,qscuten,qicuten
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: NR_CU,QR_CU,NS_CU,QS_CU
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: EFCG,EFIG,EFSG,WACT
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: CCN1_GS,CCN2_GS,CCN3_GS,CCN4_GS
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: CCN5_GS,CCN6_GS,CCN7_GS
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme,NAERCU) :: aerocu
  REAL, DIMENSION(ims:ime,jms:jme)         :: HT, CU_UAF
  REAL, DIMENSION(ims:ime,jms:jme)         :: RAINNC,RAINNCV,SR
  REAL, DIMENSION(ims:ime,jms:jme)         :: SNOWNC,SNOWNCV,GRAUPELNC,GRAUPELNCV

  ! saved input copies (column, k=1..KX; KZH k=1..KX+1)
  REAL, DIMENSION(KX)   :: TH0,QV0,QC0,QR0,QI0,QS0,QG0,NI0,NS0,NR0,NG0,NC0
  REAL, DIMENSION(KX)   :: PII0,P0,DZ0,W0
  REAL, DIMENSION(KX+1) :: KZH0
  REAL, DIMENSION(KX,NAERCU) :: AER0

  REAL :: DT
  INTEGER :: k, l, case_id, itimestep
  CHARACTER(LEN=32) :: arg

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 60.0          ! representative microphysics dt (s)
  itimestep = 1

  ! ---- initialize Morrison-aero module constants (graupel mode, IHAIL=0) ----
  CALL MORR_TWO_MOMENT_INIT_AERO(0)

  ! ---- build the chosen column (same 6 soundings as the base oracle) ----
  ! Pre-zero so the k=KX+1 padding level of every array is defined.
  TH=0.; QV=0.; QC=0.; QR=0.; QI=0.; QS=0.; QG=0.
  NI=0.; NS=0.; NR=0.; NG=0.; NC=0.
  RHO=1.; PII=1.; P=0.; DZ=1.; W=0.; KZH=0.
  CALL build_column(case_id, TH, QV, QC, QR, QI, QS, QG, NI, NS, NR, NG, &
                     RHO, PII, P, DZ, W)
  ! fill the never-read padding level with a finite copy of the top tile level
  TH (1,KX+1,1)=TH (1,KX,1); QV (1,KX+1,1)=QV (1,KX,1)
  PII(1,KX+1,1)=PII(1,KX,1); P  (1,KX+1,1)=P  (1,KX,1)
  DZ (1,KX+1,1)=DZ (1,KX,1); W  (1,KX+1,1)=W  (1,KX,1)
  RHO(1,KX+1,1)=RHO(1,KX,1)

  ! ---- aero-specific prescribed inputs: NC, KZH, AEROCU ----
  CALL build_aero(case_id, QC, NC, KZH, aerocu)

  ! save inputs
  DO k = 1, KX
    TH0(k)=TH(1,k,1); QV0(k)=QV(1,k,1); QC0(k)=QC(1,k,1); QR0(k)=QR(1,k,1)
    QI0(k)=QI(1,k,1); QS0(k)=QS(1,k,1); QG0(k)=QG(1,k,1)
    NI0(k)=NI(1,k,1); NS0(k)=NS(1,k,1); NR0(k)=NR(1,k,1); NG0(k)=NG(1,k,1)
    NC0(k)=NC(1,k,1)
    PII0(k)=PII(1,k,1); P0(k)=P(1,k,1); DZ0(k)=DZ(1,k,1); W0(k)=W(1,k,1)
    DO l = 1, NAERCU
      AER0(k,l) = aerocu(1,k,1,l)
    END DO
  END DO
  DO k = 1, KX+1
    KZH0(k)=KZH(1,k,1)
  END DO

  HT = 0.0
  CU_UAF = 0.0
  refl_10cm = 0.0
  mskf_refl_10cm = 0.0
  qrcuten = 0.0; qscuten = 0.0; qicuten = 0.0
  NR_CU = 0.0; QR_CU = 0.0; NS_CU = 0.0; QS_CU = 0.0
  EFCG = 0.0; EFIG = 0.0; EFSG = 0.0; WACT = 0.0
  CCN1_GS=0.; CCN2_GS=0.; CCN3_GS=0.; CCN4_GS=0.
  CCN5_GS=0.; CCN6_GS=0.; CCN7_GS=0.
  RAINNC=0.; RAINNCV=0.; SR=0.
  SNOWNC=0.; SNOWNCV=0.; GRAUPELNC=0.; GRAUPELNCV=0.

  ! ---- call the real aerosol-aware Morrison microphysics wrapper ----
  ! All-keyword call mirroring WRF's microphysics_driver CASE (MORR_TM_AERO);
  ! WRF-Chem-only optionals (qndrop, wetscav_on, rainprod, evapprod, QLSINK,
  ! PRECR/I/S/G) omitted; F_QNDROP=.FALSE. as in non-chem WRF.
  CALL MP_MORR_TWO_MOMENT_AERO(                                       &
       ITIMESTEP=itimestep,                                           &
       TH=TH, QV=QV, QC=QC, QR=QR, QI=QI, QS=QS, QG=QG,               &
       NI=NI, NS=NS, NR=NR, NG=NG, NC=NC,                             &
       KZH=KZH, RHO=RHO, PII=PII, P=P, DT_IN=DT, DZ=DZ, HT=HT, W=W,   &
       RAINNC=RAINNC, RAINNCV=RAINNCV, SR=SR,                         &
       SNOWNC=SNOWNC, SNOWNCV=SNOWNCV,                                &
       GRAUPELNC=GRAUPELNC, GRAUPELNCV=GRAUPELNCV,                    &
       REFL_10CM=refl_10cm, MSKF_REFL_10CM=mskf_refl_10cm,            &
       DIAGFLAG=.FALSE., DO_RADAR_REF=0,                              &
       QRCUTEN=qrcuten, QSCUTEN=qscuten, QICUTEN=qicuten,             &
       F_QNDROP=.FALSE.,                                              &
       IDS=IDS,IDE=IDE, JDS=JDS,JDE=JDE, KDS=KDS,KDE=KDE,             &
       IMS=IMS,IME=IME, JMS=JMS,JME=JME, KMS=KMS,KME=KME,             &
       ITS=ITS,ITE=ITE, JTS=JTS,JTE=JTE, KTS=KTS,KTE=KTE,             &
       PBL=PBL,                                                       &
       AEROCU=aerocu, AERCU_OPT=AERCU_OPT, AERCU_FCT=AERCU_FCT,       &
       NO_SRC_TYPES_CU=NAERCU,                                        &
       EFCG=EFCG, EFIG=EFIG, EFSG=EFSG, WACT=WACT,                    &
       CCN1_GS=CCN1_GS, CCN2_GS=CCN2_GS, CCN3_GS=CCN3_GS,             &
       CCN4_GS=CCN4_GS, CCN5_GS=CCN5_GS, CCN6_GS=CCN6_GS,             &
       CCN7_GS=CCN7_GS,                                               &
       NR_CU=NR_CU, QR_CU=QR_CU, NS_CU=NS_CU, QS_CU=QS_CU,            &
       CU_UAF=CU_UAF )

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,I0)') 'KXP1=', KX+1
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  WRITE(*,'(A,I0)') 'AERCU_OPT=', AERCU_OPT
  WRITE(*,'(A,ES23.15)') 'AERCU_FCT=', AERCU_FCT
  WRITE(*,'(A,I0)') 'NO_SRC_TYPES_CU=', NAERCU
  WRITE(*,'(A,I0)') 'PBL=', PBL
  WRITE(*,'(A,ES23.15)') 'HT=', HT(1,1)
  WRITE(*,'(A,ES23.15)') 'CU_UAF=', CU_UAF(1,1)
  ! inputs
  CALL dump_col('TH_IN', TH0)
  CALL dump_col('QV_IN', QV0)
  CALL dump_col('QC_IN', QC0)
  CALL dump_col('QR_IN', QR0)
  CALL dump_col('QI_IN', QI0)
  CALL dump_col('QS_IN', QS0)
  CALL dump_col('QG_IN', QG0)
  CALL dump_col('NI_IN', NI0)
  CALL dump_col('NS_IN', NS0)
  CALL dump_col('NR_IN', NR0)
  CALL dump_col('NG_IN', NG0)
  CALL dump_col('NC_IN', NC0)
  CALL dump_col('PII',   PII0)
  CALL dump_col('P',     P0)
  CALL dump_col('DZ',    DZ0)
  CALL dump_col('W',     W0)
  CALL dump_colp1('KZH_IN', KZH0)
  CALL dump_col3('QRCUTEN_IN', qrcuten)
  CALL dump_col3('QSCUTEN_IN', qscuten)
  CALL dump_col3('QICUTEN_IN', qicuten)
  CALL dump_col3('NR_CU_IN', NR_CU)
  CALL dump_col3('QR_CU_IN', QR_CU)
  CALL dump_col3('NS_CU_IN', NS_CU)
  CALL dump_col3('QS_CU_IN', QS_CU)
  CALL dump_aer('AEROCU_DUST1',   1)
  CALL dump_aer('AEROCU_DUST2',   2)
  CALL dump_aer('AEROCU_DUST3',   3)
  CALL dump_aer('AEROCU_DUST4',   4)
  CALL dump_aer('AEROCU_SEASALT', 5)
  CALL dump_aer('AEROCU_SULFATE', 6)
  CALL dump_aer('AEROCU_BCPHOB',  7)
  CALL dump_aer('AEROCU_BCPHIL',  8)
  CALL dump_aer('AEROCU_OCPHOB',  9)
  CALL dump_aer('AEROCU_OCPHIL', 10)
  ! outputs (post-microphysics)
  CALL dump_col3('TH_OUT', TH)
  CALL dump_col3('QV_OUT', QV)
  CALL dump_col3('QC_OUT', QC)
  CALL dump_col3('QR_OUT', QR)
  CALL dump_col3('QI_OUT', QI)
  CALL dump_col3('QS_OUT', QS)
  CALL dump_col3('QG_OUT', QG)
  CALL dump_col3('NI_OUT', NI)
  CALL dump_col3('NS_OUT', NS)
  CALL dump_col3('NR_OUT', NR)
  CALL dump_col3('NG_OUT', NG)
  CALL dump_col3('NC_OUT', NC)
  CALL dump_col3('EFCG_OUT', EFCG)
  CALL dump_col3('EFIG_OUT', EFIG)
  CALL dump_col3('EFSG_OUT', EFSG)
  CALL dump_col3('WACT_OUT', WACT)
  CALL dump_col3('CCN1_GS_OUT', CCN1_GS)
  CALL dump_col3('CCN2_GS_OUT', CCN2_GS)
  CALL dump_col3('CCN3_GS_OUT', CCN3_GS)
  CALL dump_col3('CCN4_GS_OUT', CCN4_GS)
  CALL dump_col3('CCN5_GS_OUT', CCN5_GS)
  CALL dump_col3('CCN6_GS_OUT', CCN6_GS)
  CALL dump_col3('CCN7_GS_OUT', CCN7_GS)
  WRITE(*,'(A,ES23.15)') 'RAINNC=',     RAINNC(1,1)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=',    RAINNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNC=',     SNOWNC(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNCV=',    SNOWNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNC=',  GRAUPELNC(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNCV=', GRAUPELNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'SR=',         SR(1,1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(KX), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = 1, KX
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_colp1(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(KX+1), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = 1, KX+1
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_colp1

  SUBROUTINE dump_col3(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = 1, KX
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col3

  SUBROUTINE dump_aer(name, l_idx)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: l_idx
    INTEGER :: kk
    DO kk = 1, KX
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', AER0(kk,l_idx)
    END DO
  END SUBROUTINE dump_aer

  ! ------------------------------------------------------------------
  ! Predeclared single columns. case_id:
  !  1 = warm moist BL, supersaturated low levels
  !      -> droplet condensation, autoconversion (KK2000), accretion,
  !         warm-rain self-collection + sedimentation.
  !  2 = deep mixed-phase with melting layer: cloud/rain low, ice/snow/
  !      graupel aloft, T crossing 273.15 -> psmlt/pgmlt melting, riming,
  !      snow/graupel processes, multi-species sedimentation.
  !  3 = cold ice/snow column (all subfreezing), ice-supersaturated
  !      -> ice nucleation (Cooper), deposition (PRD/PRDS/PRDG), snow
  !         autoconversion (NPRCI), aggregation (NSAGG), ice sedimentation.
  !  4 = graupel-dominant convective core: large qc/qr/qi/qs/qg, cold
  !      -> riming (PSACWS/PSACWG), rain freezing (MNUCCR), rain-ice
  !      collection (PIACR/PRACI), conversion to graupel, rime-splinter.
  !  5 = subsaturated mid-level with rain/snow/graupel falling into dry air
  !      -> rain evaporation (PRE<0), snow/graupel sublimation (EPRDS/EPRDG),
  !         number sublimation (NSUBR/NSUBS/NSUBG).
  !  6 = clean column, slight liquid supersaturation, trace cloud only
  !      -> pure saturation-adjustment condensation (PCC) path.
  ! Bottom-up index (k=1 lowest model layer). 3D index (1,k,1).
  ! COPIED AS-IS from morrison_oracle_driver.f90 (v0.6.0 base oracle).
  ! ------------------------------------------------------------------
  SUBROUTINE build_column(cid, tth, qqv, qqc, qqr, qqi, qqs, qqg, &
                          nni, nns, nnr, nng, ddrho, eexner, ppr, ddz, ww)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: &
         tth,qqv,qqc,qqr,qqi,qqs,qqg,nni,nns,nnr,nng,ddrho,eexner,ppr,ddz,ww
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
      ddrho(1,kk,1)  = p_k/(R_D*tv_k)              ! not used internally; passed for completeness
      IF (kk == 1) THEN
        ddz(1,kk,1) = 2.0*zz(1)
      ELSE
        ddz(1,kk,1) = zz(kk)-zz(kk-1)
      END IF
      ww(1,kk,1)   = wmax*EXP(-((z_k-zml-1000.0)/3000.0)**2)
    END DO

    ! Seed hydrometeors + number concentrations per regime. Number concs are
    ! seeded consistent with the mass (representative mean sizes) so the
    ! Morrison size-distribution slope limiters and process rates are exercised.
    qqc=0.; qqr=0.; qqi=0.; qqs=0.; qqg=0.
    nni=0.; nns=0.; nnr=0.; nng=0.
    DO kk = 1, KX
      z_k = zz(kk)
      SELECT CASE (cid)
      CASE (1)   ! warm: cloud + a little rain low
        IF (z_k < 3000.0)  qqc(1,kk,1) = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0)  qqr(1,kk,1) = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
        nnr(1,kk,1) = qqr(1,kk,1)/5.0E-9        ! ~ Nr for mean rain mass
      CASE (2)   ! mixed-phase: cloud low, rain mid, ice/snow/graupel aloft
        IF (z_k < 4000.0)  qqc(1,kk,1) = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k < 5000.0)  qqr(1,kk,1) = 6.0E-4*EXP(-((z_k-1200.0)/1500.0)**2)
        IF (z_k > 4000.0)  qqi(1,kk,1) = 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        IF (z_k > 3500.0)  qqs(1,kk,1) = 8.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        IF (z_k > 3500.0)  qqg(1,kk,1) = 5.0E-4*EXP(-((z_k-5500.0)/2500.0)**2)
        nnr(1,kk,1) = qqr(1,kk,1)/5.0E-9
        nni(1,kk,1) = qqi(1,kk,1)/1.0E-10
        nns(1,kk,1) = qqs(1,kk,1)/2.0E-8
        nng(1,kk,1) = qqg(1,kk,1)/5.0E-8
      CASE (3)   ! cold: ice + snow aloft
        qqi(1,kk,1) = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qqs(1,kk,1) = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
        IF (z_k > 7000.0) qqg(1,kk,1) = 1.0E-4*EXP(-((z_k-9000.0)/2500.0)**2)
        nni(1,kk,1) = qqi(1,kk,1)/1.0E-10
        nns(1,kk,1) = qqs(1,kk,1)/2.0E-8
        nng(1,kk,1) = qqg(1,kk,1)/5.0E-8
      CASE (4)   ! graupel-dominant convective core
        IF (z_k < 5000.0)  qqc(1,kk,1) = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        IF (z_k < 6000.0)  qqr(1,kk,1) = 1.5E-3*EXP(-((z_k-2500.0)/2000.0)**2)
        IF (z_k > 4000.0)  qqi(1,kk,1) = 4.0E-4*EXP(-((z_k-7500.0)/2500.0)**2)
        qqs(1,kk,1) = 1.0E-3*EXP(-((z_k-6000.0)/3000.0)**2)
        qqg(1,kk,1) = 2.0E-3*EXP(-((z_k-6000.0)/2500.0)**2)
        nnr(1,kk,1) = qqr(1,kk,1)/5.0E-9
        nni(1,kk,1) = qqi(1,kk,1)/1.0E-10
        nns(1,kk,1) = qqs(1,kk,1)/2.0E-8
        nng(1,kk,1) = qqg(1,kk,1)/5.0E-8
      CASE (5)   ! subsaturated: rain + snow + graupel falling into dry air
        IF (z_k < 6000.0)  qqr(1,kk,1) = 7.0E-4*EXP(-((z_k-3000.0)/2000.0)**2)
        qqs(1,kk,1) = 6.0E-4*EXP(-((z_k-5000.0)/2500.0)**2)
        IF (z_k > 5000.0) qqg(1,kk,1) = 2.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        nnr(1,kk,1) = qqr(1,kk,1)/5.0E-9
        nns(1,kk,1) = qqs(1,kk,1)/2.0E-8
        nng(1,kk,1) = qqg(1,kk,1)/5.0E-8
      CASE (6)   ! clean: trace cloud only -> condensation path
        IF (z_k < 3000.0)  qqc(1,kk,1) = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
      END SELECT
    END DO
  END SUBROUTINE build_column

  ! ------------------------------------------------------------------
  ! Aero-specific prescribed inputs (see README.md for the rationale):
  !  NC     droplet number (#/kg), seeded consistent with QC assuming a
  !         10 um mean-radius droplet (m = 4.18879e-12 kg) -> continental
  !         ~300-450 cm-3 in-cloud values.
  !  KZH    heat eddy diffusivity (m2/s) as a YSU-like convective-BL
  !         profile: 1 + 59*exp(-((z-600)/500)^2), defined on k=1..KX+1
  !         (scheme reads k+1). WVAR = KZH/20 in [0.1,50].
  !  AEROCU 10-species prescribed aerosol mass concentration (ug/m3),
  !         Registry order (dust1..4, seasalt, sulfate, BCphob, BCphil,
  !         OCphob, OCphil), continental-background surface values scaled
  !         by exp(-z/2000) with a 0.005 free-troposphere floor, and a
  !         per-case loading factor (polluted convective x2, cold x0.5,
  !         clean x0.2).
  ! ------------------------------------------------------------------
  SUBROUTINE build_aero(cid, qqc, nnc, kkzh, aaer)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN)    :: qqc
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(INOUT) :: nnc, kkzh
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme,NAERCU), INTENT(INOUT) :: aaer
    ! surface aerosol mass concentrations (ug/m3), aerocu Registry order
    REAL, PARAMETER, DIMENSION(NAERCU) :: AERSFC = &
       (/ 0.50, 0.30, 0.15, 0.05, 0.50, 1.50, 0.05, 0.05, 0.40, 0.40 /)
    REAL :: z_k, fac, af
    INTEGER :: kk, ll

    SELECT CASE (cid)          ! per-case aerosol loading factor
    CASE (4)
      af = 2.0                 ! polluted convective core
    CASE (3)
      af = 0.5                 ! cold, cleaner airmass
    CASE (6)
      af = 0.2                 ! clean condensation-path column
    CASE DEFAULT
      af = 1.0                 ! continental background
    END SELECT

    DO kk = 1, KX+1
      ! same height grid as build_column, extended one level up
      z_k = 16000.0 * ( (REAL(kk)-0.5)/REAL(KX) )**1.15
      kkzh(1,kk,1) = 1.0 + 59.0*EXP(-((z_k-600.0)/500.0)**2)
      fac = MAX(EXP(-z_k/2000.0), 0.005)
      DO ll = 1, NAERCU
        aaer(1,kk,1,ll) = af * AERSFC(ll) * fac
      END DO
      nnc(1,kk,1) = qqc(1,kk,1)/DROPMASS
    END DO
  END SUBROUTINE build_aero

END PROGRAM morraero_oracle
