! =====================================================================
! v0.6.0 single-column Morrison 2-moment (WRF mp_physics=10) oracle driver.
!
! Drives the UNMODIFIED WRF Morrison scheme:
!   MORR_TWO_MOMENT_INIT(morr_rimed_ice=0)   -> IHAIL=0, IGRAUP=0, graupel mode
!   MP_MORR_TWO_MOMENT(...)                   -> full 3D->1D->3D wrapper that
!       converts th<->t via the Exner function, recomputes rho internally,
!       calls MORR_TWO_MOMENT_MICRO on the column, and binds surface precip.
! on a prescribed single column (1x1xKX) and dumps the FULL input state
! (TH,QV,QC,QR,QI,QS,QG,NI,NS,NR,NG,PII,P,DZ,W) and the FULL output state
! (the same after the call) plus surface accumulators
! (RAINNCV,SNOWNCV,GRAUPELNCV,SR) and effective radii (EFFC/EFFI/EFFS/EFFR/EFFG).
!
! This is the GOLD oracle for the JAX port: it is the real Fortran Morrison
! scheme (module_mp_morr_two_moment.F), not a re-implementation, so the JAX
! port cannot "self-compare". The init is the real MORR_TWO_MOMENT_INIT so
! every saved module constant (GAMMA-derived CONS1..41, fall-speed prefactors,
! size limits) is exactly WRF.
!
! Usage: ./morrison_oracle <case_id>
! Output: flat key=value text dump on stdout (parsed by Python into JSON).
!
! Morrison REALs are default REAL (single precision, REAL*4) just as in WRF
! when built with default real kind; the JAX port runs fp64. Parity is to a
! PREDECLARED physical tolerance, never bitwise (see run_morrison_parity.py).
! A second build (real-8 promotion via -fdefault-real-8) supplies an fp64
! reference for diagnostics that have categorical fp32 detection-floor dust.
! =====================================================================
PROGRAM morrison_oracle
  USE module_mp_morr_two_moment, ONLY : MORR_TWO_MOMENT_INIT, MP_MORR_TWO_MOMENT
  IMPLICIT NONE

  ! WRF model constants used to build the column (share/module_model_constants.F)
  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: P1000  = 1.0E5
  REAL, PARAMETER :: ROVCP  = R_D/CP

  INTEGER, PARAMETER :: KX  = 40
  INTEGER, PARAMETER :: ids=1,ide=2, jds=1,jde=2, kds=1,kde=KX+1
  INTEGER, PARAMETER :: ims=1,ime=1, jms=1,jme=1, kms=1,kme=KX
  INTEGER, PARAMETER :: its=1,ite=1, jts=1,jte=1, kts=1,kte=KX

  ! 3D (1,KX,1) arrays for the wrapper
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: TH,QV,QC,QR,QI,QS,QG,NI,NS,NR,NG
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RHO,PII,P,DZ,W
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: refl_10cm
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: qrcuten,qscuten,qicuten
  REAL, DIMENSION(ims:ime,jms:jme)         :: HT
  REAL, DIMENSION(ims:ime,jms:jme)         :: RAINNC,RAINNCV,SR
  REAL, DIMENSION(ims:ime,jms:jme)         :: SNOWNC,SNOWNCV,GRAUPELNC,GRAUPELNCV

  ! saved input copies (column, k=1..KX)
  REAL, DIMENSION(KX) :: TH0,QV0,QC0,QR0,QI0,QS0,QG0,NI0,NS0,NR0,NG0
  REAL, DIMENSION(KX) :: PII0,P0,DZ0,W0

  REAL :: DT
  INTEGER :: k, case_id, itimestep
  CHARACTER(LEN=32) :: arg

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 60.0          ! representative microphysics dt (s)
  itimestep = 1

  ! ---- initialize Morrison module constants (graupel mode, IHAIL=0) ----
  CALL MORR_TWO_MOMENT_INIT(0)

  ! ---- build the chosen column ----
  CALL build_column(case_id, TH, QV, QC, QR, QI, QS, QG, NI, NS, NR, NG, &
                     RHO, PII, P, DZ, W)

  ! save inputs
  DO k = 1, KX
    TH0(k)=TH(1,k,1); QV0(k)=QV(1,k,1); QC0(k)=QC(1,k,1); QR0(k)=QR(1,k,1)
    QI0(k)=QI(1,k,1); QS0(k)=QS(1,k,1); QG0(k)=QG(1,k,1)
    NI0(k)=NI(1,k,1); NS0(k)=NS(1,k,1); NR0(k)=NR(1,k,1); NG0(k)=NG(1,k,1)
    PII0(k)=PII(1,k,1); P0(k)=P(1,k,1); DZ0(k)=DZ(1,k,1); W0(k)=W(1,k,1)
  END DO

  HT = 0.0
  refl_10cm = 0.0
  qrcuten = 0.0; qscuten = 0.0; qicuten = 0.0
  RAINNC=0.; RAINNCV=0.; SR=0.
  SNOWNC=0.; SNOWNCV=0.; GRAUPELNC=0.; GRAUPELNCV=0.

  ! ---- call the real Morrison microphysics wrapper ----
  ! Dims passed as keyword args so the OPTIONAL F_QNDROP/qndrop (which appear
  ! positionally before the dims in the wrapper signature) are cleanly skipped.
  CALL MP_MORR_TWO_MOMENT(itimestep,                                  &
       TH, QV, QC, QR, QI, QS, QG, NI, NS, NR, NG,                    &
       RHO, PII, P, DT, DZ, HT, W,                                    &
       RAINNC, RAINNCV, SR,                                           &
       SNOWNC, SNOWNCV, GRAUPELNC, GRAUPELNCV,                        &
       refl_10cm, .FALSE., 0,                                         &
       qrcuten, qscuten, qicuten,                                     &
       IDS=IDS,IDE=IDE, JDS=JDS,JDE=JDE, KDS=KDS,KDE=KDE,             &
       IMS=IMS,IME=IME, JMS=JMS,JME=JME, KMS=KMS,KME=KME,             &
       ITS=ITS,ITE=ITE, JTS=JTS,JTE=JTE, KTS=KTS,KTE=KTE)

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
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
  CALL dump_col('PII',   PII0)
  CALL dump_col('P',     P0)
  CALL dump_col('DZ',    DZ0)
  CALL dump_col('W',     W0)
  ! outputs (post-Morrison)
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
  WRITE(*,'(A,ES23.15)') 'RAINNCV=',    RAINNCV(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNCV=',    SNOWNCV(1,1)
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

  SUBROUTINE dump_col3(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = 1, KX
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col3

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
  !         collection (PIACR/PRACI), conversion to graupel, rime-splinter.
  !  5 = subsaturated mid-level with rain/snow/graupel falling into dry air
  !      -> rain evaporation (PRE<0), snow/graupel sublimation (EPRDS/EPRDG),
  !         number sublimation (NSUBR/NSUBS/NSUBG).
  !  6 = clean column, slight liquid supersaturation, trace cloud only
  !      -> pure saturation-adjustment condensation (PCC) path.
  ! Bottom-up index (k=1 lowest model layer). 3D index (1,k,1).
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

END PROGRAM morrison_oracle
