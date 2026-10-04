! =====================================================================
! P0-4 single-column Kain-Fritsch-eta (WRF cu_physics=1) oracle driver.
!
! Drives the UNMODIFIED WRF module_cu_kfeta.F (KF_eta_CPS -> KF_eta_PARA)
! on a prescribed single-column sounding and dumps:
!   * the full input state (T, QV, P, dz, rho, U, V, W0AVG, DX, DT, STEPCU)
!   * the full output tendencies (RTHCUTEN, RQVCUTEN, RQCCUTEN, RQRCUTEN,
!     RQICUTEN, RQSCUTEN), RAINCV, PRATEC, NCA, CUTOP, CUBOT, ISHALL/SHALL
!
! This is the GOLD oracle for the JAX port. It is the real Fortran scheme,
! not a re-implementation, so the JAX port cannot "self-compare". The W0AVG
! that the scheme actually uses is dumped AFTER the CPS running-mean update so
! the JAX port can be fed the identical W0AVG and we test KF_eta_PARA in
! isolation from the trivial W0AVG recurrence.
!
! Usage: ./kf_oracle <case_id>
!   case_id selects one of the predeclared soundings (see build_sounding).
! Output: a flat key=value text dump on stdout (parsed by Python into JSON).
! =====================================================================
PROGRAM kf_oracle
  USE module_cu_kfeta, ONLY : kf_eta_cps, kf_eta_init
  IMPLICIT NONE

  ! WRF model constants (share/module_model_constants.F)
  REAL, PARAMETER :: G    = 9.81
  REAL, PARAMETER :: R_D  = 287.0
  REAL, PARAMETER :: CP   = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V  = 461.6
  REAL, PARAMETER :: XLV0 = 3.15E6
  REAL, PARAMETER :: XLV1 = 2370.0
  REAL, PARAMETER :: XLS0 = 2.905E6
  REAL, PARAMETER :: XLS1 = 259.532
  REAL, PARAMETER :: SVP1 = 0.6112
  REAL, PARAMETER :: SVP2 = 17.67
  REAL, PARAMETER :: SVP3 = 29.65
  REAL, PARAMETER :: SVPT0= 273.15
  REAL, PARAMETER :: EP_1 = R_V/R_D - 1.0
  REAL, PARAMETER :: EP_2 = R_D/R_V
  REAL, PARAMETER :: P1000= 1.0E5
  REAL, PARAMETER :: ROVCP= R_D/CP

  ! Single-column domain (i=j=1)
  INTEGER, PARAMETER :: KX = 40
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,W,TH,T,QV,DZ8W,PCPS,RHO,PII
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: W0AVG, RQVFTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RTHCUTEN,RQVCUTEN,RQCCUTEN,RQRCUTEN,RQICUTEN,RQSCUTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: CLDFRA_DP_KF,CLDFRA_SH_KF,QC_KF,QI_KF
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: UDR_KF,DDR_KF,UER_KF,DER_KF
  REAL, DIMENSION(ims:ime,jms:jme) :: RAINCV,PRATEC,NCA,CUTOP,CUBOT,SHALL,TIMEC_KF
  LOGICAL, DIMENSION(ims:ime,jms:jme) :: CU_ACT_FLAG

  REAL :: DT, DX, CUDT, W0PRESET
  INTEGER :: STEPCU, KTAU, trigger, KF_EDRATES
  LOGICAL :: warm_rain, adapt_step_flag
  LOGICAL :: F_QV,F_QC,F_QR,F_QI,F_QS
  INTEGER :: k, case_id
  CHARACTER(LEN=32) :: arg
  REAL, DIMENSION(KX) :: zlev

  ! --- parse case id ---
  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  ! --- model timestep / grid for a 9 km parent ---
  DX     = 9000.0
  DT     = 54.0            ! ~6*DX/1000 s; representative parent dt
  CUDT   = 0.0             ! cumulus called every step in adaptive WRF default path
  STEPCU = 5               ! KF called every 5 steps (=> TST=10 in W0AVG mean)
  KTAU   = 100
  trigger= 1               ! classic Kain-Fritsch-Chappell trigger (WRF default)
  KF_EDRATES = 1           ! dump entrainment/detrainment rates too
  warm_rain = .FALSE.
  adapt_step_flag = .FALSE.
  ! Thompson-style mixed-phase microphysics flags
  F_QV=.TRUE.; F_QC=.TRUE.; F_QR=.TRUE.; F_QI=.TRUE.; F_QS=.TRUE.

  ! --- init tendency arrays, NCA, W0AVG, and build the lookup table ---
  CALL kf_eta_init(RTHCUTEN,RQVCUTEN,RQCCUTEN,RQRCUTEN,RQICUTEN,RQSCUTEN, &
                   NCA,W0AVG,2,3,SVP1,SVP2,SVP3,SVPT0,1,.FALSE.,.TRUE.,   &
                   ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme,      &
                   its,ite,jts,jte,kts,kte)

  ! --- build the chosen sounding -> fills T, QV, PCPS, DZ8W, RHO, U, V, zlev, W0PRESET ---
  CALL build_sounding(case_id, T, QV, PCPS, DZ8W, RHO, U, V, zlev, W0PRESET)

  ! Exner / potential temperature (TH used only for trigger==2; harmless to set)
  DO k=kts,kte
    PII(1,k,1) = (PCPS(1,k,1)/P1000)**ROVCP
    TH (1,k,1) = T(1,k,1)/PII(1,k,1)
    W  (1,k,1) = 0.0
    RQVFTEN(1,k,1) = 0.0
  END DO

  ! Prescribe the running-mean vertical velocity profile DIRECTLY into W0AVG.
  ! We bypass the trivial CPS recurrence by setting W=0 and forcing the
  ! running mean to W0PRESET via the analytic fixed point: with W0den=TST and
  ! W0fctr=1, W0=0 => W0AVG_new = W0AVG*(TST-1)/TST. To pin a steady value we
  ! seed W0AVG to W0PRESET and pass W consistent with that value.
  ! Simpler & exact: set W so that 0.5*(w(k)+w(k+1)) == W0PRESET*TST - W0AVG*(TST-1),
  ! but we instead dump the W0AVG the scheme USES and feed the same to JAX.
  DO k=kts,kte
    W0AVG(1,k,1) = w0_profile(zlev(k), W0PRESET)
  END DO
  ! make W consistent so the CPS update keeps W0AVG at the prescribed value:
  ! W0AVG_new = (W0AVG*(TST-1) + W0)/TST ; for W0AVG_new==W0AVG we need W0=W0AVG
  ! and W0=0.5*(w(k)+w(k+1)). Set w(k)=W0AVG(k) so 0.5*(w(k)+w(k+1)) ~ W0AVG.
  DO k=kts,kte
    W(1,k,1) = W0AVG(1,k,1)
  END DO

  RAINCV=0.0; PRATEC=0.0; CUTOP=0.0; CUBOT=0.0; SHALL=0.0; TIMEC_KF=0.0
  CLDFRA_DP_KF=0.0; CLDFRA_SH_KF=0.0; QC_KF=0.0; QI_KF=0.0
  UDR_KF=0.0; DDR_KF=0.0; UER_KF=0.0; DER_KF=0.0
  CU_ACT_FLAG=.TRUE.
  NCA(1,1) = -100.0        ! allow convection this step

  CALL KF_ETA_CPS( &
       ids,ide, jds,jde, kds,kde, &
       ims,ime, jms,jme, kms,kme, &
       its,ite, jts,jte, kts,kte, &
       trigger, DT,KTAU,DX,CUDT,adapt_step_flag, &
       RHO,RAINCV,PRATEC,NCA, &
       U,V,TH,T,W,DZ8W,PCPS,PII, &
       W0AVG,XLV0,XLV1,XLS0,XLS1,CP,R_D,G,EP_1, &
       EP_2,SVP1,SVP2,SVP3,SVPT0, &
       STEPCU,CU_ACT_FLAG,warm_rain,CUTOP,CUBOT, &
       QV, SHALL, &
       F_QV,F_QC,F_QR,F_QI,F_QS, &
       RTHCUTEN,RQVCUTEN,RQCCUTEN,RQRCUTEN, &
       RQICUTEN,RQSCUTEN, RQVFTEN, &
       CLDFRA_DP_KF,CLDFRA_SH_KF, &
       QC_KF,QI_KF, &
       UDR_KF,DDR_KF, &
       UER_KF,DER_KF, &
       TIMEC_KF,KF_EDRATES )

  ! ---------------- dump everything (flat key=value) -----------------
  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  WRITE(*,'(A,ES23.15)') 'DX=', DX
  WRITE(*,'(A,I0)') 'STEPCU=', STEPCU
  WRITE(*,'(A,I0)') 'TRIGGER=', trigger
  CALL dump_col('T',   T)
  CALL dump_col('QV',  QV)
  CALL dump_col('P',   PCPS)
  CALL dump_col('DZ',  DZ8W)
  CALL dump_col('RHO', RHO)
  CALL dump_col('U',   U)
  CALL dump_col('V',   V)
  CALL dump_col('W0AVG', W0AVG)
  CALL dump_col('RTHCUTEN', RTHCUTEN)
  CALL dump_col('RQVCUTEN', RQVCUTEN)
  CALL dump_col('RQCCUTEN', RQCCUTEN)
  CALL dump_col('RQRCUTEN', RQRCUTEN)
  CALL dump_col('RQICUTEN', RQICUTEN)
  CALL dump_col('RQSCUTEN', RQSCUTEN)
  WRITE(*,'(A,ES23.15)') 'RAINCV=', RAINCV(1,1)
  WRITE(*,'(A,ES23.15)') 'PRATEC=', PRATEC(1,1)
  WRITE(*,'(A,ES23.15)') 'NCA=', NCA(1,1)
  WRITE(*,'(A,ES23.15)') 'CUTOP=', CUTOP(1,1)
  WRITE(*,'(A,ES23.15)') 'CUBOT=', CUBOT(1,1)
  WRITE(*,'(A,ES23.15)') 'SHALL=', SHALL(1,1)
  WRITE(*,'(A,ES23.15)') 'TIMEC=', TIMEC_KF(1,1)

CONTAINS

  ! prescribed running-mean w profile (m/s): a smooth low-level updraft bump
  REAL FUNCTION w0_profile(z, w0peak)
    REAL, INTENT(IN) :: z, w0peak
    REAL :: zc, sig
    zc = 1500.0; sig = 1200.0
    w0_profile = w0peak * EXP(-((z-zc)/sig)**2)
  END FUNCTION w0_profile

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk=kts,kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  ! ------------------------------------------------------------------
  ! Predeclared soundings. case_id:
  !   1 = warm moist tropical (Weisman-Klemp-like) -> deep convection
  !   2 = drier/cooler        -> marginal / shallow / no-convection
  !   3 = strong updraft, very moist -> vigorous deep convection
  !   4 = capped (warm dry mid-level) -> shallow
  !   5 = stable dry subsident -> no convection
  ! Pressure/height from a hydrostatic integration of the temperature/qv.
  ! ------------------------------------------------------------------
  SUBROUTINE build_sounding(cid, Tt, Qq, Pp, Dz, Rr, Uu, Vv, zz, w0peak)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Tt,Qq,Pp,Dz,Rr,Uu,Vv
    REAL, DIMENSION(KX), INTENT(OUT) :: zz
    REAL, INTENT(OUT) :: w0peak
    REAL :: psfc, tsfc, theta_sfc, qsfc, ztop
    REAL :: th_k, t_k, q_k, p_k, tv_k, rh_ml, rh_trop
    REAL :: z_k, es, qs, ushr, zml, theta_trop_lapse, rh_k
    INTEGER :: kk

    ! common: stretched vertical grid, ~20 km top over 40 levels
    ztop = 20000.0
    DO kk=1,KX
      zz(kk) = ztop * ( (REAL(kk)-0.5)/REAL(KX) )**1.18
    END DO

    ! Conditionally-unstable profile (Weisman-Klemp-like, simplified):
    !   - well-mixed sub-cloud layer of depth zml with constant theta
    !   - above zml a conditionally-unstable troposphere (small theta lapse)
    !   - RH ~ constant in ML, tapering aloft
    ! Parameters per case tune CAPE / cloud depth / trigger strength.
    SELECT CASE (cid)
    CASE (1)   ! warm moist tropical -> deep convection (interior cloud top)
      psfc=1000.0E2; tsfc=300.0; zml=1000.0; theta_trop_lapse=4.0E-3
      rh_ml=0.88; rh_trop=0.55; ushr=0.0;  w0peak=0.40
    CASE (2)   ! cooler/drier -> marginal/shallow/none
      psfc=1000.0E2; tsfc=294.0; zml=900.0;  theta_trop_lapse=4.2E-3
      rh_ml=0.70; rh_trop=0.40; ushr=0.0;  w0peak=0.20
    CASE (3)   ! very warm/moist + shear -> vigorous deep (decisive interior top)
      psfc=1008.0E2; tsfc=302.0; zml=1200.0; theta_trop_lapse=4.5E-3
      rh_ml=0.90; rh_trop=0.55; ushr=12.0; w0peak=0.55
    CASE (4)   ! capped: warm-dry mid-troposphere -> suppressed/shallow
      psfc=1000.0E2; tsfc=300.0; zml=800.0;  theta_trop_lapse=5.5E-3
      rh_ml=0.78; rh_trop=0.25; ushr=4.0;  w0peak=0.25
    CASE (5)   ! stable, dry, weak subsidence -> no convection
      psfc=1000.0E2; tsfc=286.0; zml=250.0;  theta_trop_lapse=9.0E-3
      rh_ml=0.35; rh_trop=0.08; ushr=0.0;  w0peak=-0.08
    CASE DEFAULT
      psfc=1000.0E2; tsfc=301.0; zml=1000.0; theta_trop_lapse=3.0E-3
      rh_ml=0.90; rh_trop=0.70; ushr=0.0;  w0peak=0.45
    END SELECT

    theta_sfc = tsfc * (P1000/psfc)**ROVCP
    qsfc = 0.0   ! unused; moisture set from RH target below

    ! Hydrostatic integration upward (bottom-up KF ordering: index 1 = lowest)
    p_k = psfc
    DO kk=1,KX
      z_k = zz(kk)
      ! potential temperature: constant in ML, then conditionally-unstable lapse
      IF (z_k <= zml) THEN
        th_k = theta_sfc
      ELSE
        th_k = theta_sfc + theta_trop_lapse*(z_k - zml)
      END IF
      ! target RH: full in ML, taper toward rh_trop with height
      IF (z_k <= zml) THEN
        rh_k = rh_ml
      ELSE
        rh_k = rh_trop + (rh_ml-rh_trop)*EXP(-(z_k-zml)/2500.0)
      END IF

      ! pressure: hydrostatic from the level below using virtual temp there
      IF (kk==1) THEN
        t_k  = th_k*(psfc/P1000)**ROVCP
        tv_k = t_k*(1.0+0.0)
        p_k  = psfc * EXP(-G*zz(1)/(R_D*tv_k))
      ELSE
        t_k  = th_k*(p_k/P1000)**ROVCP
        tv_k = t_k*(1.0+0.608*Qq(1,kk-1,1))
        p_k  = p_k * EXP(-G*(zz(kk)-zz(kk-1))/(R_D*tv_k))
      END IF
      t_k = th_k*(p_k/P1000)**ROVCP
      es  = (SVP1*1000.0)*EXP((SVP2*t_k - SVP2*SVPT0)/(t_k - SVP3))
      qs  = 0.622*es/(p_k-es)
      q_k = rh_k*qs
      q_k = MAX(q_k, 1.0E-6)
      tv_k = t_k*(1.0+0.608*q_k)

      Tt(1,kk,1) = t_k
      Qq(1,kk,1) = q_k
      Pp(1,kk,1) = p_k
      Rr(1,kk,1) = p_k/(R_D*tv_k)
      IF (kk==1) THEN
        Dz(1,kk,1) = 2.0*zz(1)
      ELSE
        Dz(1,kk,1) = zz(kk)-zz(kk-1)
      END IF
      Uu(1,kk,1) = ushr * z_k/10000.0
      Vv(1,kk,1) = 0.0
    END DO
  END SUBROUTINE build_sounding

END PROGRAM kf_oracle
