! v0.18 single-column KEPS PBL oracle driver (bl_pbl_physics=17).
PROGRAM keps_oracle
  USE module_bl_keps, ONLY : keps
  IMPLICIT NONE

  REAL, PARAMETER :: G = 9.81
  REAL, PARAMETER :: R_D = 287.0
  REAL, PARAMETER :: CP = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V = 461.6
  REAL, PARAMETER :: ROVCP = R_D/CP
  REAL, PARAMETER :: EP1 = R_V/R_D - 1.0
  REAL, PARAMETER :: P1000 = 1.0E5

  INTEGER, PARAMETER :: KX = 32
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: DZ,U,V,TH,PII,RTHRATEN,P8W,RHO,QV,QC
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN,RQCBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: TKE,DISS,TPE,PR,TKE_ADV,DISS_ADV,TPE_ADV
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: WU,WV,WT,WQ,EXCH_H,EXCH_M
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: A_U,A_V,A_T,A_Q,B_U,B_V,B_T,B_Q,B_E,SF,VL
  REAL, DIMENSION(ims:ime,jms:jme) :: MOL,TSK,FRC_URB2D,HFX,QFX,USTAR,BR,ZNT,PSIM,PSIH,PBLH
  CHARACTER(LEN=32) :: arg, name
  INTEGER :: case_id

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_case(case_id, name, DZ, U, V, TH, PII, P8W, RHO, QV, QC, MOL, TSK, HFX, QFX, USTAR, BR, ZNT, PSIM, PSIH)

  RTHRATEN=0.0
  RUBLTEN=0.0; RVBLTEN=0.0; RTHBLTEN=0.0; RQVBLTEN=0.0; RQCBLTEN=0.0
  TKE=0.05; DISS=1.0E-3; TPE=1.0E-3; PR=1.0
  TKE_ADV=TKE; DISS_ADV=DISS; TPE_ADV=TPE
  WU=0.0; WV=0.0; WT=0.0; WQ=0.0; EXCH_H=0.0; EXCH_M=0.0; PBLH=500.0
  A_U=0.0; A_V=0.0; A_T=0.0; A_Q=0.0; B_U=0.0; B_V=0.0; B_T=0.0; B_Q=0.0; B_E=0.0
  SF=1.0; VL=1.0; FRC_URB2D=0.0

  CALL keps(MOL=MOL, TSK=TSK, XTIME=3, FRC_URB2D=FRC_URB2D, FLAG_BEP=.FALSE., &
            DZ8W=DZ, DT=60.0, U_PHY=U, V_PHY=V, TH_PHY=TH, PI_PHY=PII, RTHRATEN=RTHRATEN, P8W=P8W, &
            RHO=RHO, QV_CURR=QV, QC_CURR=QC, HFX=HFX, QFX=QFX, USTAR=USTAR, CP=CP, G=G, &
            RUBLTEN=RUBLTEN, RVBLTEN=RVBLTEN, RTHBLTEN=RTHBLTEN, RQVBLTEN=RQVBLTEN, RQCBLTEN=RQCBLTEN, &
            TKE_PBL=TKE, DISS_PBL=DISS, TPE_PBL=TPE, TKE_ADV=TKE_ADV, DISS_ADV=DISS_ADV, TPE_ADV=TPE_ADV, &
            PR_PBL=PR, WU=WU, WV=WV, WT=WT, WQ=WQ, EXCH_H=EXCH_H, EXCH_M=EXCH_M, PBLH=PBLH, &
            A_U_BEP=A_U, A_V_BEP=A_V, A_T_BEP=A_T, A_Q_BEP=A_Q, B_U_BEP=B_U, B_V_BEP=B_V, &
            B_T_BEP=B_T, B_Q_BEP=B_Q, B_E_BEP=B_E, SF_BEP=SF, VL_BEP=VL, &
            BR=BR, ZNT=ZNT, PSIM=PSIM, PSIH=PSIH, &
            IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde, &
            IMS=ims, IME=ime, JMS=jms, JME=jme, KMS=kms, KME=kme, &
            ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte)

  CALL dump_case(case_id, name, DZ, U, V, TH, PII, P8W, RHO, QV, QC, MOL, TSK, HFX, QFX, USTAR, BR, ZNT, PBLH, &
                 RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, RQCBLTEN, TKE, DISS, TPE, PR, TKE_ADV, DISS_ADV, TPE_ADV, &
                 WU, WV, WT, WQ, EXCH_H, EXCH_M)

CONTAINS

  SUBROUTINE build_case(cid, nm, DzA,Uu,Vv,Tht,Exner,P8wA,RhoA,Qq,QcA,MolA,TskA,HfxA,QfxA,UstA,BrA,ZntA,PsimA,PsihA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: DzA,Uu,Vv,Tht,Exner,P8wA,RhoA,Qq,QcA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: MolA,TskA,HfxA,QfxA,UstA,BrA,ZntA,PsimA,PsihA
    REAL, DIMENSION(KX+1) :: zint, pint
    REAL, DIMENSION(KX) :: zmid, theta, qprof, uprof, vprof, temp, pfull
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, shear, z, tv, pi_mass
    REAL :: hfx0, qfx0, ust0, br0, znt0, tsk0, mol0
    INTEGER :: k

    ztop = 12000.0
    zint(1) = 0.0
    DO k = 1, KX
      zint(k+1) = ztop * (REAL(k)/REAL(KX))**1.18
      zmid(k) = 0.5*(zint(k)+zint(k+1))
    END DO

    SELECT CASE (cid)
    CASE (1)
      nm='unstable_daytime'
      psfc0=100000.0; theta0=300.0; zml=1100.0; lapse_ml=0.0003; lapse_ft=0.0045
      q0=0.0140; qscale=2300.0; shear=0.0015; hfx0=350.0; qfx0=1.20E-4; ust0=0.55; br0=-0.08; znt0=0.10; tsk0=302.0; mol0=-50.0
    CASE (2)
      nm='strong_unstable'
      psfc0=100800.0; theta0=302.0; zml=1400.0; lapse_ml=0.0002; lapse_ft=0.0038
      q0=0.0160; qscale=2600.0; shear=0.0020; hfx0=600.0; qfx0=9.00E-5; ust0=0.80; br0=-0.18; znt0=0.08; tsk0=305.0; mol0=-25.0
    CASE (3)
      nm='stable_nocturnal'
      psfc0=100500.0; theta0=287.0; zml=150.0; lapse_ml=0.0100; lapse_ft=0.0060
      q0=0.0060; qscale=1600.0; shear=0.0060; hfx0=-60.0; qfx0=0.0; ust0=0.25; br0=0.15; znt0=0.05; tsk0=285.0; mol0=40.0
    CASE (4)
      nm='stable_marine'
      psfc0=101200.0; theta0=289.0; zml=250.0; lapse_ml=0.0040; lapse_ft=0.0032
      q0=0.0100; qscale=2600.0; shear=0.0010; hfx0=-15.0; qfx0=2.00E-5; ust0=0.18; br0=0.05; znt0=0.001; tsk0=288.0; mol0=80.0
    CASE (5)
      nm='neutral'
      psfc0=100000.0; theta0=296.0; zml=800.0; lapse_ml=0.0010; lapse_ft=0.0030
      q0=0.0090; qscale=2200.0; shear=0.0025; hfx0=0.0; qfx0=0.0; ust0=0.40; br0=0.0; znt0=0.08; tsk0=296.0; mol0=1.0E6
    CASE DEFAULT
      nm='low_ust_edge'
      psfc0=99500.0; theta0=295.0; zml=700.0; lapse_ml=0.0010; lapse_ft=0.0040
      q0=0.0080; qscale=1800.0; shear=0.0010; hfx0=20.0; qfx0=1.00E-5; ust0=0.05; br0=-0.01; znt0=0.03; tsk0=296.0; mol0=-200.0
    END SELECT

    pint(1) = psfc0
    DO k = 1, KX
      z = zmid(k)
      IF (z <= zml) THEN
        theta(k) = theta0 + lapse_ml*z
      ELSE
        theta(k) = theta0 + lapse_ml*zml + lapse_ft*(z-zml)
      END IF
      qprof(k) = MAX(q0*EXP(-z/qscale), 1.0E-5)
      uprof(k) = 4.0 + shear*z
      vprof(k) = 1.0 + 0.25*shear*z
      pi_mass = (MAX(pint(k), 1000.0)/P1000)**ROVCP
      temp(k) = theta(k)*pi_mass
      tv = temp(k)*(1.0 + EP1*qprof(k))
      pint(k+1) = pint(k)*EXP(-G*(zint(k+1)-zint(k))/(R_D*MAX(tv, 180.0)))
      pfull(k) = 0.5*(pint(k)+pint(k+1))
      pi_mass = (pfull(k)/P1000)**ROVCP
      temp(k) = theta(k)*pi_mass
    END DO

    DzA=0.0; Uu=0.0; Vv=0.0; Tht=0.0; Exner=0.0; P8wA=0.0; RhoA=0.0; Qq=0.0; QcA=0.0
    DO k = 1, KX
      DzA(1,k,1) = zint(k+1)-zint(k)
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Tht(1,k,1) = theta(k)
      Exner(1,k,1) = (pfull(k)/P1000)**ROVCP
      P8wA(1,k,1) = pint(k)
      RhoA(1,k,1) = pfull(k)/(R_D*temp(k)*(1.0 + EP1*qprof(k)))
      Qq(1,k,1) = qprof(k)
    END DO
    P8wA(1,KX+1,1) = pint(KX+1)
    MolA(1,1)=mol0; TskA(1,1)=tsk0; HfxA(1,1)=hfx0; QfxA(1,1)=qfx0
    UstA(1,1)=ust0; BrA(1,1)=br0; ZntA(1,1)=znt0; PsimA(1,1)=0.0; PsihA(1,1)=0.0
  END SUBROUTINE build_case

  SUBROUTINE dump_col(nm, arr)
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') nm,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_case(cid,nm,DzA,Uu,Vv,Tht,Exner,P8wA,RhoA,Qq,QcA,MolA,TskA,HfxA,QfxA,UstA,BrA,ZntA,PblA, &
                       Uten,Vten,Thten,Qvten,Qcten,TkeA,DissA,TpeA,PrA,TkeAdvA,DissAdvA,TpeAdvA,WuA,WvA,WtA,WqA,ExchH,ExchM)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: DzA,Uu,Vv,Tht,Exner,P8wA,RhoA,Qq,QcA
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uten,Vten,Thten,Qvten,Qcten,TkeA,DissA,TpeA,PrA,TkeAdvA,DissAdvA,TpeAdvA
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: WuA,WvA,WtA,WqA,ExchH,ExchM
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: MolA,TskA,HfxA,QfxA,UstA,BrA,ZntA,PblA
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(nm)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,ES23.15)') 'DT=', 60.0
    WRITE(*,'(A,ES23.15)') 'MOL=', MolA(1,1)
    WRITE(*,'(A,ES23.15)') 'TSK=', TskA(1,1)
    WRITE(*,'(A,ES23.15)') 'HFX=', HfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'QFX=', QfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'UST=', UstA(1,1)
    WRITE(*,'(A,ES23.15)') 'BR=', BrA(1,1)
    WRITE(*,'(A,ES23.15)') 'ZNT=', ZntA(1,1)
    WRITE(*,'(A,ES23.15)') 'PBL=', PblA(1,1)
    CALL dump_col('DZ', DzA)
    CALL dump_col('U', Uu)
    CALL dump_col('V', Vv)
    CALL dump_col('TH', Tht)
    CALL dump_col('PI', Exner)
    CALL dump_col('P8W', P8wA)
    CALL dump_col('RHO', RhoA)
    CALL dump_col('QV', Qq)
    CALL dump_col('QC', QcA)
    CALL dump_col('RUBLTEN', Uten)
    CALL dump_col('RVBLTEN', Vten)
    CALL dump_col('RTHBLTEN', Thten)
    CALL dump_col('RQVBLTEN', Qvten)
    CALL dump_col('RQCBLTEN', Qcten)
    CALL dump_col('TKE_PBL', TkeA)
    CALL dump_col('DISS_PBL', DissA)
    CALL dump_col('TPE_PBL', TpeA)
    CALL dump_col('PR_PBL', PrA)
    CALL dump_col('TKE_ADV', TkeAdvA)
    CALL dump_col('DISS_ADV', DissAdvA)
    CALL dump_col('TPE_ADV', TpeAdvA)
    CALL dump_col('WU_TUR', WuA)
    CALL dump_col('WV_TUR', WvA)
    CALL dump_col('WT_TUR', WtA)
    CALL dump_col('WQ_TUR', WqA)
    CALL dump_col('EXCH_H', ExchH)
    CALL dump_col('EXCH_M', ExchM)
  END SUBROUTINE dump_case
END PROGRAM keps_oracle
