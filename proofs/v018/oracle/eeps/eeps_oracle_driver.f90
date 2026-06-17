! v0.18 single-column EEPS PBL oracle driver (bl_pbl_physics=16).
PROGRAM eeps_oracle
  USE module_bl_eeps, ONLY : eeps
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

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,T,TH,QV,QC,QI,QR,QS,QG,P,PII,DZ,RHO,RTHRATEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN,RQCBLTEN,RQIBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: PEK,PEP,PEK_ADV,PEP_ADV,EXCH_H,EXCH_M
  REAL, DIMENSION(ims:ime,jms:jme) :: PSFC,QSFC,TSK,UST,RMOL,WSPD,XLAND,HFX,QFX,PBLH
  INTEGER, DIMENSION(ims:ime,jms:jme) :: KPBL
  CHARACTER(LEN=32) :: arg, name
  INTEGER :: case_id

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_case(case_id, name, U, V, T, TH, QV, QC, QI, QR, QS, QG, P, PII, DZ, RHO, &
                  PSFC, QSFC, TSK, UST, RMOL, WSPD, XLAND, HFX, QFX)

  RTHRATEN=0.0
  RUBLTEN=0.0; RVBLTEN=0.0; RTHBLTEN=0.0; RQVBLTEN=0.0; RQCBLTEN=0.0; RQIBLTEN=0.0
  PEK=1.0E-3; PEP=1.0E-4; PEK_ADV=PEK; PEP_ADV=PEP
  EXCH_H=0.0; EXCH_M=0.0; PBLH=0.0; KPBL=0

  CALL eeps(u3d=U, v3d=V, t3d=T, qv3d=QV, qc3d=QC, qi3d=QI, qr3d=QR, qs3d=QS, qg3d=QG, &
            p3d=P, pi3d=PII, rho3d=RHO, rthraten=RTHRATEN, &
            rublten=RUBLTEN, rvblten=RVBLTEN, rthblten=RTHBLTEN, &
            rqvblten=RQVBLTEN, rqcblten=RQCBLTEN, rqiblten=RQIBLTEN, &
            pek=PEK, pep=PEP, dz8w=DZ, psfc=PSFC, qsfc=QSFC, tsk=TSK, ust=UST, rmol=RMOL, wspd=WSPD, &
            xland=XLAND, hfx=HFX, qfx=QFX, dt=60.0, dx=3000.0, itimestep=1, &
            exch_h=EXCH_H, exch_m=EXCH_M, pblh=PBLH, kpbl=KPBL, pek_adv=PEK_ADV, pep_adv=PEP_ADV, &
            ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
            ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
            its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

  CALL dump_case(case_id, name, U, V, T, TH, QV, QC, P, PII, DZ, RHO, &
                 PSFC, QSFC, TSK, UST, RMOL, WSPD, XLAND, HFX, QFX, PBLH, KPBL, &
                 RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, RQCBLTEN, RQIBLTEN, &
                 PEK, PEP, PEK_ADV, PEP_ADV, EXCH_H, EXCH_M)

CONTAINS

  SUBROUTINE build_case(cid, nm, Uu,Vv,Tt,Tht,Qq,Qc,Qi,Qr,QsA,Qg,Pp,Exner,DzA,RhoA, &
                        Ps,QsfcA,TskA,UstA,RmolA,WspdA,XlandA,HfxA,QfxA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Uu,Vv,Tt,Tht,Qq,Qc,Qi,Qr,QsA,Qg,Pp,Exner,DzA,RhoA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: Ps,QsfcA,TskA,UstA,RmolA,WspdA,XlandA,HfxA,QfxA
    REAL, DIMENSION(KX+1) :: zint, pint
    REAL, DIMENSION(KX) :: zmid, theta, qprof, uprof, vprof, temp, pfull
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, shear, z, tv, pi_mass
    REAL :: hfx0, qfx0, ust0, rmol0, xland0, tsk0
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
      q0=0.0140; qscale=2300.0; shear=0.0015; hfx0=350.0; qfx0=1.20E-4; ust0=0.55; rmol0=-0.02
      xland0=1.0; tsk0=302.0
    CASE (2)
      nm='strong_unstable'
      psfc0=100800.0; theta0=302.0; zml=1400.0; lapse_ml=0.0002; lapse_ft=0.0038
      q0=0.0160; qscale=2600.0; shear=0.0020; hfx0=600.0; qfx0=9.00E-5; ust0=0.80; rmol0=-0.04
      xland0=1.0; tsk0=305.0
    CASE (3)
      nm='stable_nocturnal'
      psfc0=100500.0; theta0=287.0; zml=150.0; lapse_ml=0.0100; lapse_ft=0.0060
      q0=0.0060; qscale=1600.0; shear=0.0060; hfx0=-60.0; qfx0=0.0; ust0=0.25; rmol0=0.05
      xland0=1.0; tsk0=285.0
    CASE (4)
      nm='stable_marine'
      psfc0=101200.0; theta0=289.0; zml=250.0; lapse_ml=0.0040; lapse_ft=0.0032
      q0=0.0100; qscale=2600.0; shear=0.0010; hfx0=-15.0; qfx0=2.00E-5; ust0=0.18; rmol0=0.02
      xland0=2.0; tsk0=288.0
    CASE (5)
      nm='neutral'
      psfc0=100000.0; theta0=296.0; zml=800.0; lapse_ml=0.0010; lapse_ft=0.0030
      q0=0.0090; qscale=2200.0; shear=0.0025; hfx0=0.0; qfx0=0.0; ust0=0.40; rmol0=0.0
      xland0=1.0; tsk0=296.0
    CASE DEFAULT
      nm='low_ust_edge'
      psfc0=99500.0; theta0=295.0; zml=700.0; lapse_ml=0.0010; lapse_ft=0.0040
      q0=0.0080; qscale=1800.0; shear=0.0010; hfx0=20.0; qfx0=1.00E-5; ust0=0.05; rmol0=-0.005
      xland0=1.0; tsk0=296.0
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

    Uu=0.0; Vv=0.0; Tt=0.0; Tht=0.0; Qq=0.0; Qc=0.0; Qi=0.0; Qr=0.0; QsA=0.0; Qg=0.0
    Pp=0.0; Exner=0.0; DzA=0.0; RhoA=0.0
    DO k = 1, KX
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Tt(1,k,1) = temp(k)
      Tht(1,k,1) = theta(k)
      Qq(1,k,1) = qprof(k)
      Pp(1,k,1) = pfull(k)
      Exner(1,k,1) = (pfull(k)/P1000)**ROVCP
      DzA(1,k,1) = zint(k+1)-zint(k)
      RhoA(1,k,1) = pfull(k)/(R_D*temp(k)*(1.0 + EP1*qprof(k)))
    END DO

    Ps(1,1)=psfc0; QsfcA(1,1)=qprof(1); TskA(1,1)=tsk0; UstA(1,1)=ust0
    RmolA(1,1)=rmol0; WspdA(1,1)=SQRT(uprof(1)*uprof(1)+vprof(1)*vprof(1))
    XlandA(1,1)=xland0; HfxA(1,1)=hfx0; QfxA(1,1)=qfx0
  END SUBROUTINE build_case

  SUBROUTINE dump_col(nm, arr)
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') nm,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_case(cid,nm,Uu,Vv,Tt,Tht,Qq,Qc,Pp,Exner,DzA,RhoA,Ps,QsfcA,TskA,UstA,RmolA,WspdA,XlandA,HfxA,QfxA,PblA,KpblA, &
                       Uten,Vten,Thten,Qvten,Qcten,Qiten,PekA,PepA,PekAdvA,PepAdvA,ExchH,ExchM)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uu,Vv,Tt,Tht,Qq,Qc,Pp,Exner,DzA,RhoA
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uten,Vten,Thten,Qvten,Qcten,Qiten,PekA,PepA,PekAdvA,PepAdvA,ExchH,ExchM
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: Ps,QsfcA,TskA,UstA,RmolA,WspdA,XlandA,HfxA,QfxA,PblA
    INTEGER, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: KpblA
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(nm)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,ES23.15)') 'DT=', 60.0
    WRITE(*,'(A,ES23.15)') 'PSFC=', Ps(1,1)
    WRITE(*,'(A,ES23.15)') 'QSFC=', QsfcA(1,1)
    WRITE(*,'(A,ES23.15)') 'TSK=', TskA(1,1)
    WRITE(*,'(A,ES23.15)') 'UST=', UstA(1,1)
    WRITE(*,'(A,ES23.15)') 'RMOL=', RmolA(1,1)
    WRITE(*,'(A,ES23.15)') 'WSPD=', WspdA(1,1)
    WRITE(*,'(A,ES23.15)') 'XLAND=', XlandA(1,1)
    WRITE(*,'(A,ES23.15)') 'HFX=', HfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'QFX=', QfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'PBL=', PblA(1,1)
    WRITE(*,'(A,I0)') 'KPBL=', KpblA(1,1)
    CALL dump_col('U', Uu)
    CALL dump_col('V', Vv)
    CALL dump_col('T', Tt)
    CALL dump_col('TH', Tht)
    CALL dump_col('QV', Qq)
    CALL dump_col('QC', Qc)
    CALL dump_col('P', Pp)
    CALL dump_col('PI', Exner)
    CALL dump_col('DZ', DzA)
    CALL dump_col('RHO', RhoA)
    CALL dump_col('RUBLTEN', Uten)
    CALL dump_col('RVBLTEN', Vten)
    CALL dump_col('RTHBLTEN', Thten)
    CALL dump_col('RQVBLTEN', Qvten)
    CALL dump_col('RQCBLTEN', Qcten)
    CALL dump_col('RQIBLTEN', Qiten)
    CALL dump_col('PEK', PekA)
    CALL dump_col('PEP', PepA)
    CALL dump_col('PEK_ADV', PekAdvA)
    CALL dump_col('PEP_ADV', PepAdvA)
    CALL dump_col('EXCH_H', ExchH)
    CALL dump_col('EXCH_M', ExchM)
  END SUBROUTINE dump_case
END PROGRAM eeps_oracle
