! v0.18 single-column TEMF PBL oracle driver (bl_pbl_physics=10).
PROGRAM temf_oracle
  USE module_bl_temf, ONLY : temfpbl
  IMPLICIT NONE

  REAL, PARAMETER :: G = 9.81
  REAL, PARAMETER :: R_D = 287.0
  REAL, PARAMETER :: CP = 7.0*R_D/2.0
  REAL, PARAMETER :: RCP = R_D/CP
  REAL, PARAMETER :: R_V = 461.6
  REAL, PARAMETER :: CPV = 1870.0
  REAL, PARAMETER :: XLV = 2.5E6
  REAL, PARAMETER :: ROVCP = R_D/CP
  REAL, PARAMETER :: EP1 = R_V/R_D - 1.0
  REAL, PARAMETER :: EP2 = R_D/R_V
  REAL, PARAMETER :: P1000 = 1.0E5

  INTEGER, PARAMETER :: KX = 32
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,T,TH,QV,QC,QI,P,PDI,PII,RHO,Z
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN,RQCBLTEN,RQIBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: KH_TEMF,KM_TEMF,TE_TEMF,SHF_TEMF,QF_TEMF,UW_TEMF,VW_TEMF
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: WUPD_TEMF,MF_TEMF,THUP_TEMF,QTUP_TEMF,QLUP_TEMF,CF3D_TEMF
  REAL, DIMENSION(ims:ime,jms:jme) :: PSFC,ZNT,HT,UST,ZOL,HOL,HPBL,PSIM,PSIH,XLAND,HFX,QFX,TSK,QSFC
  REAL, DIMENSION(ims:ime,jms:jme) :: GZ1OZ0,WSPD,BR,U10,V10,T2,FLHC,FLQC,EXCH_TEMF,FCOR
  REAL, DIMENSION(ims:ime,jms:jme) :: HD_TEMF,LCL_TEMF,HCT_TEMF,CFM_TEMF
  INTEGER, DIMENSION(ims:ime,jms:jme) :: KPBL2D
  CHARACTER(LEN=32) :: arg, name
  INTEGER :: case_id

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_case(case_id, name, U, V, T, TH, QV, QC, QI, P, PDI, PII, RHO, Z, &
                  PSFC, ZNT, HT, UST, ZOL, HOL, HPBL, PSIM, PSIH, XLAND, HFX, QFX, &
                  TSK, QSFC, GZ1OZ0, WSPD, BR, U10, V10, T2, FLHC, FLQC, EXCH_TEMF, FCOR)

  RUBLTEN=0.0; RVBLTEN=0.0; RTHBLTEN=0.0; RQVBLTEN=0.0; RQCBLTEN=0.0; RQIBLTEN=0.0
  KH_TEMF=0.0; KM_TEMF=0.0; TE_TEMF=1.0E-3
  SHF_TEMF=0.0; QF_TEMF=0.0; UW_TEMF=0.0; VW_TEMF=0.0
  WUPD_TEMF=0.0; MF_TEMF=0.0; THUP_TEMF=0.0; QTUP_TEMF=0.0; QLUP_TEMF=0.0; CF3D_TEMF=0.0
  HD_TEMF=0.0; LCL_TEMF=0.0; HCT_TEMF=0.0; CFM_TEMF=0.0; KPBL2D=0

  CALL temfpbl(u3d=U, v3d=V, th3d=TH, t3d=T, qv3d=QV, qc3d=QC, qi3d=QI, &
               p3d=P, p3di=PDI, pi3d=PII, rho=RHO, &
               rublten=RUBLTEN, rvblten=RVBLTEN, rthblten=RTHBLTEN, &
               rqvblten=RQVBLTEN, rqcblten=RQCBLTEN, rqiblten=RQIBLTEN, flag_qi=.FALSE., &
               g=G, cp=CP, rcp=RCP, r_d=R_D, r_v=R_V, cpv=CPV, z=Z, xlv=XLV, psfc=PSFC, &
               znt=ZNT, ht=HT, ust=UST, zol=ZOL, hol=HOL, hpbl=HPBL, psim=PSIM, psih=PSIH, &
               xland=XLAND, hfx=HFX, qfx=QFX, tsk=TSK, qsfc=QSFC, gz1oz0=GZ1OZ0, &
               wspd=WSPD, br=BR, dt=60.0, dtmin=1.0, kpbl2d=KPBL2D, &
               svp1=0.6112, svp2=17.67, svp3=29.65, svpt0=273.15, &
               ep1=EP1, ep2=EP2, karman=0.4, eomeg=7.292E-5, stbolt=5.670374419E-8, &
               kh_temf=KH_TEMF, km_temf=KM_TEMF, u10=U10, v10=V10, t2=T2, &
               te_temf=TE_TEMF, shf_temf=SHF_TEMF, qf_temf=QF_TEMF, uw_temf=UW_TEMF, vw_temf=VW_TEMF, &
               wupd_temf=WUPD_TEMF, mf_temf=MF_TEMF, thup_temf=THUP_TEMF, qtup_temf=QTUP_TEMF, &
               qlup_temf=QLUP_TEMF, cf3d_temf=CF3D_TEMF, cfm_temf=CFM_TEMF, &
               hd_temf=HD_TEMF, lcl_temf=LCL_TEMF, hct_temf=HCT_TEMF, &
               flhc=FLHC, flqc=FLQC, exch_temf=EXCH_TEMF, fCor=FCOR, &
               ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
               ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
               its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

  CALL dump_case(case_id, name, U, V, T, TH, QV, QC, P, PDI, PII, RHO, Z, &
                 PSFC, ZNT, UST, HPBL, HFX, QFX, TSK, QSFC, GZ1OZ0, WSPD, BR, &
                 U10, V10, T2, HD_TEMF, LCL_TEMF, HCT_TEMF, CFM_TEMF, KPBL2D, &
                 RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, RQCBLTEN, &
                 KH_TEMF, KM_TEMF, TE_TEMF, SHF_TEMF, QF_TEMF, UW_TEMF, VW_TEMF, &
                 WUPD_TEMF, MF_TEMF, THUP_TEMF, QTUP_TEMF, QLUP_TEMF, CF3D_TEMF)

CONTAINS

  SUBROUTINE build_case(cid, nm, Uu, Vv, Tt, Tht, Qq, Qc, Qi, Pp, PdiA, Exner, RhoA, ZA, &
                        Ps, Z0, HtA, UstA, ZolA, HolA, HpblA, PsimA, PsihA, XlandA, HfxA, QfxA, &
                        TskA, QsfcA, Gz, WspdA, BrA, U10A, V10A, T2A, FlhcA, FlqcA, ExchA, FCorA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Uu,Vv,Tt,Tht,Qq,Qc,Qi,Pp,PdiA,Exner,RhoA,ZA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: Ps,Z0,HtA,UstA,ZolA,HolA,HpblA,PsimA,PsihA,XlandA,HfxA,QfxA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: TskA,QsfcA,Gz,WspdA,BrA,U10A,V10A,T2A,FlhcA,FlqcA,ExchA,FCorA
    REAL, DIMENSION(KX+1) :: zint, pint
    REAL, DIMENSION(KX) :: zmid, theta, qprof, uprof, vprof, temp, pfull
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, shear, z, tv, pi_mass
    REAL :: hfx0, qfx0, ust0, br0, xland0, znt0, tsk0
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
      psfc0=100000.0; theta0=303.0; zml=1100.0; lapse_ml=-0.0015; lapse_ft=0.0045
      q0=0.0140; qscale=2300.0; shear=0.0015; hfx0=350.0; qfx0=1.20E-4; ust0=0.55; br0=-0.08
      xland0=1.0; znt0=0.10; tsk0=306.0
    CASE (2)
      nm='strong_unstable'
      psfc0=100800.0; theta0=306.0; zml=1400.0; lapse_ml=-0.0020; lapse_ft=0.0038
      q0=0.0160; qscale=2600.0; shear=0.0020; hfx0=600.0; qfx0=9.00E-5; ust0=0.80; br0=-0.18
      xland0=1.0; znt0=0.08; tsk0=310.0
    CASE (3)
      nm='stable_nocturnal'
      psfc0=100500.0; theta0=287.0; zml=150.0; lapse_ml=0.0100; lapse_ft=0.0060
      q0=0.0060; qscale=1600.0; shear=0.0060; hfx0=-60.0; qfx0=0.0; ust0=0.25; br0=0.15
      xland0=1.0; znt0=0.05; tsk0=285.0
    CASE (4)
      nm='stable_marine'
      psfc0=101200.0; theta0=289.0; zml=250.0; lapse_ml=0.0040; lapse_ft=0.0032
      q0=0.0100; qscale=2600.0; shear=0.0010; hfx0=-15.0; qfx0=2.00E-5; ust0=0.18; br0=0.05
      xland0=2.0; znt0=0.001; tsk0=288.0
    CASE (5)
      nm='neutral'
      psfc0=100000.0; theta0=296.0; zml=800.0; lapse_ml=0.0010; lapse_ft=0.0030
      q0=0.0090; qscale=2200.0; shear=0.0025; hfx0=0.0; qfx0=0.0; ust0=0.40; br0=0.0
      xland0=1.0; znt0=0.08; tsk0=296.0
    CASE DEFAULT
      nm='low_ust_edge'
      psfc0=99500.0; theta0=295.0; zml=700.0; lapse_ml=0.0010; lapse_ft=0.0040
      q0=0.0080; qscale=1800.0; shear=0.0010; hfx0=20.0; qfx0=1.00E-5; ust0=0.05; br0=-0.01
      xland0=1.0; znt0=0.03; tsk0=296.0
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

    Uu=0.0; Vv=0.0; Tt=0.0; Tht=0.0; Qq=0.0; Qc=0.0; Qi=0.0
    Pp=0.0; PdiA=0.0; Exner=0.0; RhoA=0.0; ZA=0.0
    DO k = 1, KX
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Tt(1,k,1) = temp(k)
      Tht(1,k,1) = theta(k)
      Qq(1,k,1) = qprof(k)
      Pp(1,k,1) = pfull(k)
      PdiA(1,k,1) = pint(k)
      Exner(1,k,1) = (pfull(k)/P1000)**ROVCP
      RhoA(1,k,1) = pfull(k)/(R_D*temp(k)*(1.0 + EP1*qprof(k)))
      ZA(1,k,1) = zmid(k)
    END DO
    PdiA(1,KX+1,1) = pint(KX+1)
    ZA(1,KX+1,1) = zint(KX+1)

    Ps(1,1)=psfc0; Z0(1,1)=znt0; HtA(1,1)=0.0; UstA(1,1)=ust0
    ZolA(1,1)=0.0; HolA(1,1)=0.0; HpblA(1,1)=1000.0
    PsimA(1,1)=0.0; PsihA(1,1)=0.0; XlandA(1,1)=xland0
    HfxA(1,1)=hfx0; QfxA(1,1)=qfx0; TskA(1,1)=tsk0; QsfcA(1,1)=qprof(1)
    Gz(1,1)=LOG(MAX(zmid(1),znt0)/znt0)
    WspdA(1,1)=SQRT(uprof(1)*uprof(1)+vprof(1)*vprof(1)); BrA(1,1)=br0
    U10A(1,1)=uprof(1); V10A(1,1)=vprof(1); T2A(1,1)=tsk0
    FlhcA(1,1)=0.0; FlqcA(1,1)=0.0; ExchA(1,1)=0.0; FCorA(1,1)=1.0E-4
  END SUBROUTINE build_case

  SUBROUTINE dump_col(nm, arr)
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') nm,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_case(cid, nm, Uu,Vv,Tt,Tht,Qq,Qc,Pp,PdiA,Exner,RhoA,ZA, &
                       Ps,Z0,UstA,HpblA,HfxA,QfxA,TskA,QsfcA,Gz,WspdA,BrA, &
                       U10A,V10A,T2A,HdA,LclA,HctA,CfmA,KpblA, &
                       Uten,Vten,Thten,Qvten,Qcten,KhA,KmA,TeA,ShfA,QfA,UwA,VwA,WupdA,MfA,ThupA,QtupA,QlupA,Cf3dA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uu,Vv,Tt,Tht,Qq,Qc,Pp,PdiA,Exner,RhoA,ZA
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uten,Vten,Thten,Qvten,Qcten,KhA,KmA,TeA,ShfA,QfA,UwA,VwA
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: WupdA,MfA,ThupA,QtupA,QlupA,Cf3dA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: Ps,Z0,UstA,HpblA,HfxA,QfxA,TskA,QsfcA,Gz,WspdA,BrA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: U10A,V10A,T2A,HdA,LclA,HctA,CfmA
    INTEGER, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: KpblA
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(nm)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,ES23.15)') 'DT=', 60.0
    WRITE(*,'(A,ES23.15)') 'PSFC=', Ps(1,1)
    WRITE(*,'(A,ES23.15)') 'ZNT=', Z0(1,1)
    WRITE(*,'(A,ES23.15)') 'UST=', UstA(1,1)
    WRITE(*,'(A,ES23.15)') 'HPBL=', HpblA(1,1)
    WRITE(*,'(A,ES23.15)') 'HFX=', HfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'QFX=', QfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'TSK=', TskA(1,1)
    WRITE(*,'(A,ES23.15)') 'QSFC=', QsfcA(1,1)
    WRITE(*,'(A,ES23.15)') 'GZ1OZ0=', Gz(1,1)
    WRITE(*,'(A,ES23.15)') 'WSPD=', WspdA(1,1)
    WRITE(*,'(A,ES23.15)') 'BR=', BrA(1,1)
    WRITE(*,'(A,ES23.15)') 'U10=', U10A(1,1)
    WRITE(*,'(A,ES23.15)') 'V10=', V10A(1,1)
    WRITE(*,'(A,ES23.15)') 'T2=', T2A(1,1)
    WRITE(*,'(A,ES23.15)') 'HD=', HdA(1,1)
    WRITE(*,'(A,ES23.15)') 'LCL=', LclA(1,1)
    WRITE(*,'(A,ES23.15)') 'HCT=', HctA(1,1)
    WRITE(*,'(A,ES23.15)') 'CFM=', CfmA(1,1)
    WRITE(*,'(A,I0)') 'KPBL=', KpblA(1,1)
    CALL dump_col('U', Uu)
    CALL dump_col('V', Vv)
    CALL dump_col('T', Tt)
    CALL dump_col('TH', Tht)
    CALL dump_col('QV', Qq)
    CALL dump_col('QC', Qc)
    CALL dump_col('P', Pp)
    CALL dump_col('PDI', PdiA)
    CALL dump_col('PI', Exner)
    CALL dump_col('RHO', RhoA)
    CALL dump_col('Z', ZA)
    CALL dump_col('RUBLTEN', Uten)
    CALL dump_col('RVBLTEN', Vten)
    CALL dump_col('RTHBLTEN', Thten)
    CALL dump_col('RQVBLTEN', Qvten)
    CALL dump_col('RQCBLTEN', Qcten)
    CALL dump_col('KH_TEMF', KhA)
    CALL dump_col('KM_TEMF', KmA)
    CALL dump_col('TE_TEMF', TeA)
    CALL dump_col('SHF_TEMF', ShfA)
    CALL dump_col('QF_TEMF', QfA)
    CALL dump_col('UW_TEMF', UwA)
    CALL dump_col('VW_TEMF', VwA)
    CALL dump_col('WUPD_TEMF', WupdA)
    CALL dump_col('MF_TEMF', MfA)
    CALL dump_col('THUP_TEMF', ThupA)
    CALL dump_col('QTUP_TEMF', QtupA)
    CALL dump_col('QLUP_TEMF', QlupA)
    CALL dump_col('CF3D_TEMF', Cf3dA)
  END SUBROUTINE dump_case
END PROGRAM temf_oracle
