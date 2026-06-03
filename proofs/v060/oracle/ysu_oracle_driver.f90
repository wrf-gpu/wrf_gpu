! =====================================================================
! v0.6.0 single-column YSU PBL oracle driver.
!
! Drives the UNMODIFIED WRF module_bl_ysu.F wrapper (which calls
! physics_mmm/bl_ysu.F90:bl_ysu_run) on prescribed single-column soundings.
! The dump captures inputs plus WRF YSU tendencies and diagnostics for
! JAX savepoint parity. This is a real WRF-module oracle, not a JAX
! self-compare; it is not a full coupled wrf.exe run.
! =====================================================================
PROGRAM ysu_oracle
  USE module_bl_ysu, ONLY : ysu
  IMPLICIT NONE

  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: XLV    = 2.5E6
  REAL, PARAMETER :: ROVCP  = R_D/CP
  REAL, PARAMETER :: ROVG   = R_D/G
  REAL, PARAMETER :: EP1    = R_V/R_D - 1.0
  REAL, PARAMETER :: EP2    = R_D/R_V
  REAL, PARAMETER :: KARMAN = 0.4
  REAL, PARAMETER :: P1000  = 1.0E5

  INTEGER, PARAMETER :: KX = 32
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,T,QV,QC,QI,P,PI,DZ,RTHRATEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RUBLTEN,RVBLTEN,RTHBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RQVBLTEN,RQCBLTEN,RQIBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: EXCH_H,EXCH_M
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: PDI
  REAL, DIMENSION(ims:ime,jms:jme) :: PSFC,ZNT,UST,HPBL,PSIM,PSIH,XLAND,HFX,QFX
  REAL, DIMENSION(ims:ime,jms:jme) :: WSPD,BR,WSTAR,DELTA,U10,V10,UOCE,VOCE,CTOPO,CTOPO2
  INTEGER, DIMENSION(ims:ime,jms:jme) :: KPBL2D
  CHARACTER(LEN=256) :: errmsg
  CHARACTER(LEN=32) :: arg, regime
  INTEGER :: errflg, case_id, topdown, idiff
  LOGICAL :: flag_qc, flag_qi, flag_bep

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_case(case_id, regime, U, V, T, QV, QC, QI, P, PDI, PI, DZ, &
                  PSFC, ZNT, UST, PSIM, PSIH, XLAND, HFX, QFX, WSPD, BR, &
                  U10, V10, UOCE, VOCE, CTOPO, CTOPO2)

  RTHRATEN = 0.0
  RUBLTEN = 0.0; RVBLTEN = 0.0; RTHBLTEN = 0.0
  RQVBLTEN = 0.0; RQCBLTEN = 0.0; RQIBLTEN = 0.0
  EXCH_H = 0.0; EXCH_M = 0.0
  HPBL = 0.0; WSTAR = 0.0; DELTA = 0.0; KPBL2D = 0
  topdown = 0
  idiff = 0
  flag_qc = .FALSE.
  flag_qi = .FALSE.
  flag_bep = .FALSE.

  CALL ysu(u3d=U,v3d=V,t3d=T,qv3d=QV,qc3d=QC,qi3d=QI,p3d=P,p3di=PDI,pi3d=PI, &
           rublten=RUBLTEN,rvblten=RVBLTEN,rthblten=RTHBLTEN, &
           rqvblten=RQVBLTEN,rqcblten=RQCBLTEN,rqiblten=RQIBLTEN, &
           flag_qc=flag_qc,flag_qi=flag_qi, &
           cp=CP,g=G,rovcp=ROVCP,rd=R_D,rovg=ROVG,ep1=EP1,ep2=EP2,karman=KARMAN,xlv=XLV,rv=R_V, &
           dz8w=DZ,psfc=PSFC, &
           znt=ZNT,ust=UST,hpbl=HPBL,psim=PSIM,psih=PSIH, &
           xland=XLAND,hfx=HFX,qfx=QFX,wspd=WSPD,br=BR, &
           dt=60.0,kpbl2d=KPBL2D, &
           exch_h=EXCH_H,exch_m=EXCH_M, &
           wstar=WSTAR,delta=DELTA, &
           u10=U10,v10=V10, &
           uoce=UOCE,voce=VOCE, &
           rthraten=RTHRATEN,ysu_topdown_pblmix=topdown, &
           ctopo=CTOPO,ctopo2=CTOPO2, &
           idiff=idiff,flag_bep=flag_bep, &
           ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
           ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme, &
           its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte, &
           errmsg=errmsg,errflg=errflg)

  CALL dump_case(case_id, regime, U, V, T, QV, P, PDI, PI, DZ, &
                 PSFC, ZNT, UST, PSIM, PSIH, XLAND, HFX, QFX, WSPD, BR, &
                 U10, V10, HPBL, KPBL2D, WSTAR, DELTA, &
                 RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, EXCH_H, EXCH_M, topdown)

CONTAINS

  SUBROUTINE build_case(cid, name, Uu, Vv, Tt, Qq, Qc, Qi, Pp, Pi_int, Exner, Dz, &
                        Ps, Z0, UstA, PsimA, PsihA, XlandA, HfxA, QfxA, WspdA, BrA, &
                        U10A, V10A, UoceA, VoceA, CtopoA, Ctopo2A)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Uu,Vv,Tt,Qq,Qc,Qi,Pp,Pi_int,Exner,Dz
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: Ps,Z0,UstA,PsimA,PsihA,XlandA,HfxA,QfxA,WspdA,BrA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: U10A,V10A,UoceA,VoceA,CtopoA,Ctopo2A
    REAL, DIMENSION(KX+1) :: zint, pint
    REAL, DIMENSION(KX) :: zmid, theta, qprof, uprof, vprof, temp, pfull
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, shear, z, tv, pi_mass
    REAL :: hfx0, qfx0, ust0, br0, xland0, znt0
    INTEGER :: k

    ztop = 12000.0
    zint(1) = 0.0
    DO k = 1, KX
      zint(k+1) = ztop * (REAL(k)/REAL(KX))**1.18
      zmid(k) = 0.5*(zint(k)+zint(k+1))
    END DO

    SELECT CASE (cid)
    CASE (1)
      name='unstable_daytime'
      psfc0=100000.0; theta0=300.0; zml=1100.0; lapse_ml=0.0003; lapse_ft=0.0045
      q0=0.0140; qscale=2300.0; shear=0.0015; hfx0=350.0; qfx0=1.20E-4; ust0=0.55; br0=-0.08
      xland0=1.0; znt0=0.10
    CASE (2)
      name='strong_unstable'
      psfc0=100800.0; theta0=302.0; zml=1400.0; lapse_ml=0.0002; lapse_ft=0.0038
      q0=0.0160; qscale=2600.0; shear=0.0020; hfx0=600.0; qfx0=9.00E-5; ust0=0.80; br0=-0.18
      xland0=1.0; znt0=0.08
    CASE (3)
      name='stable_nocturnal'
      psfc0=100500.0; theta0=287.0; zml=150.0; lapse_ml=0.0100; lapse_ft=0.0060
      q0=0.0060; qscale=1600.0; shear=0.0060; hfx0=-60.0; qfx0=0.0; ust0=0.25; br0=0.15
      xland0=1.0; znt0=0.05
    CASE (4)
      name='stable_marine'
      psfc0=101200.0; theta0=289.0; zml=250.0; lapse_ml=0.0040; lapse_ft=0.0032
      q0=0.0100; qscale=2600.0; shear=0.0010; hfx0=-15.0; qfx0=2.00E-5; ust0=0.18; br0=0.05
      xland0=2.0; znt0=0.001
    CASE (5)
      name='neutral'
      psfc0=100000.0; theta0=296.0; zml=800.0; lapse_ml=0.0010; lapse_ft=0.0030
      q0=0.0090; qscale=2200.0; shear=0.0025; hfx0=0.0; qfx0=0.0; ust0=0.40; br0=0.0
      xland0=1.0; znt0=0.08
    CASE DEFAULT
      name='low_ust_edge'
      psfc0=99500.0; theta0=295.0; zml=700.0; lapse_ml=0.0010; lapse_ft=0.0040
      q0=0.0080; qscale=1800.0; shear=0.0010; hfx0=20.0; qfx0=1.00E-5; ust0=0.05; br0=-0.01
      xland0=1.0; znt0=0.03
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

    Uu=0.0; Vv=0.0; Tt=0.0; Qq=0.0; Qc=0.0; Qi=0.0; Pp=0.0; Pi_int=0.0; Exner=0.0; Dz=0.0
    DO k = 1, KX
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Tt(1,k,1) = temp(k)
      Qq(1,k,1) = qprof(k)
      Pp(1,k,1) = pfull(k)
      Exner(1,k,1) = (pfull(k)/P1000)**ROVCP
      Dz(1,k,1) = zint(k+1)-zint(k)
      Pi_int(1,k,1) = pint(k)
    END DO
    Pi_int(1,KX+1,1) = pint(KX+1)

    Ps(1,1)=psfc0; Z0(1,1)=znt0; UstA(1,1)=ust0
    HfxA(1,1)=hfx0; QfxA(1,1)=qfx0; XlandA(1,1)=xland0; BrA(1,1)=br0
    WspdA(1,1)=SQRT(uprof(1)*uprof(1)+vprof(1)*vprof(1))
    U10A(1,1)=0.85*uprof(1); V10A(1,1)=0.85*vprof(1)
    UoceA(1,1)=0.0; VoceA(1,1)=0.0
    CtopoA(1,1)=1.0; Ctopo2A(1,1)=1.0
    PsimA(1,1)=LOG(MAX(zmid(1),znt0)/znt0)
    PsihA(1,1)=PsimA(1,1)
  END SUBROUTINE build_case

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col_interface(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte+1
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col_interface

  SUBROUTINE dump_case(cid, name, Uu, Vv, Tt, Qq, Pp, Pi_int, Exner, Dz, &
                       Ps, Z0, UstA, PsimA, PsihA, XlandA, HfxA, QfxA, WspdA, BrA, &
                       U10A, V10A, HpblA, KpblA, WstarA, DeltaA, &
                       Uten, Vten, Thten, Qvten, Kh, Km, topdown)
    INTEGER, INTENT(IN) :: cid, topdown
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uu,Vv,Tt,Qq,Pp,Pi_int,Exner,Dz
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uten,Vten,Thten,Qvten,Kh,Km
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: Ps,Z0,UstA,PsimA,PsihA,XlandA,HfxA,QfxA,WspdA,BrA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: U10A,V10A,HpblA,WstarA,DeltaA
    INTEGER, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: KpblA
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(name)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,I0)') 'YSU_TOPDOWN_PBLMIX=', topdown
    WRITE(*,'(A,ES23.15)') 'DT=', 60.0
    WRITE(*,'(A,ES23.15)') 'PSFC=', Ps(1,1)
    WRITE(*,'(A,ES23.15)') 'ZNT=', Z0(1,1)
    WRITE(*,'(A,ES23.15)') 'UST=', UstA(1,1)
    WRITE(*,'(A,ES23.15)') 'HFX=', HfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'QFX=', QfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'WSPD=', WspdA(1,1)
    WRITE(*,'(A,ES23.15)') 'BR=', BrA(1,1)
    WRITE(*,'(A,ES23.15)') 'PSIM=', PsimA(1,1)
    WRITE(*,'(A,ES23.15)') 'PSIH=', PsihA(1,1)
    WRITE(*,'(A,ES23.15)') 'XLAND=', XlandA(1,1)
    WRITE(*,'(A,ES23.15)') 'U10=', U10A(1,1)
    WRITE(*,'(A,ES23.15)') 'V10=', V10A(1,1)
    WRITE(*,'(A,ES23.15)') 'PBLH=', HpblA(1,1)
    WRITE(*,'(A,I0)') 'KPBL=', KpblA(1,1)
    WRITE(*,'(A,ES23.15)') 'WSTAR=', WstarA(1,1)
    WRITE(*,'(A,ES23.15)') 'DELTA=', DeltaA(1,1)
    CALL dump_col('U', Uu)
    CALL dump_col('V', Vv)
    CALL dump_col('T', Tt)
    CALL dump_col('QV', Qq)
    CALL dump_col('P', Pp)
    CALL dump_col_interface('PDI', Pi_int)
    CALL dump_col('PI', Exner)
    CALL dump_col('DZ', Dz)
    CALL dump_col('RUBLTEN', Uten)
    CALL dump_col('RVBLTEN', Vten)
    CALL dump_col('RTHBLTEN', Thten)
    CALL dump_col('RQVBLTEN', Qvten)
    CALL dump_col('EXCH_H', Kh)
    CALL dump_col('EXCH_M', Km)
  END SUBROUTINE dump_case
END PROGRAM ysu_oracle
