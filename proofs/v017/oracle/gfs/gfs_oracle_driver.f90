! =====================================================================
! v0.17 single-column GFS PBL oracle driver (bl_pbl_physics=3).
!
! Drives the UNMODIFIED pristine WRF module_bl_gfs.F:BL_GFS wrapper
! (which calls the internal MONINP -> TRIDI2/TRIDIN nonlocal-K GFS
! Hybrid-EDMF-ancestor PBL) on prescribed single-column soundings and
! dumps inputs + WRF GFS tendencies/diagnostics for JAX savepoint
! parity. This is a real WRF-module oracle (NOT a JAX self-compare); it
! is not a full coupled wrf.exe run.
!
! Built BOTH fp32 (default WRF REAL) and fp64 (-fdefault-real-8). The
! GFS PBL internals run at kind_phys=selected_real_kind(13,60) (real*8)
! REGARDLESS of the build's default REAL, so the v0.17 gate uses the
! fp64 savepoints at ~1e-12 (kind_phys-native).
!
! The six soundings reuse the v0.13 MRF / v0.6.0 YSU oracle regimes so
! the GFS gate spans the same unstable/stable/neutral land + marine
! regimes (identical case-builder for cross-scheme comparability).
! =====================================================================
PROGRAM gfs_oracle
  USE module_bl_gfs, ONLY : bl_gfs
  IMPLICIT NONE

  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: ROVCP  = R_D/CP
  REAL, PARAMETER :: ROVG   = R_D/G
  REAL, PARAMETER :: EP1    = R_V/R_D - 1.0
  REAL, PARAMETER :: KARMAN = 0.4
  REAL, PARAMETER :: P1000  = 1.0E5
  ! No ice path (NTRAC=2): P_QI < P_FIRST_SCALAR.
  INTEGER, PARAMETER :: P_QI = 1, P_FIRST_SCALAR = 2

  INTEGER, PARAMETER :: KX = 32
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,T,TH,QV,QC,QI,P,PII,DZ,ZF
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RUBLTEN,RVBLTEN,RTHBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RQVBLTEN,RQCBLTEN,RQIBLTEN
  REAL, DIMENSION(ims:ime,jms:jme) :: PSFC,UST,PBL,PSIM,PSIH,XLAND
  REAL, DIMENSION(ims:ime,jms:jme) :: HFX,QFX,TSK,GZ1OZ0,WSPD,BR
  INTEGER, DIMENSION(ims:ime,jms:jme) :: KPBL2D
  CHARACTER(LEN=32) :: arg, name
  INTEGER :: case_id

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_case(case_id, name, U, V, T, TH, QV, QC, QI, P, PII, DZ, ZF, &
                  PSFC, UST, PSIM, PSIH, XLAND, HFX, QFX, TSK, &
                  GZ1OZ0, WSPD, BR)

  RUBLTEN = 0.0; RVBLTEN = 0.0; RTHBLTEN = 0.0
  RQVBLTEN = 0.0; RQCBLTEN = 0.0; RQIBLTEN = 0.0
  PBL = 0.0; KPBL2D = 0

  CALL bl_gfs(u3d=U,v3d=V,th3d=TH,t3d=T,qv3d=QV,qc3d=QC,qi3d=QI,p3d=P,pi3d=PII, &
              rublten=RUBLTEN,rvblten=RVBLTEN,rthblten=RTHBLTEN, &
              rqvblten=RQVBLTEN,rqcblten=RQCBLTEN,rqiblten=RQIBLTEN, &
              cp=CP,g=G,rovcp=ROVCP,r=R_D,rovg=ROVG, &
              p_qi=P_QI,p_first_scalar=P_FIRST_SCALAR, &
              dz8w=DZ,z=ZF,psfc=PSFC, &
              ust=UST,pbl=PBL,psim=PSIM,psih=PSIH, &
              hfx=HFX,qfx=QFX,tsk=TSK,gz1oz0=GZ1OZ0,wspd=WSPD,br=BR, &
              dt=60.0,kpbl2d=KPBL2D,ep1=EP1,karman=KARMAN, &
              ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
              ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme, &
              its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte)

  CALL dump_case(case_id, name, U, V, T, QV, P, PII, DZ, ZF, &
                 PSFC, UST, PSIM, PSIH, XLAND, HFX, QFX, TSK, &
                 GZ1OZ0, WSPD, BR, PBL, KPBL2D, &
                 RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, RQCBLTEN)

CONTAINS

  SUBROUTINE build_case(cid, nm, Uu, Vv, Tt, Tht, Qq, Qc, Qi, Pp, Exner, Dz, Zfull, &
                        Ps, UstA, PsimA, PsihA, XlandA, HfxA, QfxA, TskA, &
                        Gz, WspdA, BrA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Uu,Vv,Tt,Tht,Qq,Qc,Qi,Pp,Exner,Dz,Zfull
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: Ps,UstA,PsimA,PsihA,XlandA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: HfxA,QfxA,TskA,Gz,WspdA,BrA
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
      psfc0=100000.0; theta0=300.0; zml=1100.0; lapse_ml=0.0003; lapse_ft=0.0045
      q0=0.0140; qscale=2300.0; shear=0.0015; hfx0=350.0; qfx0=1.20E-4; ust0=0.55; br0=-0.08
      xland0=1.0; znt0=0.10; tsk0=302.0
    CASE (2)
      nm='strong_unstable'
      psfc0=100800.0; theta0=302.0; zml=1400.0; lapse_ml=0.0002; lapse_ft=0.0038
      q0=0.0160; qscale=2600.0; shear=0.0020; hfx0=600.0; qfx0=9.00E-5; ust0=0.80; br0=-0.18
      xland0=1.0; znt0=0.08; tsk0=305.0
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

    Uu=0.0; Vv=0.0; Tt=0.0; Tht=0.0; Qq=0.0; Qc=0.0; Qi=0.0; Pp=0.0; Exner=0.0; Dz=0.0; Zfull=0.0
    DO k = 1, KX
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Tt(1,k,1) = temp(k)
      Tht(1,k,1) = theta(k)
      Qq(1,k,1) = qprof(k)
      Pp(1,k,1) = pfull(k)
      Exner(1,k,1) = (pfull(k)/P1000)**ROVCP
      Dz(1,k,1) = zint(k+1)-zint(k)
      Zfull(1,k,1) = zmid(k)
    END DO

    Ps(1,1)=psfc0; UstA(1,1)=ust0
    HfxA(1,1)=hfx0; QfxA(1,1)=qfx0; XlandA(1,1)=xland0; BrA(1,1)=br0; TskA(1,1)=tsk0
    WspdA(1,1)=SQRT(uprof(1)*uprof(1)+vprof(1)*vprof(1))
    Gz(1,1)=LOG(MAX(zmid(1),znt0)/znt0)
    PsimA(1,1)=0.0; PsihA(1,1)=0.0
  END SUBROUTINE build_case

  SUBROUTINE dump_col(nm, arr)
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') nm,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_case(cid, nm, Uu, Vv, Tt, Qq, Pp, Exner, Dz, Zfull, &
                       Ps, UstA, PsimA, PsihA, XlandA, HfxA, QfxA, TskA, &
                       Gz, WspdA, BrA, PblA, KpblA, &
                       Uten, Vten, Thten, Qvten, Qcten)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(IN) :: nm
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uu,Vv,Tt,Qq,Pp,Exner,Dz,Zfull
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uten,Vten,Thten,Qvten,Qcten
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: Ps,UstA,PsimA,PsihA,XlandA
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: HfxA,QfxA,TskA,Gz,WspdA,BrA,PblA
    INTEGER, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: KpblA
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(nm)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,ES23.15)') 'DT=', 60.0
    WRITE(*,'(A,ES23.15)') 'PSFC=', Ps(1,1)
    WRITE(*,'(A,ES23.15)') 'UST=', UstA(1,1)
    WRITE(*,'(A,ES23.15)') 'HFX=', HfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'QFX=', QfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'TSK=', TskA(1,1)
    WRITE(*,'(A,ES23.15)') 'GZ1OZ0=', Gz(1,1)
    WRITE(*,'(A,ES23.15)') 'WSPD=', WspdA(1,1)
    WRITE(*,'(A,ES23.15)') 'BR=', BrA(1,1)
    WRITE(*,'(A,ES23.15)') 'PSIM=', PsimA(1,1)
    WRITE(*,'(A,ES23.15)') 'PSIH=', PsihA(1,1)
    WRITE(*,'(A,ES23.15)') 'XLAND=', XlandA(1,1)
    WRITE(*,'(A,ES23.15)') 'PBL=', PblA(1,1)
    WRITE(*,'(A,I0)') 'KPBL=', KpblA(1,1)
    CALL dump_col('U', Uu)
    CALL dump_col('V', Vv)
    CALL dump_col('T', Tt)
    CALL dump_col('QV', Qq)
    CALL dump_col('P', Pp)
    CALL dump_col('PI', Exner)
    CALL dump_col('DZ', Dz)
    CALL dump_col('Z', Zfull)
    CALL dump_col('RUBLTEN', Uten)
    CALL dump_col('RVBLTEN', Vten)
    CALL dump_col('RTHBLTEN', Thten)
    CALL dump_col('RQVBLTEN', Qvten)
    CALL dump_col('RQCBLTEN', Qcten)
  END SUBROUTINE dump_case
END PROGRAM gfs_oracle
