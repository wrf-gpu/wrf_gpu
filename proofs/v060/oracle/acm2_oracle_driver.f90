! =====================================================================
! v0.6.0 single-column ACM2 PBL oracle driver.
!
! Drives the UNMODIFIED WRF module_bl_acm.F ACM2 implementation on
! prescribed single-column soundings. The dump captures inputs plus WRF
! ACM2 tendencies and diagnostics for JAX savepoint parity. This is a
! real WRF-module oracle, not a JAX self-compare; it is not a full
! coupled wrf.exe run.
! =====================================================================
PROGRAM acm2_oracle
  USE module_bl_acm, ONLY : ACMPBL
  IMPLICIT NONE

  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: ROVCP  = R_D/CP
  REAL, PARAMETER :: EP1    = R_V/R_D - 1.0
  REAL, PARAMETER :: P1000  = 1.0E5

  INTEGER, PARAMETER :: KX = 32
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,TH,T,QV,QC,QI,P,DZ,RR
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RUBLTEN,RVBLTEN,RTHBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RQVBLTEN,RQCBLTEN,RQIBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: EXCH_H,EXCH_M
  REAL, DIMENSION(ims:ime,jms:jme) :: PSFC,UST,HFX,QFX,TSK,PBLH,REGIME
  REAL, DIMENSION(ims:ime,jms:jme) :: GZ1OZ0,WSPD,PSIM,MUT,RMOL
  INTEGER, DIMENSION(ims:ime,jms:jme) :: KPBL2D
  CHARACTER(LEN=256) :: arg
  CHARACTER(LEN=48) :: regime_name
  INTEGER :: case_id, xtime
  REAL :: pblh_initial

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_case(case_id, regime_name, U, V, TH, T, QV, QC, QI, P, DZ, RR, &
                  PSFC, UST, HFX, QFX, TSK, PBLH, GZ1OZ0, WSPD, PSIM, MUT)

  RUBLTEN = 0.0; RVBLTEN = 0.0; RTHBLTEN = 0.0
  RQVBLTEN = 0.0; RQCBLTEN = 0.0; RQIBLTEN = 0.0
  EXCH_H = 0.0; EXCH_M = 0.0
  REGIME = 0.0; RMOL = 0.0; KPBL2D = 0
  xtime = 60
  pblh_initial = PBLH(1,1)

  CALL ACMPBL(XTIME=xtime, DTPBL=60.0, U3D=U, V3D=V, PP3D=P, DZ8W=DZ, &
              TH3D=TH, T3D=T, QV3D=QV, QC3D=QC, QI3D=QI, RR3D=RR, &
              UST=UST, HFX=HFX, QFX=QFX, TSK=TSK, PSFC=PSFC, &
              EP1=EP1, G=G, ROVCP=ROVCP, RD=R_D, CPD=CP, &
              PBLH=PBLH, KPBL2D=KPBL2D, EXCH_H=EXCH_H, EXCH_M=EXCH_M, &
              REGIME=REGIME, GZ1OZ0=GZ1OZ0, WSPD=WSPD, PSIM=PSIM, &
              MUT=MUT, RMOL=RMOL, RUBLTEN=RUBLTEN, RVBLTEN=RVBLTEN, &
              RTHBLTEN=RTHBLTEN, RQVBLTEN=RQVBLTEN, RQCBLTEN=RQCBLTEN, &
              RQIBLTEN=RQIBLTEN, ids=ids, ide=ide, jds=jds, jde=jde, &
              kds=kds, kde=kde, ims=ims, ime=ime, jms=jms, jme=jme, &
              kms=kms, kme=kme, its=its, ite=ite, jts=jts, jte=jte, &
              kts=kts, kte=kte)

  CALL dump_case(case_id, regime_name, xtime, pblh_initial, U, V, TH, T, QV, QC, QI, P, DZ, RR, &
                 PSFC, UST, HFX, QFX, TSK, PBLH, REGIME, RMOL, GZ1OZ0, WSPD, PSIM, MUT, KPBL2D, &
                 RUBLTEN, RVBLTEN, RTHBLTEN, RQVBLTEN, RQCBLTEN, RQIBLTEN, EXCH_H, EXCH_M)

CONTAINS

  SUBROUTINE build_case(cid, name, Uu, Vv, Th, Tt, Qq, Qc, Qi, Pp, Dz, Rr, &
                        Ps, UstA, HfxA, QfxA, TskA, PblA, GzA, WspdA, PsimA, MutA)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=48), INTENT(OUT) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Uu,Vv,Th,Tt,Qq,Qc,Qi,Pp,Dz,Rr
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: Ps,UstA,HfxA,QfxA,TskA,PblA,GzA,WspdA,PsimA,MutA
    REAL, DIMENSION(KX+1) :: zint, pint
    REAL, DIMENSION(KX) :: zmid, theta, qprof, uprof, vprof, temp, pfull
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, shear, z, tv, pi_mass
    REAL :: hfx0, qfx0, ust0, pbl0, znt0
    INTEGER :: k

    ztop = 9000.0
    zint(1) = 0.0
    DO k = 1, KX
      zint(k+1) = ztop * (REAL(k)/REAL(KX))**1.22
      zmid(k) = 0.5*(zint(k)+zint(k+1))
    END DO

    SELECT CASE (cid)
    CASE (1)
      name='unstable_acm2_daytime'
      psfc0=100000.0; theta0=302.0; zml=1200.0; lapse_ml=-0.00045; lapse_ft=0.0042
      q0=0.0140; qscale=2300.0; shear=0.0016; hfx0=360.0; qfx0=1.15E-4; ust0=0.55
      pbl0=900.0; znt0=0.10
    CASE (2)
      name='strong_unstable_acm2'
      psfc0=100800.0; theta0=304.0; zml=1600.0; lapse_ml=-0.00080; lapse_ft=0.0038
      q0=0.0160; qscale=2600.0; shear=0.0020; hfx0=650.0; qfx0=9.00E-5; ust0=0.80
      pbl0=1200.0; znt0=0.08
    CASE (3)
      name='stable_nocturnal_local'
      psfc0=100500.0; theta0=287.0; zml=180.0; lapse_ml=0.0100; lapse_ft=0.0058
      q0=0.0060; qscale=1600.0; shear=0.0060; hfx0=-60.0; qfx0=0.0; ust0=0.25
      pbl0=160.0; znt0=0.05
    CASE (4)
      name='stable_marine_local'
      psfc0=101200.0; theta0=289.0; zml=250.0; lapse_ml=0.0040; lapse_ft=0.0032
      q0=0.0100; qscale=2600.0; shear=0.0010; hfx0=-15.0; qfx0=2.00E-5; ust0=0.18
      pbl0=250.0; znt0=0.001
    CASE (5)
      name='neutral_local'
      psfc0=100000.0; theta0=296.0; zml=800.0; lapse_ml=0.0010; lapse_ft=0.0030
      q0=0.0090; qscale=2200.0; shear=0.0025; hfx0=0.0; qfx0=0.0; ust0=0.40
      pbl0=600.0; znt0=0.08
    CASE DEFAULT
      name='weak_unstable_transition'
      psfc0=99500.0; theta0=298.0; zml=900.0; lapse_ml=-0.00020; lapse_ft=0.0040
      q0=0.0080; qscale=1800.0; shear=0.0010; hfx0=80.0; qfx0=3.00E-5; ust0=0.35
      pbl0=700.0; znt0=0.03
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

    Uu=0.0; Vv=0.0; Th=0.0; Tt=0.0; Qq=0.0; Qc=0.0; Qi=0.0; Pp=0.0; Dz=0.0; Rr=0.0
    DO k = 1, KX
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Th(1,k,1) = theta(k)
      Tt(1,k,1) = temp(k)
      Qq(1,k,1) = qprof(k)
      Pp(1,k,1) = pfull(k)
      Dz(1,k,1) = zint(k+1)-zint(k)
      Rr(1,k,1) = pfull(k)/(R_D*temp(k)*(1.0 + EP1*qprof(k)))
    END DO

    Ps(1,1)=psfc0; UstA(1,1)=ust0; HfxA(1,1)=hfx0; QfxA(1,1)=qfx0
    TskA(1,1)=temp(1); PblA(1,1)=pbl0
    WspdA(1,1)=SQRT(uprof(1)*uprof(1)+vprof(1)*vprof(1))
    GzA(1,1)=LOG(MAX(zmid(1),znt0)/znt0)
    PsimA(1,1)=GzA(1,1)
    MutA(1,1)=psfc0-pint(KX+1)
  END SUBROUTINE build_case

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_case(cid, name, xt, pbl0, Uu, Vv, Th, Tt, Qq, Qc, Qi, Pp, Dz, Rr, &
                       Ps, UstA, HfxA, QfxA, TskA, PblA, RegA, RmolA, GzA, WspdA, PsimA, MutA, KpblA, &
                       Uten, Vten, Thten, Qvten, Qcten, Qiten, Kh, Km)
    INTEGER, INTENT(IN) :: cid, xt
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, INTENT(IN) :: pbl0
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uu,Vv,Th,Tt,Qq,Qc,Qi,Pp,Dz,Rr
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uten,Vten,Thten,Qvten,Qcten,Qiten,Kh,Km
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: Ps,UstA,HfxA,QfxA,TskA,PblA,RegA,RmolA,GzA,WspdA,PsimA,MutA
    INTEGER, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: KpblA
    INTEGER :: nonlocal_flag

    nonlocal_flag = 0
    IF (RegA(1,1) .EQ. 4.0) nonlocal_flag = 1
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME_NAME=', TRIM(name)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,I0)') 'XTIME=', xt
    WRITE(*,'(A,ES23.15)') 'DT=', 60.0
    WRITE(*,'(A,ES23.15)') 'PSFC=', Ps(1,1)
    WRITE(*,'(A,ES23.15)') 'PBLH_INITIAL=', pbl0
    WRITE(*,'(A,ES23.15)') 'PBLH=', PblA(1,1)
    WRITE(*,'(A,I0)') 'KPBL=', KpblA(1,1)
    WRITE(*,'(A,ES23.15)') 'REGIME=', RegA(1,1)
    WRITE(*,'(A,I0)') 'NOCONV=', nonlocal_flag
    WRITE(*,'(A,ES23.15)') 'RMOL=', RmolA(1,1)
    WRITE(*,'(A,ES23.15)') 'UST=', UstA(1,1)
    WRITE(*,'(A,ES23.15)') 'HFX=', HfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'QFX=', QfxA(1,1)
    WRITE(*,'(A,ES23.15)') 'TSK=', TskA(1,1)
    WRITE(*,'(A,ES23.15)') 'WSPD=', WspdA(1,1)
    WRITE(*,'(A,ES23.15)') 'MUT=', MutA(1,1)
    WRITE(*,'(A,ES23.15)') 'PSIM=', PsimA(1,1)
    WRITE(*,'(A,ES23.15)') 'GZ1OZ0=', GzA(1,1)
    CALL dump_col('U', Uu)
    CALL dump_col('V', Vv)
    CALL dump_col('THETA', Th)
    CALL dump_col('T', Tt)
    CALL dump_col('QV', Qq)
    CALL dump_col('QC', Qc)
    CALL dump_col('QI', Qi)
    CALL dump_col('P', Pp)
    CALL dump_col('DZ', Dz)
    CALL dump_col('RR', Rr)
    CALL dump_col('RUBLTEN', Uten)
    CALL dump_col('RVBLTEN', Vten)
    CALL dump_col('RTHBLTEN', Thten)
    CALL dump_col('RQVBLTEN', Qvten)
    CALL dump_col('RQCBLTEN', Qcten)
    CALL dump_col('RQIBLTEN', Qiten)
    CALL dump_col('EXCH_H', Kh)
    CALL dump_col('EXCH_M', Km)
  END SUBROUTINE dump_case
END PROGRAM acm2_oracle

SUBROUTINE wrf_error_fatal(message)
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: message
  WRITE(*,'(A,A)') 'WRF_ERROR_FATAL=', TRIM(message)
  STOP 2
END SUBROUTINE wrf_error_fatal
