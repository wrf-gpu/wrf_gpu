! =====================================================================
! v0.13 single-column KIM Simplified Arakawa-Schubert (KSAS,
! WRF cu_physics=14) oracle driver.
!
! Drives the UNMODIFIED WRF module_cu_ksas.F (cu_ksas, which wraps the
! GFS-lineage core nsas2d) on prescribed WRF-layout columns and dumps:
!   * full input state in WRF bottom-up orientation
!   * the WRF output tendencies RTH/RQV/RQC/RQI/RU/RV CUTEN
!   * RAINCV, PRATEC
!
! Compiled with -fdefault-real-8 -fdefault-double-8 so WRF's REAL
! (default single) becomes double -> a true fp64 reference for the
! (fp64) JAX kernel. CPU-only, cores 0-3.
!
! Usage: ./ksas_oracle <case_id>; output = flat key=value text.
! Sounding generator identical to the New-Tiedtke oracle (same regimes).
! =====================================================================
PROGRAM ksas_oracle
  USE module_cu_ksas, ONLY : cu_ksas
  IMPLICIT NONE

  REAL, PARAMETER :: G = 9.81
  REAL, PARAMETER :: R_D = 287.0
  REAL, PARAMETER :: R_V = 461.6
  REAL, PARAMETER :: CP = 7.0 * R_D / 2.0
  REAL, PARAMETER :: P1000 = 1.0E5
  REAL, PARAMETER :: ROVCP = R_D / CP
  REAL, PARAMETER :: XLV = 2.5E6
  REAL, PARAMETER :: XLS = 2.85E6
  REAL, PARAMETER :: CLIQ = 4190.0
  REAL, PARAMETER :: CPV = 1870.0
  REAL, PARAMETER :: CICE = 2106.0
  REAL, PARAMETER :: PSAT = 610.78
  REAL, PARAMETER :: EP_1 = R_V/R_D - 1.0
  REAL, PARAMETER :: EP_2 = R_D/R_V

  INTEGER, PARAMETER :: KX = 40
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,W,T,QV,QC,QI,PII,RHO
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: DZ8W,PCPS,P8W
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RTHCUTEN,RQVCUTEN,RQCCUTEN,RQICUTEN,RUCUTEN,RVCUTEN
  REAL, DIMENSION(ims:ime,jms:jme) :: RAINCV, PRATEC, QFX, HFX, XLAND, HBOT, HTOP
  REAL, DIMENSION(ims:ime,jms:jme) :: HPBL, HPBL_HOLD
  REAL, DIMENSION(kms:kme) :: ZNU
  LOGICAL, DIMENSION(ims:ime,jms:jme) :: CU_ACT_FLAG

  REAL :: DT, DX, PGCON
  INTEGER :: STEPCU, ITIMESTEP, MP_PHYSICS, DX_FACTOR_NSAS
  INTEGER :: P_QC, P_QI, P_FIRST_SCALAR
  INTEGER :: case_id, k
  CHARACTER(LEN=32) :: arg
  LOGICAL :: F_QC,F_QI

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DX = 9000.0
  DT = 54.0
  STEPCU = 5
  ITIMESTEP = 2
  MP_PHYSICS = 8        ! Thompson (controls ncloud branch; qc+qi present)
  DX_FACTOR_NSAS = 0
  PGCON = 0.55
  P_QC = 2; P_QI = 3; P_FIRST_SCALAR = 2
  F_QC=.TRUE.; F_QI=.TRUE.

  U=0.0; V=0.0; W=0.0; T=0.0; QV=0.0; QC=0.0; QI=0.0; PII=1.0; RHO=0.0
  DZ8W=0.0; PCPS=0.0; P8W=0.0; ZNU=0.0
  RTHCUTEN=0.0; RQVCUTEN=0.0; RQCCUTEN=0.0; RQICUTEN=0.0; RUCUTEN=0.0; RVCUTEN=0.0
  RAINCV=0.0; PRATEC=0.0; QFX=0.0; HFX=0.0; XLAND=1.0; CU_ACT_FLAG=.TRUE.
  HBOT=0.0; HTOP=0.0; HPBL=1000.0; HPBL_HOLD=1000.0

  CALL build_sounding(case_id, T, QV, QC, QI, PCPS, P8W, DZ8W, RHO, PII, U, V, W, &
                      QFX, HFX, XLAND, HPBL, ZNU)
  HPBL_HOLD = HPBL

  CALL cu_ksas(dt=DT,dx=DX,p3di=P8W,p3d=PCPS,pi3d=PII,qc3d=QC,qi3d=QI,rho3d=RHO, &
       itimestep=ITIMESTEP,stepcu=STEPCU, &
       hbot=HBOT,htop=HTOP,cu_act_flag=CU_ACT_FLAG, &
       rthcuten=RTHCUTEN,rqvcuten=RQVCUTEN,rqccuten=RQCCUTEN,rqicuten=RQICUTEN, &
       rucuten=RUCUTEN,rvcuten=RVCUTEN, &
       qv3d=QV,t3d=T,raincv=RAINCV,pratec=PRATEC,xland=XLAND,dz8w=DZ8W,w=W,u3d=U,v3d=V, &
       hpbl=HPBL,hfx=HFX,qfx=QFX, &
       hpbl_hold=HPBL_HOLD,znu=ZNU, &
       mp_physics=MP_PHYSICS,dx_factor_nsas=DX_FACTOR_NSAS, &
       p_qc=P_QC,p_qi=P_QI,p_first_scalar=P_FIRST_SCALAR, &
       pgcon=PGCON, &
       cp=CP,cliq=CLIQ,cpv=CPV,g=G,xlv=XLV,r_d=R_D,r_v=R_V,ep_1=EP_1,ep_2=EP_2, &
       cice=CICE,xls=XLS,psat=PSAT,f_qi=F_QI,f_qc=F_QC, &
       ids=ids,ide=ide, jds=jds,jde=jde, kds=kds,kde=kde, &
       ims=ims,ime=ime, jms=jms,jme=jme, kms=kms,kme=kme, &
       its=its,ite=ite, jts=jts,jte=jte, kts=kts,kte=kte)

  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  WRITE(*,'(A,ES23.15)') 'DX=', DX
  WRITE(*,'(A,I0)') 'STEPCU=', STEPCU
  WRITE(*,'(A,I0)') 'ITIMESTEP=', ITIMESTEP
  WRITE(*,'(A,ES23.15)') 'RAINCV=', RAINCV(1,1)
  WRITE(*,'(A,ES23.15)') 'PRATEC=', PRATEC(1,1)
  WRITE(*,'(A,I0)') 'CU_ACT_FLAG=', MERGE(1,0,CU_ACT_FLAG(1,1))
  WRITE(*,'(A,ES23.15)') 'QFX=', QFX(1,1)
  WRITE(*,'(A,ES23.15)') 'HFX=', HFX(1,1)
  WRITE(*,'(A,ES23.15)') 'XLAND=', XLAND(1,1)
  CALL dump_col('T', T)
  CALL dump_col('QV', QV)
  CALL dump_col('QC', QC)
  CALL dump_col('QI', QI)
  CALL dump_col('P', PCPS)
  CALL dump_col('PI', PII)
  CALL dump_col('DZ', DZ8W)
  CALL dump_col('RHO', RHO)
  CALL dump_col('U', U)
  CALL dump_col('V', V)
  CALL dump_col('RTHCUTEN', RTHCUTEN)
  CALL dump_col('RQVCUTEN', RQVCUTEN)
  CALL dump_col('RQCCUTEN', RQCCUTEN)
  CALL dump_col('RQICUTEN', RQICUTEN)
  CALL dump_col('RUCUTEN', RUCUTEN)
  CALL dump_col('RVCUTEN', RVCUTEN)
  CALL dump_iface('P8W', P8W)
  CALL dump_iface('W', W)

CONTAINS

  REAL FUNCTION esat_water(t)
    REAL, INTENT(IN) :: t
    esat_water = 610.78 * EXP(17.269 * (t - 273.16) / (t - 35.86))
  END FUNCTION esat_water

  REAL FUNCTION qsat_mix(t, p)
    REAL, INTENT(IN) :: t, p
    REAL :: e
    e = MIN(0.5 * p, esat_water(t))
    qsat_mix = 0.622 * e / MAX(1.0, p - e)
  END FUNCTION qsat_mix

  SUBROUTINE build_sounding(cid, Tt, Qq, Qc, Qi, Pp, Pi8, Dz, Rr, Exn, Uu, Vv, Ww, &
                            Qfx2, Hfx2, Xland2, Hpbl2, Znu1)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(INOUT) :: Tt,Qq,Qc,Qi,Pp,Pi8,Dz,Rr,Exn,Uu,Vv,Ww
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(INOUT) :: Qfx2, Hfx2, Xland2, Hpbl2
    REAL, DIMENSION(kms:kme), INTENT(INOUT) :: Znu1
    REAL :: zint(KX+1), zmid(KX), psfc, ptop, dzk, pcur, pmid, theta_sfc
    REAL :: tsfc, zml, theta_lapse, rh_ml, rh_trop, qten_peak, pbl_frac, ushr
    REAL :: z, theta, temp, rh, qsat, tv
    INTEGER :: n

    ptop = 5000.0
    SELECT CASE (cid)
    CASE (1)
      psfc=100800.0; tsfc=302.0; zml=1250.0; theta_lapse=3.0E-3
      rh_ml=0.92; rh_trop=0.60; pbl_frac=0.65; ushr=8.0
      Qfx2(1,1)=2.0E-5; Hfx2(1,1)=120.0; Xland2(1,1)=1.0; Hpbl2(1,1)=1250.0
    CASE (2)
      psfc=100500.0; tsfc=298.0; zml=700.0; theta_lapse=6.0E-3
      rh_ml=0.86; rh_trop=0.25; pbl_frac=0.85; ushr=2.0
      Qfx2(1,1)=1.4E-4; Hfx2(1,1)=20.0; Xland2(1,1)=2.0; Hpbl2(1,1)=700.0
    CASE (3)
      psfc=100900.0; tsfc=304.0; zml=1500.0; theta_lapse=2.5E-3
      rh_ml=0.94; rh_trop=0.65; pbl_frac=0.55; ushr=14.0
      Qfx2(1,1)=2.5E-5; Hfx2(1,1)=150.0; Xland2(1,1)=1.0; Hpbl2(1,1)=1500.0
    CASE (4)
      psfc=100000.0; tsfc=299.0; zml=600.0; theta_lapse=8.0E-3
      rh_ml=0.78; rh_trop=0.18; pbl_frac=0.90; ushr=4.0
      Qfx2(1,1)=1.8E-4; Hfx2(1,1)=15.0; Xland2(1,1)=2.0; Hpbl2(1,1)=600.0
    CASE (5)
      psfc=100000.0; tsfc=287.0; zml=250.0; theta_lapse=9.5E-3
      rh_ml=0.35; rh_trop=0.08; pbl_frac=0.0; ushr=0.0
      Qfx2(1,1)=0.0; Hfx2(1,1)=0.0; Xland2(1,1)=1.0; Hpbl2(1,1)=250.0
    CASE DEFAULT
      psfc=100800.0; tsfc=301.0; zml=1000.0; theta_lapse=4.0E-3
      rh_ml=0.88; rh_trop=0.50; pbl_frac=0.7; ushr=5.0
      Qfx2(1,1)=2.0E-5; Hfx2(1,1)=100.0; Xland2(1,1)=1.0; Hpbl2(1,1)=1000.0
    END SELECT

    DO n=1,KX+1
      zint(n) = 21000.0 * (REAL(n-1) / REAL(KX)) ** 1.18
    END DO
    theta_sfc = tsfc * (P1000 / psfc) ** ROVCP
    pcur = psfc
    Pi8(1,1,1) = psfc
    DO n=1,KX
      dzk = zint(n+1) - zint(n)
      zmid(n) = 0.5 * (zint(n+1) + zint(n))
      z = zmid(n)
      pmid = MAX(ptop, pcur * EXP(-G * 0.5 * dzk / (R_D * tsfc)))
      IF (z <= zml) THEN
        theta = theta_sfc; rh = rh_ml
      ELSE
        theta = theta_sfc + theta_lapse * (z - zml)
        rh = rh_trop + (rh_ml - rh_trop) * EXP(-(z - zml) / 4200.0)
      END IF
      IF (cid == 4 .AND. z > 1500.0 .AND. z < 4500.0) THEN
        theta = theta + 6.0; rh = rh * 0.45
      END IF
      temp = theta * (pmid / P1000) ** ROVCP
      qsat = qsat_mix(temp, pmid)
      Qq(1,n,1) = MAX(1.0E-8, rh * qsat)
      Tt(1,n,1) = temp
      Pp(1,n,1) = pmid
      Exn(1,n,1) = (pmid / P1000) ** ROVCP
      Dz(1,n,1) = dzk
      tv = temp * (1.0 + 0.608 * Qq(1,n,1))
      Rr(1,n,1) = pmid / (R_D * tv)
      Qc(1,n,1) = 0.0
      Qi(1,n,1) = 0.0
      Uu(1,n,1) = ushr * (z / 12000.0)
      Vv(1,n,1) = 0.3 * ushr * (z / 12000.0)
      pcur = MAX(ptop, pcur * EXP(-G * dzk / (R_D * tsfc)))
      P8W(1,n+1,1) = pcur
      Znu1(n) = (pmid - ptop) / (psfc - ptop)
    END DO
    Ww(1,1,1) = 0.0
    DO n=1,KX
      z = zint(n+1)
      Ww(1,n+1,1) = pbl_frac * 0.05 * EXP(-((z - zml)/4000.0)**2)
    END DO
  END SUBROUTINE build_sounding

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk=1,KX
      WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name), '[', kk, ']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_iface(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk=1,KX+1
      WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name), '[', kk, ']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_iface

END PROGRAM ksas_oracle
