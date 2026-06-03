! =====================================================================
! v0.6.0 single-column modified-Tiedtke (WRF cu_physics=6) oracle driver.
!
! Drives the UNMODIFIED WRF module_cu_tiedtke.F on prescribed WRF-layout
! columns and dumps:
!   * full input state in WRF bottom-up orientation
!   * top-level WRF tendencies RTH/RQV/RQC/RQI/RU/RV CUTEN
!   * RAINCV, PRATEC, and a direct tiecnv KTYPE diagnostic for regime labels
!
! Usage: ./tiedtke_oracle <case_id>
! Output: flat key=value text parsed by dump_to_json.py.
! =====================================================================
PROGRAM tiedtke_oracle
  USE module_cu_tiedtke, ONLY : cu_tiedtke, tiecnv
  IMPLICIT NONE

  REAL, PARAMETER :: G = 9.81
  REAL, PARAMETER :: R_D = 287.0
  REAL, PARAMETER :: R_V = 461.6
  REAL, PARAMETER :: CP = 7.0 * R_D / 2.0
  REAL, PARAMETER :: P1000 = 1.0E5
  REAL, PARAMETER :: ROVCP = R_D / CP
  REAL, PARAMETER :: XLV = 2.5E6

  INTEGER, PARAMETER :: KX = 40
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,W,T,QV,QC,QI,PII,RHO
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: QVFTEN,QVPBLTEN,DZ8W,PCPS,P8W
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RTHCUTEN,RQVCUTEN,RQCCUTEN,RQICUTEN,RUCUTEN,RVCUTEN
  REAL, DIMENSION(ims:ime,jms:jme) :: RAINCV, PRATEC, QFX, XLAND
  LOGICAL, DIMENSION(ims:ime,jms:jme) :: CU_ACT_FLAG
  REAL, DIMENSION(kms:kme) :: ZNU

  REAL, DIMENSION(its:ite,kts:kte) :: u1,v1,t1,q1,q2,q3,q1b,q1bl,ght,omg,prsl
  REAL, DIMENSION(its:ite,kts:kte+1) :: prsi
  REAL, DIMENSION(kts:kte) :: sig1
  REAL, DIMENSION(its:ite) :: evap, rn_direct
  INTEGER, DIMENSION(its:ite) :: slimsk, ktype

  REAL :: DT, DX
  INTEGER :: STEPCU, ITIMESTEP
  INTEGER :: case_id, k, kk, km, zz
  CHARACTER(LEN=32) :: arg
  LOGICAL :: F_QV,F_QC,F_QR,F_QI,F_QS

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
  F_QV=.TRUE.; F_QC=.TRUE.; F_QR=.FALSE.; F_QI=.TRUE.; F_QS=.FALSE.

  U=0.0; V=0.0; W=0.0; T=0.0; QV=0.0; QC=0.0; QI=0.0; PII=1.0; RHO=0.0
  QVFTEN=0.0; QVPBLTEN=0.0; DZ8W=0.0; PCPS=0.0; P8W=0.0; ZNU=0.0
  RTHCUTEN=0.0; RQVCUTEN=0.0; RQCCUTEN=0.0; RQICUTEN=0.0; RUCUTEN=0.0; RVCUTEN=0.0
  RAINCV=0.0; PRATEC=0.0; QFX=0.0; XLAND=1.0; CU_ACT_FLAG=.TRUE.

  CALL build_sounding(case_id, T, QV, QC, QI, PCPS, P8W, DZ8W, RHO, PII, U, V, W, &
                      QVFTEN, QVPBLTEN, QFX, XLAND, ZNU)

  CALL cu_tiedtke( &
       DT, ITIMESTEP, STEPCU, &
       RAINCV, PRATEC, QFX, ZNU, &
       U, V, W, T, QV, QC, QI, PII, RHO, &
       QVFTEN, QVPBLTEN, &
       DZ8W, PCPS, P8W, XLAND, CU_ACT_FLAG, &
       ids,ide, jds,jde, kds,kde, &
       ims,ime, jms,jme, kms,kme, &
       its,ite, jts,jte, kts,kte, &
       RTHCUTEN,RQVCUTEN,RQCCUTEN,RQICUTEN, &
       RUCUTEN,RVCUTEN, &
       F_QV,F_QC,F_QR,F_QI,F_QS)

  ! Direct unmodified tiecnv call for the KTYPE regime diagnostic.
  CALL prepare_tiecnv_columns(U,V,T,QV,QC,QI,QVFTEN,QVPBLTEN,DZ8W,PCPS,P8W,RHO,W,XLAND,ZNU, &
                              u1,v1,t1,q1,q2,q3,q1b,q1bl,ght,omg,prsl,prsi,evap,slimsk,sig1)
  rn_direct=0.0; ktype=0
  CALL tiecnv(u1,v1,t1,q1,q2,q3,q1b,q1bl,ght,omg,prsl,prsi,evap, &
              rn_direct,slimsk,ktype,1,KX,KX+1,sig1,DT*STEPCU)

  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  WRITE(*,'(A,ES23.15)') 'DX=', DX
  WRITE(*,'(A,I0)') 'STEPCU=', STEPCU
  WRITE(*,'(A,I0)') 'ITIMESTEP=', ITIMESTEP
  WRITE(*,'(A,ES23.15)') 'RAINCV=', RAINCV(1,1)
  WRITE(*,'(A,ES23.15)') 'PRATEC=', PRATEC(1,1)
  WRITE(*,'(A,ES23.15)') 'RAINCV_DIRECT=', rn_direct(1) / REAL(STEPCU)
  WRITE(*,'(A,I0)') 'KTYPE=', ktype(1)
  WRITE(*,'(A,I0)') 'CU_ACT_FLAG=', MERGE(1,0,CU_ACT_FLAG(1,1))
  WRITE(*,'(A,ES23.15)') 'QFX=', QFX(1,1)
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
  CALL dump_col('QVFTEN', QVFTEN)
  CALL dump_col('QVPBLTEN', QVPBLTEN)
  CALL dump_col('RTHCUTEN', RTHCUTEN)
  CALL dump_col('RQVCUTEN', RQVCUTEN)
  CALL dump_col('RQCCUTEN', RQCCUTEN)
  CALL dump_col('RQICUTEN', RQICUTEN)
  CALL dump_col('RUCUTEN', RUCUTEN)
  CALL dump_col('RVCUTEN', RVCUTEN)
  CALL dump_iface('P8W', P8W)
  CALL dump_iface('W', W)
  CALL dump_znu('ZNU', ZNU)

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
                            Qvf, Qvbl, Qfx2, Xland2, Znu1)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(INOUT) :: Tt,Qq,Qc,Qi,Pp,Pi8,Dz,Rr,Exn,Uu,Vv,Ww,Qvf,Qvbl
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(INOUT) :: Qfx2, Xland2
    REAL, DIMENSION(kms:kme), INTENT(INOUT) :: Znu1
    REAL :: zint(KX+1), zmid(KX), psfc, ptop, dzk, pcur, pmid, theta_sfc
    REAL :: tsfc, zml, theta_lapse, rh_ml, rh_trop, qten_peak, pbl_frac, ushr
    REAL :: z, theta, temp, rh, qsat, tv, qten, sigma
    INTEGER :: n

    ptop = 5000.0
    SELECT CASE (cid)
    CASE (1) ! deep warm/moist, strong moisture convergence
      psfc=100800.0; tsfc=302.0; zml=1250.0; theta_lapse=3.0E-3
      rh_ml=0.92; rh_trop=0.60; qten_peak=3.0E-8; pbl_frac=0.65; ushr=8.0
      Qfx2(1,1)=2.0E-5; Xland2(1,1)=1.0
    CASE (2) ! shallow marine cumulus
      psfc=100500.0; tsfc=298.0; zml=700.0; theta_lapse=6.0E-3
      rh_ml=0.86; rh_trop=0.25; qten_peak=1.2E-8; pbl_frac=0.85; ushr=2.0
      Qfx2(1,1)=1.4E-4; Xland2(1,1)=2.0
    CASE (3) ! vigorous deep tropical
      psfc=100900.0; tsfc=304.0; zml=1500.0; theta_lapse=2.5E-3
      rh_ml=0.94; rh_trop=0.65; qten_peak=4.5E-8; pbl_frac=0.55; ushr=14.0
      Qfx2(1,1)=2.5E-5; Xland2(1,1)=1.0
    CASE (4) ! capped shallow/non-precipitating
      psfc=100000.0; tsfc=299.0; zml=600.0; theta_lapse=8.0E-3
      rh_ml=0.78; rh_trop=0.18; qten_peak=8.0E-9; pbl_frac=0.90; ushr=4.0
      Qfx2(1,1)=1.8E-4; Xland2(1,1)=2.0
    CASE (5) ! stable dry non-triggering
      psfc=100000.0; tsfc=287.0; zml=250.0; theta_lapse=9.5E-3
      rh_ml=0.35; rh_trop=0.08; qten_peak=0.0; pbl_frac=0.0; ushr=0.0
      Qfx2(1,1)=0.0; Xland2(1,1)=1.0
    CASE DEFAULT
      psfc=100800.0; tsfc=301.0; zml=1000.0; theta_lapse=4.0E-3
      rh_ml=0.88; rh_trop=0.50; qten_peak=2.0E-8; pbl_frac=0.7; ushr=5.0
      Qfx2(1,1)=2.0E-5; Xland2(1,1)=1.0
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
        theta = theta_sfc
        rh = rh_ml
      ELSE
        theta = theta_sfc + theta_lapse * (z - zml)
        rh = rh_trop + (rh_ml - rh_trop) * EXP(-(z - zml) / 4200.0)
      END IF
      IF (cid == 4 .AND. z > 1500.0 .AND. z < 4500.0) THEN
        theta = theta + 6.0
        rh = rh * 0.45
      END IF
      temp = theta * (pmid / P1000) ** ROVCP
      qsat = qsat_mix(temp, pmid)
      Qq(1,n,1) = MAX(1.0E-7, MIN(0.024, rh * qsat))
      Tt(1,n,1) = temp
      Qc(1,n,1) = 0.0
      Qi(1,n,1) = 0.0
      tv = temp * (1.0 + 0.608 * Qq(1,n,1))
      Rr(1,n,1) = pmid / (R_D * tv)
      Pp(1,n,1) = pmid
      Exn(1,n,1) = (pmid / P1000) ** ROVCP
      Dz(1,n,1) = dzk
      Uu(1,n,1) = ushr * z / 12000.0
      Vv(1,n,1) = 2.0 * SIN(z / 3000.0)
      Ww(1,n,1) = 0.15 * EXP(-((z - 1700.0) / 1400.0) ** 2)
      sigma = MAX(0.0, MIN(1.0, (pmid - ptop) / (psfc - ptop)))
      Znu1(n) = sigma
      qten = qten_peak * EXP(-(z / 2500.0) ** 2)
      Qvbl(1,n,1) = pbl_frac * qten
      Qvf(1,n,1) = (1.0 - pbl_frac) * qten
      Pi8(1,n+1,1) = MAX(ptop, pcur - Rr(1,n,1) * G * dzk)
      pcur = Pi8(1,n+1,1)
    END DO
    Ww(1,KX+1,1) = 0.0
    Znu1(KX+1) = 0.0
    Pp(1,KX+1,1) = Pp(1,KX,1)
    Dz(1,KX+1,1) = Dz(1,KX,1)
    Rr(1,KX+1,1) = Rr(1,KX,1)
    Exn(1,KX+1,1) = Exn(1,KX,1)
  END SUBROUTINE build_sounding

  SUBROUTINE prepare_tiecnv_columns(Uu,Vv,Tt,Qq,Qc,Qi,Qvf,Qvbl,Dz,Pp,Pi8,Rr,Ww,Xland2,Znu1, &
                                    pu,pv,pt,pqv,pqc,pqi,pqvf,pqvbl,poz,pomg,pap,paph,evap,slimsk,sig1)
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: Uu,Vv,Tt,Qq,Qc,Qi,Qvf,Qvbl,Dz,Pp,Pi8,Rr,Ww
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: Xland2
    REAL, DIMENSION(kms:kme), INTENT(IN) :: Znu1
    REAL, DIMENSION(its:ite,kts:kte), INTENT(OUT) :: pu,pv,pt,pqv,pqc,pqi,pqvf,pqvbl,poz,pomg,pap
    REAL, DIMENSION(its:ite,kts:kte+1), INTENT(OUT) :: paph
    REAL, DIMENSION(its:ite), INTENT(OUT) :: evap
    INTEGER, DIMENSION(its:ite), INTENT(OUT) :: slimsk
    REAL, DIMENSION(kts:kte), INTENT(OUT) :: sig1
    REAL :: zi(kts:kte), zl(kts:kte), dot(kts:kte)
    INTEGER :: n, kk2

    zi(kts)=0.0
    DO n=kts+1,kte
      zi(n)=zi(n-1)+Dz(1,n-1,1)
    END DO
    DO n=kts+1,kte
      zl(n-1)=0.5*(zi(n)+zi(n-1))
    END DO
    zl(kte)=2.0*zi(kte)-zl(kte-1)
    DO n=kts,kte
      dot(n)=-0.5*G*Rr(1,n,1)*(Ww(1,n,1)+Ww(1,n+1,1))
    END DO
    slimsk(1)=INT(ABS(Xland2(1,1)-2.0))
    evap(1)=QFX(1,1)
    DO n=kts,kte
      kk2=kte+1-n
      pu(1,kk2)=Uu(1,n,1)
      pv(1,kk2)=Vv(1,n,1)
      pt(1,kk2)=Tt(1,n,1)
      pqv(1,kk2)=Qq(1,n,1)
      pqvf(1,kk2)=Qvf(1,n,1)
      pqvbl(1,kk2)=Qvbl(1,n,1)
      pqc(1,kk2)=Qc(1,n,1)
      pqi(1,kk2)=Qi(1,n,1)
      pomg(1,kk2)=dot(n)
      poz(1,kk2)=zl(n)
      pap(1,kk2)=Pp(1,n,1)
      sig1(kk2)=Znu1(n)
    END DO
    DO n=kts,kte+1
      kk2=kte+2-n
      paph(1,kk2)=Pi8(1,n,1)
    END DO
  END SUBROUTINE prepare_tiecnv_columns

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: n
    DO n=kts,kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',n,']=', arr(1,n,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_iface(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: n
    DO n=kts,kte+1
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',n,']=', arr(1,n,1)
    END DO
  END SUBROUTINE dump_iface

  SUBROUTINE dump_znu(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(kms:kme), INTENT(IN) :: arr
    INTEGER :: n
    DO n=kts,kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',n,']=', arr(n)
    END DO
  END SUBROUTINE dump_znu

END PROGRAM tiedtke_oracle
