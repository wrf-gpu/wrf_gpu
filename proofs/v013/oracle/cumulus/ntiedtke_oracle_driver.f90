! =====================================================================
! v0.13 single-column New-Tiedtke (WRF cu_physics=16) oracle driver.
!
! Drives the UNMODIFIED WRF module_cu_ntiedtke.F (cu_ntiedtke_driver,
! which wraps the CCPP-style core cu_ntiedtke.F90) on prescribed
! WRF-layout columns and dumps:
!   * full input state in WRF bottom-up orientation
!   * the WRF output tendencies RTH/RQV/RQC/RQI/RU/RV CUTEN
!   * RAINCV, PRATEC
!
! Usage: ./ntiedtke_oracle <case_id>
! Output: flat key=value text parsed by dump_to_json.py.
! The sounding generator mirrors the v0.6.0 modified-Tiedtke oracle so
! the JAX kernel is validated on the same physical regimes (deep/shallow/
! capped/stable), but driven through the New-Tiedtke code path.
! =====================================================================
PROGRAM ntiedtke_oracle
  USE module_cu_ntiedtke, ONLY : cu_ntiedtke_driver
  IMPLICIT NONE

  INTEGER, PARAMETER :: KP = SELECTED_REAL_KIND(15)   ! kind_phys = double in WRF MMM physics
  REAL(KP), PARAMETER :: G = 9.81_KP
  REAL(KP), PARAMETER :: R_D = 287.0_KP
  REAL(KP), PARAMETER :: R_V = 461.6_KP
  REAL(KP), PARAMETER :: CP = 7.0_KP * R_D / 2.0_KP
  REAL(KP), PARAMETER :: P1000 = 1.0E5_KP
  REAL(KP), PARAMETER :: ROVCP = R_D / CP
  REAL(KP), PARAMETER :: XLV = 2.5E6_KP
  REAL(KP), PARAMETER :: XLS = 2.85E6_KP
  REAL(KP), PARAMETER :: XLF = 3.5E5_KP

  INTEGER, PARAMETER :: KX = 40
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL(KP), DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,W,T,QV,QC,QI,PII,RHO
  REAL(KP), DIMENSION(ims:ime,kms:kme,jms:jme) :: QVFTEN,THFTEN,DZ8W,PCPS,P8W
  REAL(KP), DIMENSION(ims:ime,kms:kme,jms:jme) :: RTHCUTEN,RQVCUTEN,RQCCUTEN,RQICUTEN,RUCUTEN,RVCUTEN
  REAL(KP), DIMENSION(ims:ime,jms:jme) :: RAINCV, PRATEC, QFX, HFX, XLAND, DX2
  LOGICAL, DIMENSION(ims:ime,jms:jme) :: CU_ACT_FLAG

  REAL(KP) :: DT, DX
  INTEGER :: STEPCU, ITIMESTEP
  INTEGER :: case_id, k
  CHARACTER(LEN=32) :: arg
  CHARACTER(LEN=256) :: errmsg
  INTEGER :: errflg
  LOGICAL :: F_QV,F_QC,F_QR,F_QI,F_QS

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DX = 9000.0_KP
  DT = 54.0_KP
  STEPCU = 5
  ITIMESTEP = 2
  F_QV=.TRUE.; F_QC=.TRUE.; F_QR=.FALSE.; F_QI=.TRUE.; F_QS=.FALSE.

  U=0.0_KP; V=0.0_KP; W=0.0_KP; T=0.0_KP; QV=0.0_KP; QC=0.0_KP; QI=0.0_KP; PII=1.0_KP; RHO=0.0_KP
  QVFTEN=0.0_KP; THFTEN=0.0_KP; DZ8W=0.0_KP; PCPS=0.0_KP; P8W=0.0_KP
  RTHCUTEN=0.0_KP; RQVCUTEN=0.0_KP; RQCCUTEN=0.0_KP; RQICUTEN=0.0_KP; RUCUTEN=0.0_KP; RVCUTEN=0.0_KP
  RAINCV=0.0_KP; PRATEC=0.0_KP; QFX=0.0_KP; HFX=0.0_KP; XLAND=1.0_KP; CU_ACT_FLAG=.TRUE.
  DX2=DX

  CALL build_sounding(case_id, T, QV, QC, QI, PCPS, P8W, DZ8W, RHO, PII, U, V, W, &
                      QVFTEN, THFTEN, QFX, HFX, XLAND)

  CALL cu_ntiedtke_driver( &
       dt=DT, itimestep=ITIMESTEP, stepcu=STEPCU, &
       raincv=RAINCV, pratec=PRATEC, qfx=QFX, hfx=HFX, &
       u3d=U, v3d=V, w=W, t3d=T, qv3d=QV, qc3d=QC, qi3d=QI, pi3d=PII, rho3d=RHO, &
       qvften=QVFTEN, thften=THFTEN, &
       dz8w=DZ8W, pcps=PCPS, p8w=P8W, xland=XLAND, cu_act_flag=CU_ACT_FLAG, dx=DX2, &
       f_qv=F_QV, f_qc=F_QC, f_qr=F_QR, f_qi=F_QI, f_qs=F_QS, &
       grav=G, xlf=XLF, xls=XLS, xlv=XLV, rd=R_D, rv=R_V, cp=CP, &
       rthcuten=RTHCUTEN, rqvcuten=RQVCUTEN, rqccuten=RQCCUTEN, rqicuten=RQICUTEN, &
       rucuten=RUCUTEN, rvcuten=RVCUTEN, &
       ids=ids,ide=ide, jds=jds,jde=jde, kds=kds,kde=kde, &
       ims=ims,ime=ime, jms=jms,jme=jme, kms=kms,kme=kme, &
       its=its,ite=ite, jts=jts,jte=jte, kts=kts,kte=kte, &
       errmsg=errmsg, errflg=errflg)

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
  CALL dump_col('QVFTEN', QVFTEN)
  CALL dump_col('THFTEN', THFTEN)
  CALL dump_col('RTHCUTEN', RTHCUTEN)
  CALL dump_col('RQVCUTEN', RQVCUTEN)
  CALL dump_col('RQCCUTEN', RQCCUTEN)
  CALL dump_col('RQICUTEN', RQICUTEN)
  CALL dump_col('RUCUTEN', RUCUTEN)
  CALL dump_col('RVCUTEN', RVCUTEN)
  CALL dump_iface('P8W', P8W)
  CALL dump_iface('W', W)

CONTAINS

  REAL(KP) FUNCTION esat_water(t)
    REAL(KP), INTENT(IN) :: t
    esat_water = 610.78_KP * EXP(17.269_KP * (t - 273.16_KP) / (t - 35.86_KP))
  END FUNCTION esat_water

  REAL(KP) FUNCTION qsat_mix(t, p)
    REAL(KP), INTENT(IN) :: t, p
    REAL(KP) :: e
    e = MIN(0.5_KP * p, esat_water(t))
    qsat_mix = 0.622_KP * e / MAX(1.0_KP, p - e)
  END FUNCTION qsat_mix

  SUBROUTINE build_sounding(cid, Tt, Qq, Qc, Qi, Pp, Pi8, Dz, Rr, Exn, Uu, Vv, Ww, &
                            Qvf, Thf, Qfx2, Hfx2, Xland2)
    INTEGER, INTENT(IN) :: cid
    REAL(KP), DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(INOUT) :: Tt,Qq,Qc,Qi,Pp,Pi8,Dz,Rr,Exn,Uu,Vv,Ww,Qvf,Thf
    REAL(KP), DIMENSION(ims:ime,jms:jme), INTENT(INOUT) :: Qfx2, Hfx2, Xland2
    REAL(KP) :: zint(KX+1), zmid(KX), psfc, ptop, dzk, pcur, pmid, theta_sfc
    REAL(KP) :: tsfc, zml, theta_lapse, rh_ml, rh_trop, qten_peak, pbl_frac, ushr
    REAL(KP) :: z, theta, temp, rh, qsat, tv, qten, sigma
    INTEGER :: n

    ptop = 5000.0_KP
    SELECT CASE (cid)
    CASE (1) ! deep warm/moist, strong moisture convergence
      psfc=100800.0_KP; tsfc=302.0_KP; zml=1250.0_KP; theta_lapse=3.0E-3_KP
      rh_ml=0.92_KP; rh_trop=0.60_KP; qten_peak=3.0E-8_KP; pbl_frac=0.65_KP; ushr=8.0_KP
      Qfx2(1,1)=2.0E-5_KP; Hfx2(1,1)=120.0_KP; Xland2(1,1)=1.0_KP
    CASE (2) ! shallow marine cumulus
      psfc=100500.0_KP; tsfc=298.0_KP; zml=700.0_KP; theta_lapse=6.0E-3_KP
      rh_ml=0.86_KP; rh_trop=0.25_KP; qten_peak=1.2E-8_KP; pbl_frac=0.85_KP; ushr=2.0_KP
      Qfx2(1,1)=1.4E-4_KP; Hfx2(1,1)=20.0_KP; Xland2(1,1)=2.0_KP
    CASE (3) ! vigorous deep tropical
      psfc=100900.0_KP; tsfc=304.0_KP; zml=1500.0_KP; theta_lapse=2.5E-3_KP
      rh_ml=0.94_KP; rh_trop=0.65_KP; qten_peak=4.5E-8_KP; pbl_frac=0.55_KP; ushr=14.0_KP
      Qfx2(1,1)=2.5E-5_KP; Hfx2(1,1)=150.0_KP; Xland2(1,1)=1.0_KP
    CASE (4) ! capped shallow/non-precipitating
      psfc=100000.0_KP; tsfc=299.0_KP; zml=600.0_KP; theta_lapse=8.0E-3_KP
      rh_ml=0.78_KP; rh_trop=0.18_KP; qten_peak=8.0E-9_KP; pbl_frac=0.90_KP; ushr=4.0_KP
      Qfx2(1,1)=1.8E-4_KP; Hfx2(1,1)=15.0_KP; Xland2(1,1)=2.0_KP
    CASE (5) ! stable dry non-triggering
      psfc=100000.0_KP; tsfc=287.0_KP; zml=250.0_KP; theta_lapse=9.5E-3_KP
      rh_ml=0.35_KP; rh_trop=0.08_KP; qten_peak=0.0_KP; pbl_frac=0.0_KP; ushr=0.0_KP
      Qfx2(1,1)=0.0_KP; Hfx2(1,1)=0.0_KP; Xland2(1,1)=1.0_KP
    CASE DEFAULT
      psfc=100800.0_KP; tsfc=301.0_KP; zml=1000.0_KP; theta_lapse=4.0E-3_KP
      rh_ml=0.88_KP; rh_trop=0.50_KP; qten_peak=2.0E-8_KP; pbl_frac=0.7_KP; ushr=5.0_KP
      Qfx2(1,1)=2.0E-5_KP; Hfx2(1,1)=100.0_KP; Xland2(1,1)=1.0_KP
    END SELECT

    DO n=1,KX+1
      zint(n) = 21000.0_KP * (REAL(n-1,KP) / REAL(KX,KP)) ** 1.18_KP
    END DO
    theta_sfc = tsfc * (P1000 / psfc) ** ROVCP
    pcur = psfc
    Pi8(1,1,1) = psfc
    DO n=1,KX
      dzk = zint(n+1) - zint(n)
      zmid(n) = 0.5_KP * (zint(n+1) + zint(n))
      z = zmid(n)
      pmid = MAX(ptop, pcur * EXP(-G * 0.5_KP * dzk / (R_D * tsfc)))
      IF (z <= zml) THEN
        theta = theta_sfc
        rh = rh_ml
      ELSE
        theta = theta_sfc + theta_lapse * (z - zml)
        rh = rh_trop + (rh_ml - rh_trop) * EXP(-(z - zml) / 4200.0_KP)
      END IF
      IF (cid == 4 .AND. z > 1500.0_KP .AND. z < 4500.0_KP) THEN
        theta = theta + 6.0_KP
        rh = rh * 0.45_KP
      END IF
      temp = theta * (pmid / P1000) ** ROVCP
      qsat = qsat_mix(temp, pmid)
      Qq(1,n,1) = MAX(1.0E-8_KP, rh * qsat)
      Tt(1,n,1) = temp
      Pp(1,n,1) = pmid
      Exn(1,n,1) = (pmid / P1000) ** ROVCP
      Dz(1,n,1) = dzk
      tv = temp * (1.0_KP + 0.608_KP * Qq(1,n,1))
      Rr(1,n,1) = pmid / (R_D * tv)
      Qc(1,n,1) = 0.0_KP
      Qi(1,n,1) = 0.0_KP
      ! advective + PBL forcing: a moisture-convergence / heating bump in the lower troposphere
      sigma = (z - 0.5_KP*zml) / 3000.0_KP
      qten = qten_peak * EXP(-0.5_KP * sigma * sigma)
      Qvf(1,n,1) = qten
      Thf(1,n,1) = qten * 2.0_KP * (P1000/pmid)**ROVCP   ! a small theta-forcing proxy (K/s)
      ! winds: linear shear with height
      Uu(1,n,1) = ushr * (z / 12000.0_KP)
      Vv(1,n,1) = 0.3_KP * ushr * (z / 12000.0_KP)
      ! pcur tracks the full-level pressure marching upward
      pcur = MAX(ptop, pcur * EXP(-G * dzk / (R_D * tsfc)))
      P8W(1,n+1,1) = pcur
    END DO
    ! interface vertical velocity: a weak updraft proxy peaking mid-PBL (m/s),
    ! W is on full (interface) levels (KX+1).
    Ww(1,1,1) = 0.0_KP
    DO n=1,KX
      z = zint(n+1)
      Ww(1,n+1,1) = pbl_frac * 0.05_KP * EXP(-((z - zml)/4000.0_KP)**2)
    END DO
  END SUBROUTINE build_sounding

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(KP), DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk=1,KX
      WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name), '[', kk, ']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_iface(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(KP), DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk=1,KX+1
      WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name), '[', kk, ']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_iface

END PROGRAM ntiedtke_oracle
